"""Write and search memories. Every function expects a connection already inside user_session():
there is no WHERE user_id = ... anywhere here, the RLS policy does that filtering."""

from __future__ import annotations

import re
import uuid

import psycopg

from .embed import Embedder, to_pgvector
from .logs import timed
from .rrf import rrf

CANDIDATES = 20  # each retriever hands RRF its top 20; callers keep the top 5
MODES = ("hybrid", "vector", "fts")

_WORD = re.compile(r"[a-z0-9]+")


def add_memory(
    conn: psycopg.Connection,
    embedder: Embedder,
    user_id: str,
    text: str,
    importance: float = 0.5,
    memory_id: str | None = None,
) -> str:
    """user_id must equal the session's app.user_id, or the policy's WITH CHECK rejects the row."""
    memory_id = memory_id or uuid.uuid4().hex
    [vec] = embedder.embed_passages([text])
    with timed("db.insert", table="memories"):
        conn.execute(
            "INSERT INTO memories (id, user_id, text, embedding, importance) VALUES (%s, %s, %s, %s::vector, %s)",
            (memory_id, user_id, text, to_pgvector(vec), importance),
        )
    return memory_id


def vector_search(conn: psycopg.Connection, embedder: Embedder, query: str, limit: int = CANDIDATES) -> list[str]:
    qv = to_pgvector(embedder.embed_query(query))
    with timed("retrieve", method="vector", limit=limit) as log:
        rows = conn.execute(
            "SELECT id FROM memories ORDER BY embedding <=> %s::vector LIMIT %s",  # <=> cosine distance, HNSW
            (qv, limit),
        ).fetchall()
        log["hits"] = len(rows)
    return [r[0] for r in rows]


def or_tsquery(query: str) -> str:
    """'when is the dentist?' -> 'when | is | the | dentist'.

    plainto_tsquery ANDs every word, so a natural-language question almost never matches.
    OR-ing them and letting ts_rank_cd order the hits behaves more like BM25.
    Stop words are dropped by to_tsquery('english', ...) itself.
    """
    return " | ".join(_WORD.findall(query.lower()))


def fts_search(conn: psycopg.Connection, query: str, limit: int = CANDIDATES) -> list[str]:
    tq = or_tsquery(query)
    if not tq:
        return []
    with timed("retrieve", method="fts", limit=limit) as log:
        rows = conn.execute(
            """
            SELECT id FROM memories, to_tsquery('english', %s) q
            WHERE tsv @@ q                                   -- @@ uses the GIN index
            ORDER BY ts_rank_cd(tsv, q) DESC, id
            LIMIT %s
            """,
            (tq, limit),
        ).fetchall()
        log["hits"] = len(rows)
    return [r[0] for r in rows]


def search(
    conn: psycopg.Connection, embedder: Embedder, query: str, mode: str = "hybrid", limit: int = 5, k: int = 60
) -> list[str]:
    if mode == "vector":
        return vector_search(conn, embedder, query)[:limit]
    if mode == "fts":
        return fts_search(conn, query)[:limit]
    if mode != "hybrid":
        raise ValueError(f"mode must be one of {MODES}")
    vec = vector_search(conn, embedder, query)
    fts = fts_search(conn, query)
    with timed("fuse", method="rrf", k=k):
        fused = rrf([vec, fts], k=k)
    return [doc_id for doc_id, _ in fused[:limit]]
