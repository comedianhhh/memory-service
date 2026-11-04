"""Three ways to search one user's memories: vector, full-text, and RRF of the two."""

from __future__ import annotations

import re

import psycopg

from .embed import embed_query, to_pgvector
from .rrf import rrf

CANDIDATES = 20  # each retriever hands RRF its top 20; we report top 5 at the end

_WORD = re.compile(r"[a-z0-9]+")


def vector_search(conn: psycopg.Connection, user_id: str, query: str, limit: int = CANDIDATES) -> list[str]:
    qv = to_pgvector(embed_query(query))
    rows = conn.execute(
        """
        SELECT id FROM memories
        WHERE user_id = %s
        ORDER BY embedding <=> %s::vector      -- <=> is cosine distance, uses the HNSW index
        LIMIT %s
        """,
        (user_id, qv, limit),
    ).fetchall()
    return [r[0] for r in rows]


def or_tsquery(query: str) -> str:
    """'when is the dentist?' -> 'when | is | the | dentist'.

    plainto_tsquery ANDs every word, so a natural-language question almost never matches.
    OR-ing them and letting ts_rank_cd order the hits behaves more like BM25.
    Stop words are dropped by to_tsquery('english', ...) itself.
    """
    return " | ".join(_WORD.findall(query.lower()))


def fts_search(conn: psycopg.Connection, user_id: str, query: str, limit: int = CANDIDATES) -> list[str]:
    tq = or_tsquery(query)
    if not tq:
        return []
    rows = conn.execute(
        """
        SELECT id FROM memories, to_tsquery('english', %s) q
        WHERE user_id = %s AND tsv @@ q                -- @@ uses the GIN index
        ORDER BY ts_rank_cd(tsv, q) DESC, id
        LIMIT %s
        """,
        (tq, user_id, limit),
    ).fetchall()
    return [r[0] for r in rows]


def hybrid_search(conn: psycopg.Connection, user_id: str, query: str, limit: int = 5, k: int = 60) -> list[str]:
    fused = rrf([vector_search(conn, user_id, query), fts_search(conn, user_id, query)], k=k)
    return [doc_id for doc_id, _ in fused[:limit]]
