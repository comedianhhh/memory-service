"""How does chunking change retrieval on real-ish conversations?

Five synthetic conversations (data/conversations/, 21-24 turns each) are stored four ways, each
under its own user (RLS keeps them apart):

    per_turn   every line is one memory
    words_40   chunk_text(max_words=40)   ~ 3-5 turns per chunk
    words_80   chunk_text(max_words=80)   (the API default)
    words_160  chunk_text(max_words=160)

Labels are facts, not memory ids, because ids change with the chunking. A fact is a list of phrases
that must ALL appear in one retrieved chunk: "what's your partner's name?" + "Theo." only counts if
question and answer land together. A bare "Noor: Theo." chunk can't be found by "partner's name",
and even if it were, it wouldn't tell the model whose name it is.

    .venv/Scripts/python eval/convo_run.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import uuid

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memory_service.db import connect, user_session  # noqa: E402
from memory_service.embed import FastEmbedder, to_pgvector  # noqa: E402
from memory_service.chunk import chunk_text  # noqa: E402
from memory_service.rerank import rerank  # noqa: E402
from memory_service.rrf import rrf  # noqa: E402
from memory_service.search import fts_search, vector_search  # noqa: E402

K = 5
NS = uuid.UUID("c0000000-0000-0000-0000-000000000000")
STRATEGIES: dict[str, int | None] = {"per_turn": None, "words_40": 40, "words_80": 80, "words_160": 160}


def conversations() -> list[str]:
    return [p.read_text(encoding="utf-8") for p in sorted((ROOT / "data" / "conversations").glob("*.txt"))]


def chunks_for(strategy: str, convos: list[str]) -> list[str]:
    max_words = STRATEGIES[strategy]
    out: list[str] = []
    for text in convos:
        if max_words is None:
            out.extend(line.strip() for line in text.splitlines() if line.strip())
        else:
            out.extend(chunk_text(text, max_words=max_words))
    return out


def load(conn, embedder, user: str, chunks: list[str]) -> None:
    vectors = embedder.embed_passages(chunks)
    with user_session(conn, user):
        conn.execute("DELETE FROM memories")  # RLS: only this strategy's user
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO memories (id, user_id, text, embedding, chunk_index) VALUES (%s, %s, %s, %s::vector, %s)",
                [(f"{user[:8]}-{i}", user, c, to_pgvector(v), i) for i, (c, v) in enumerate(zip(chunks, vectors, strict=True))],
            )


def fact_hit(fact: list[str], text: str) -> bool:
    return all(p in text for p in fact)


def score(ranked_texts: list[str], facts: list[list[str]]) -> tuple[float, float]:
    top = ranked_texts[:K]
    recall = sum(any(fact_hit(f, t) for t in top) for f in facts) / len(facts)
    rr = 0.0
    for pos, t in enumerate(ranked_texts, start=1):
        if any(fact_hit(f, t) for f in facts):
            rr = 1.0 / pos
            break
    return recall, rr


def main() -> None:
    convos = conversations()
    queries = [json.loads(line) for line in (ROOT / "eval" / "convo_queries.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    embedder = FastEmbedder()
    results = []
    misses: dict[str, list[str]] = {}
    with connect() as conn:
        for strategy in STRATEGIES:
            user = str(uuid.uuid5(NS, strategy))
            chunks = chunks_for(strategy, convos)
            load(conn, embedder, user, chunks)
            tot = {"rrf_r": 0.0, "rrf_mrr": 0.0, "rk_r": 0.0, "rk_mrr": 0.0, "words": 0}
            with user_session(conn, user):
                texts = dict(conn.execute("SELECT id, text FROM memories").fetchall())
                for q in queries:
                    fused = [d for d, _ in rrf([vector_search(conn, embedder, q["query"]), fts_search(conn, q["query"])])]
                    fused_texts = [texts[d] for d in fused]
                    reranked = [texts[d] for d in rerank(q["query"], [(d, texts[d]) for d in fused[:20]])]
                    r1, m1 = score(fused_texts, q["facts"])
                    r2, m2 = score(reranked, q["facts"])
                    tot["rrf_r"] += r1
                    tot["rrf_mrr"] += m1
                    tot["rk_r"] += r2
                    tot["rk_mrr"] += m2
                    tot["words"] += sum(len(t.split()) for t in reranked[:K])
                    if r2 < 1:
                        misses.setdefault(strategy, []).append(q["id"])
            n = len(queries)
            results.append(
                (strategy, len(chunks), sum(len(c.split()) for c in chunks) / len(chunks),
                 tot["rrf_r"] / n, tot["rrf_mrr"] / n, tot["rk_r"] / n, tot["rk_mrr"] / n, tot["words"] / n)
            )

    print(f"| chunking | chunks | avg words | RRF recall@{K} | RRF MRR | +rerank recall@{K} | +rerank MRR | words sent to LLM (top {K}) |")
    print("|---|---|---|---|---|---|---|---|")
    for s, n_chunks, avg, r1, m1, r2, m2, w in results:
        print(f"| {s} | {n_chunks} | {avg:.0f} | {r1:.3f} | {m1:.3f} | {r2:.3f} | {m2:.3f} | {w:.0f} |")
    print(f"\n{len(queries)} queries over {len(convos)} conversations")
    for s, ids in misses.items():
        print(f"missed with rerank, {s}: {' '.join(ids)}")


if __name__ == "__main__":
    main()
