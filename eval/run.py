"""recall@5, MRR and latency for vector, full-text, RRF, and RRF + cross-encoder rerank.

    .venv/Scripts/python eval/run.py            # misses only
    .venv/Scripts/python eval/run.py --all      # every query
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memory_service.db import connect, user_session  # noqa: E402
from memory_service.embed import FastEmbedder  # noqa: E402
from memory_service.rerank import rerank  # noqa: E402
from memory_service.rrf import rrf  # noqa: E402
from memory_service.search import fts_search, vector_search  # noqa: E402
from scripts.seed import DEMO_USER  # noqa: E402

K_EVAL = 5
METHODS = ("vector", "fts", "rrf", "rrf+rerank")


def recall_at(ranked: list[str], relevant: set[str], k: int = K_EVAL) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for pos, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / pos
    return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    lines = (ROOT / "eval" / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    queries = [json.loads(line) for line in lines if line.strip()]
    embedder = FastEmbedder()
    totals = {m: {"recall": 0.0, "mrr": 0.0, "ms": []} for m in METHODS}
    per_query = []
    with connect() as conn, user_session(conn, DEMO_USER):
        texts = dict(conn.execute("SELECT id, text FROM memories").fetchall())
        embedder.embed_query("warm up")  # model load shouldn't count as query latency
        rerank("warm up", [("x", "warm up")])
        for q in queries:
            rel = set(q["relevant"])
            t0 = time.perf_counter()
            vec = vector_search(conn, embedder, q["query"])
            t1 = time.perf_counter()
            fts = fts_search(conn, q["query"])
            t2 = time.perf_counter()
            fused = [d for d, _ in rrf([vec, fts])]
            t3 = time.perf_counter()
            reranked = rerank(q["query"], [(d, texts[d]) for d in fused[:20]])
            t4 = time.perf_counter()
            ms = {"vector": t1 - t0, "fts": t2 - t1, "rrf": t3 - t0, "rrf+rerank": t4 - t0}
            row = {"id": q["id"], "query": q["query"]}
            for m, ranked in zip(METHODS, (vec, fts, fused, reranked), strict=True):
                row[m] = recall_at(ranked, rel)
                totals[m]["recall"] += row[m]
                totals[m]["mrr"] += reciprocal_rank(ranked, rel)
                totals[m]["ms"].append(ms[m] * 1000)
            per_query.append(row)

    n = len(queries)
    shown = per_query if args.all else [r for r in per_query if min(r[m] for m in METHODS if m != "fts") < 1]
    print(f"{'id':4} {'query':40} {'vec':>5} {'fts':>5} {'rrf':>5} {'rr+rk':>5}")
    for row in shown:
        print(f"{row['id']:4} {row['query'][:40]:40} " + " ".join(f"{row[m]:5.2f}" for m in METHODS))
    if not args.all:
        print(f"(only queries where vector / rrf / rerank missed something; --all for every one)")
    print()
    print(f"| method | recall@{K_EVAL} | MRR | p50 ms |")
    print("|---|---|---|---|")
    for m in METHODS:
        t = totals[m]
        print(f"| {m} | {t['recall'] / n:.3f} | {t['mrr'] / n:.3f} | {statistics.median(t['ms']):.1f} |")
    print(f"\n{n} queries, {len(texts)} memories, candidates per retriever = 20, RRF k = 60, rerank top 20")


if __name__ == "__main__":
    main()
