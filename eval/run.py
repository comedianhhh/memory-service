"""recall@5 and MRR for vector-only, full-text-only, and RRF on eval/queries.jsonl."""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memory_service.db import connect, user_session  # noqa: E402
from memory_service.embed import FastEmbedder  # noqa: E402
from memory_service.rrf import rrf  # noqa: E402
from memory_service.search import fts_search, vector_search  # noqa: E402
from scripts.seed import DEMO_USER  # noqa: E402

K_EVAL = 5
METHODS = ("vector", "fts", "rrf")


def recall_at(ranked: list[str], relevant: set[str], k: int = K_EVAL) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for pos, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / pos
    return 0.0


def main() -> None:
    lines = (ROOT / "eval" / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    queries = [json.loads(line) for line in lines if line.strip()]
    totals = {m: {"recall": 0.0, "mrr": 0.0} for m in METHODS}
    per_query = []
    embedder = FastEmbedder()
    with connect() as conn, user_session(conn, DEMO_USER):
        for q in queries:
            rel = set(q["relevant"])
            vec = vector_search(conn, embedder, q["query"])
            fts = fts_search(conn, q["query"])
            fused = [d for d, _ in rrf([vec, fts])]
            row = {"id": q["id"], "query": q["query"]}
            for m, ranked in zip(METHODS, (vec, fts, fused), strict=True):
                row[m] = recall_at(ranked, rel)
                totals[m]["recall"] += row[m]
                totals[m]["mrr"] += reciprocal_rank(ranked, rel)
            per_query.append(row)

    n = len(queries)
    print(f"{'id':4} {'query':42} {'vec':>5} {'fts':>5} {'rrf':>5}")
    for row in per_query:
        print(f"{row['id']:4} {row['query'][:42]:42} {row['vector']:5.2f} {row['fts']:5.2f} {row['rrf']:5.2f}")
    print()
    print(f"| method | recall@{K_EVAL} | MRR |")
    print("|---|---|---|")
    for m in METHODS:
        print(f"| {m} | {totals[m]['recall'] / n:.3f} | {totals[m]['mrr'] / n:.3f} |")
    print(f"\n{n} queries, candidates per retriever = 20, RRF k = 60")


if __name__ == "__main__":
    main()
