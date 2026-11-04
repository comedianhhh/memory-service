"""Cross-encoder reranking: the expensive, accurate last step of the funnel.

A bi-encoder (bge-small) embeds query and memory separately, so memory vectors can be precomputed.
A cross-encoder reads (query, memory) together in one pass and outputs a relevance score: it sees
word-by-word interaction ("skips lunch" vs "skips meals"), but nothing can be precomputed, so it is
only run on the ~20 candidates that RRF already picked.
"""

from __future__ import annotations

from functools import lru_cache

from .logs import timed

MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"  # ~80 MB, trained on MS MARCO passage ranking


@lru_cache(maxsize=1)
def _model():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(MODEL)


def rerank(query: str, candidates: list[tuple[str, str]], limit: int | None = None) -> list[str]:
    """candidates: [(id, text)] -> ids sorted by cross-encoder score, best first."""
    if not candidates:
        return []
    with timed("rerank", model=MODEL, n=len(candidates)):
        scores = list(_model().rerank(query, [text for _, text in candidates]))
    order = sorted(range(len(candidates)), key=lambda i: -scores[i])
    ids = [candidates[i][0] for i in order]
    return ids[:limit] if limit else ids
