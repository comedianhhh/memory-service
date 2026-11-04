"""Reciprocal Rank Fusion over ranked id lists.

jobs-mcp/rank.py scores *every* doc with both methods, so it fuses parallel score arrays.
Here each retriever is a SQL query that only returns its own top-N, so a memory can be in one
list and missing from the other. Fusing ranked id lists handles that: a missing id just gets
no contribution from that list.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

DEFAULT_K = 60


def rrf(rankings: Sequence[Sequence[Hashable]], k: int = DEFAULT_K) -> list[tuple[Hashable, float]]:
    """Return (id, fused score) sorted best first. Rank is 1-based: score += 1 / (k + rank)."""
    scores: dict[Hashable, float] = {}
    first_seen: dict[Hashable, int] = {}
    order = 0
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in first_seen:
                first_seen[doc_id] = order
                order += 1
    # ties are broken by first appearance so the output is deterministic
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))
