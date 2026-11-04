"""Local embeddings with the same model as jobs-mcp/rank.py (384 dims)."""

from __future__ import annotations

from functools import lru_cache

MODEL = "BAAI/bge-small-en-v1.5"
DIM = 384


@lru_cache(maxsize=1)
def _model():
    from fastembed import TextEmbedding

    return TextEmbedding(MODEL)


def embed_passages(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in _model().passage_embed(texts)]


def embed_query(text: str) -> list[float]:
    return next(iter(_model().query_embed([text]))).tolist()


def to_pgvector(v: list[float]) -> str:
    """pgvector accepts the text form '[0.1,0.2,...]', so no extra driver package is needed."""
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"
