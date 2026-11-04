"""Embedders. Real one uses the same model as jobs-mcp/rank.py (384 dims); FakeEmbedder is for tests."""

from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
from typing import Protocol

from .logs import timed

MODEL = "BAAI/bge-small-en-v1.5"
DIM = 384


class Embedder(Protocol):
    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


@lru_cache(maxsize=1)
def _model():
    from fastembed import TextEmbedding

    return TextEmbedding(MODEL)


class FastEmbedder:
    """Local ONNX model; downloads once (~130 MB). Logs duration and real tokenizer token counts."""

    def _tokens(self, texts: list[str]) -> int:
        tok = _model().model.tokenizer  # the model's own tokenizer, includes [CLS]/[SEP]
        return sum(len(e.ids) for e in tok.encode_batch(texts))

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        with timed("embed", kind="passage", model=MODEL, n=len(texts)) as log:
            out = [v.tolist() for v in _model().passage_embed(texts)]
            log["tokens"] = self._tokens(texts)
        return out

    def embed_query(self, text: str) -> list[float]:
        with timed("embed", kind="query", model=MODEL, n=1) as log:
            out = next(iter(_model().query_embed([text]))).tolist()
            log["tokens"] = self._tokens([text])
        return out


class FakeEmbedder:
    """Deterministic bag-of-words hashing into 384 dims: no model download, same text -> same vector,
    texts sharing words are closer. Good enough for API and RLS tests, useless for recall numbers."""

    _word = re.compile(r"[a-z0-9]+")

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for w in self._word.findall(text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v] if any(v) else [1.0 / math.sqrt(DIM)] * DIM

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def to_pgvector(v: list[float]) -> str:
    """pgvector accepts the text form '[0.1,0.2,...]', so no extra driver package is needed."""
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"
