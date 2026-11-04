"""Split a long chat message into retrieval-sized chunks.

Why chunk at all: one embedding per message averages every topic in it into one vector. A message
that mentions the dentist, the cat and a flight ends up close to none of them. Smaller chunks keep
one idea per vector.

Why not too small: a lone "She said yes." loses who and what. So we pack whole sentences up to
`max_words` and repeat the last sentence of each chunk at the start of the next (overlap), so a
fact that straddles a boundary still appears intact in one chunk.

bge-small truncates at 512 tokens; 80 words is roughly 100-110 tokens, far below that.
"""

from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s and s.strip()]


def _hard_split(sentence: str, max_words: int) -> list[str]:
    words = sentence.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def chunk_text(text: str, max_words: int = 80, overlap_sentences: int = 1) -> list[str]:
    if max_words < 1:
        raise ValueError("max_words must be >= 1")
    sentences: list[str] = []
    for s in split_sentences(text):
        # one run-on sentence longer than a whole chunk gets cut by words
        sentences.extend(_hard_split(s, max_words) if len(s.split()) > max_words else [s])
    if not sentences:
        return []

    chunks: list[list[str]] = []
    current: list[str] = []
    fresh = 0  # sentences in `current` that are not overlap carried from the previous chunk
    for s in sentences:
        n_words = sum(len(x.split()) for x in current)
        if current and fresh and n_words + len(s.split()) > max_words:
            chunks.append(current)
            current = current[-overlap_sentences:] if overlap_sentences else []
            fresh = 0
            # drop overlap if it alone would push the next sentence over the limit
            while current and sum(len(x.split()) for x in current) + len(s.split()) > max_words:
                current.pop(0)
        current.append(s)
        fresh += 1
    if fresh:
        chunks.append(current)
    return [" ".join(c) for c in chunks]
