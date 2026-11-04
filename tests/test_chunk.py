import pytest

from memory_service.chunk import chunk_text, split_sentences


def test_short_message_is_one_chunk():
    assert chunk_text("I love oat milk. Flat whites only.") == ["I love oat milk. Flat whites only."]


def test_empty_and_whitespace():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_respects_max_words_and_keeps_sentences_whole():
    text = " ".join(f"Sentence number {i} has six words." for i in range(20))
    chunks = chunk_text(text, max_words=20, overlap_sentences=0)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.split()) <= 20
        assert c.endswith(".")


def test_overlap_repeats_last_sentence():
    text = "A one two three. B one two three. C one two three. D one two three."
    chunks = chunk_text(text, max_words=8, overlap_sentences=1)
    assert chunks == [
        "A one two three. B one two three.",
        "B one two three. C one two three.",
        "C one two three. D one two three.",
    ]


def test_no_chunk_is_only_overlap():
    text = "A one two three. B one two three."
    assert chunk_text(text, max_words=8, overlap_sentences=1) == ["A one two three. B one two three."]


def test_run_on_sentence_is_hard_split():
    text = " ".join(["word"] * 25)
    chunks = chunk_text(text, max_words=10, overlap_sentences=0)
    assert [len(c.split()) for c in chunks] == [10, 10, 5]


def test_paragraph_breaks_split_sentences():
    assert split_sentences("no punctuation here\n\nsecond paragraph") == ["no punctuation here", "second paragraph"]


def test_bad_max_words():
    with pytest.raises(ValueError):
        chunk_text("x", max_words=0)
