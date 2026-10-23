"""Tests for the from-scratch BPE. They skip until the TODOs in bpe_scratch.py are done."""

import pytest

from hnlm.tokenizer import bpe_scratch
from hnlm.tokenizer.bpe_scratch import BPETokenizer, count_pairs, merge
from hnlm.tokenizer.pretokenize import pretokenize

TRICKY = [
    "",
    "a",
    "Yaar, मैं कल Delhi जा रहा हूँ!!",
    "क्‍ष त्र ज्ञ  (ZWJ inside) ‌",
    "emoji 😀🇮🇳 and   multiple   spaces\n\n\tand tabs",
    "Numbers 1234567 and ₹500, 3.14",
    "<|endoftext|>next doc starts here",
    "words the tokenizer never saw: zxqv ञञञ",
]


def _skip_if_todo(fn, *args):
    try:
        return fn(*args)
    except NotImplementedError as e:
        pytest.skip(f"TODO not implemented yet: {e}")


@pytest.fixture
def trained(tiny_corpus):
    tok = BPETokenizer()
    _skip_if_todo(tok.train, tiny_corpus, 256 + 200 + 1)
    return tok


def test_pretokenize_keeps_devanagari_words_whole():
    assert pretokenize(" किताबें पढ़ना") == [" किताबें", " पढ़ना"]


def test_count_pairs():
    counts = _skip_if_todo(count_pairs, {(1, 2, 3): 2, (2, 3): 1})
    assert counts == {(1, 2): 2, (2, 3): 3}


def test_merge():
    assert _skip_if_todo(merge, [1, 2, 3, 1, 2], (1, 2), 9) == [9, 3, 9]
    assert merge([5, 5, 5], (5, 5), 7) == [7, 5]  # left to right, no overlap
    assert merge([1], (1, 2), 9) == [1]
    assert merge([], (1, 2), 9) == []


def test_first_merge_is_most_frequent_pair():
    tok = BPETokenizer(special_tokens=())
    _skip_if_todo(tok.train, ["aaabdaaabac"], 257)
    assert tok.merges == {(97, 97): 256}  # "aa" appears 4 times (overlaps counted)


@pytest.mark.parametrize("text", TRICKY)
def test_roundtrip(trained, text):
    assert trained.decode(trained.encode(text)) == text


def test_roundtrip_training_corpus(trained, tiny_corpus):
    for line in tiny_corpus:
        assert trained.decode(trained.encode(line)) == line


def test_compresses_and_vocab_size(trained, tiny_corpus):
    text = "\n".join(tiny_corpus)
    assert len(trained.encode(text)) < 0.6 * len(text.encode("utf-8"))
    assert trained.vocab_size == 256 + 200 + 1
    assert trained.encode("<|endoftext|>") == [trained.special_tokens["<|endoftext|>"]]


def test_save_load(trained, tmp_path):
    trained.save(tmp_path / "bpe.json")
    loaded = BPETokenizer.load(tmp_path / "bpe.json")
    for text in TRICKY:
        assert loaded.encode(text) == trained.encode(text)


def test_module_docstring_examples():
    import doctest

    try:
        count_pairs({})
    except NotImplementedError:
        pytest.skip("TODO not implemented yet")
    assert doctest.testmod(bpe_scratch).failed == 0
