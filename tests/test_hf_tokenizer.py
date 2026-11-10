import pytest

from hnlm.tokenizer.adapters import TokenizerAdapter, from_hf_tokenizer
from hnlm.tokenizer.compare import measure, segmentation_markdown
from hnlm.tokenizer.train_hf import build_tokenizer, train_tokenizer
from tests.test_bpe_roundtrip import TRICKY


@pytest.fixture(scope="module")
def hf_tok():
    from pathlib import Path

    lines = (Path(__file__).parent / "fixtures" / "tiny_corpus.txt").read_text("utf-8").splitlines()
    return from_hf_tokenizer("toy", train_tokenizer(lines, 500, ["<|endoftext|>"]))


def test_devanagari_words_are_single_pretokens():
    chunks = build_tokenizer().pre_tokenizer.pre_tokenize_str(" किताबें पढ़ना")
    assert len(chunks) == 2  # vowel signs (\p{M}) stay inside their word


@pytest.mark.parametrize("text", TRICKY)
def test_roundtrip(hf_tok, text):
    assert hf_tok.decode(hf_tok.encode(text)) == text


def test_special_token_is_one_id(hf_tok):
    assert len(hf_tok.encode("<|endoftext|>")) == 1


def test_measure_counts():
    byte_tok = TokenizerAdapter("bytes", 256, lambda s: list(s.encode()), lambda ids: "")
    m = measure(byte_tok, "toy", ["ab cd", "क"])
    assert (m.words, m.tokens, m.bytes, m.chars) == (3, 8, 8, 6)
    assert m.fertility == pytest.approx(8 / 3)
    assert m.bytes_per_token == 1.0


def test_segmentation_markdown_escapes_pipes(hf_tok):
    md = segmentation_markdown([hf_tok], ["a | b"])
    assert "| toy |" in md and "\\|" in md
