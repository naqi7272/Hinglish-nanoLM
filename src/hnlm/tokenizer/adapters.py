"""One small interface over very different tokenizers, so the comparison code doesn't care
whether a tokenizer comes from our own .json file, OpenAI's tiktoken, or the HF Hub."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer


@dataclass
class TokenizerAdapter:
    name: str
    vocab_size: int
    encode: Callable[[str], list[int]]  # never adds BOS/EOS: we compare raw text only
    decode: Callable[[list[int]], str]

    def pieces(self, text: str) -> list[str]:
        """Each token decoded on its own. A token holding only part of a multi-byte
        character (common for Hindi in English-centric tokenizers) decodes to '�'."""
        return [self.decode([i]) for i in self.encode(text)]


def from_hf_tokenizer(name: str, tok: Tokenizer) -> TokenizerAdapter:
    return TokenizerAdapter(
        name=name,
        vocab_size=tok.get_vocab_size(),
        encode=lambda s: tok.encode(s, add_special_tokens=False).ids,
        decode=lambda ids: tok.decode(ids, skip_special_tokens=False),
    )


def from_file(name: str, path: str | Path) -> TokenizerAdapter:
    return from_hf_tokenizer(name, Tokenizer.from_file(str(path)))


def from_tiktoken(name: str, encoding: str) -> TokenizerAdapter:
    import tiktoken  # optional dependency: `uv sync --extra compare`

    enc = tiktoken.get_encoding(encoding)
    return TokenizerAdapter(
        name=name,
        vocab_size=enc.n_vocab,
        encode=lambda s: enc.encode(s, disallowed_special=()),
        decode=lambda ids: enc.decode(ids, errors="replace"),
    )


def from_hub(name: str, repos: list[str]) -> TokenizerAdapter:
    """Load tokenizer.json from the first Hub repo that works (gated repos need HF_TOKEN)."""
    token = os.environ.get("HF_TOKEN")
    errors = []
    for repo in repos:
        try:
            tok = Tokenizer.from_pretrained(repo, token=token)
            print(f"{name}: loaded from {repo}")
            return from_hf_tokenizer(name, tok)
        except Exception as e:  # gated / missing / offline: try the next repo
            errors.append(f"{repo}: {type(e).__name__}: {str(e)[:120]}")
    raise RuntimeError(f"could not load {name}:\n  " + "\n  ".join(errors))


def from_reference(spec: dict) -> TokenizerAdapter:
    if "tiktoken" in spec:
        return from_tiktoken(spec["name"], spec["tiktoken"])
    return from_hub(spec["name"], spec["hub"])
