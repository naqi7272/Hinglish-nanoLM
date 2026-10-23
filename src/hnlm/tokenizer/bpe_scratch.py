"""Byte-level BPE, from scratch. For understanding, not speed.

==========================================================================================
HOW BYTE-LEVEL BPE WORKS
==========================================================================================

Start: every text is a sequence of UTF-8 BYTES, so the starting vocabulary is just the 256
possible byte values (ids 0..255). Any string in any language can be written with them:
    "a"  -> [97]                     (1 byte)
    "क"  -> [224, 164, 149]          (Devanagari letters are 3 bytes each)
    "😀" -> [240, 159, 152, 128]     (4 bytes)
So nothing is ever "unknown". The price: Hindi starts out 3x longer than English.

Training = repeat (vocab_size - 256) times:
    1. COUNT every adjacent pair of ids across the corpus.
    2. PICK the most frequent pair, e.g. (224, 164): the first two bytes of most Devanagari
       letters, so it's extremely common in Hindi text.
    3. MINT a new id for it (256, then 257, ...). Its bytes = bytes(a) + bytes(b).
    4. MERGE: replace every occurrence of that pair with the new id.
The ordered list of merges IS the tokenizer.

Encoding new text replays the merges in the order they were learned (lowest new id first).
Decoding concatenates each id's bytes and UTF-8 decodes them.

Speed trick we use: pre-tokenize the corpus into chunks (see pretokenize.py) and keep
    words: {tuple_of_ids: how_many_times_this_chunk_appears}
e.g. {(32, 107, 97): 5000, ...} for " ka". A pair inside a chunk that appears 5000 times
counts 5000 times. We never merge across chunk boundaries.

==========================================================================================
YOUR TASK: implement the three TODOs below
==========================================================================================
  1. count_pairs(words)            -> how often each adjacent pair appears
  2. merge(ids, pair, new_id)      -> replace a pair inside one id sequence
  3. the loop inside BPETokenizer.train()

Then run:   uv run pytest tests/test_bpe_roundtrip.py -v
Tests are skipped while the TODOs raise NotImplementedError.

Details that matter:
  * merge() scans left to right and does NOT overlap: merging (5, 5) in [5, 5, 5]
    gives [new, 5], not [new, new].
  * Ties: if two pairs have the same count, take the one seen FIRST while counting
    (Python's max() already does this). That keeps training deterministic.
  * Stop early if no pair is left to merge (every chunk is a single token).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import regex

from hnlm.tokenizer.pretokenize import pretokenize

Pair = tuple[int, int]


def count_pairs(words: dict[tuple[int, ...], int]) -> Counter[Pair]:
    """Count adjacent pairs across all chunks, weighted by chunk frequency.

    >>> count_pairs({(1, 2, 3): 2, (2, 3): 1})
    Counter({(2, 3): 3, (1, 2): 2})
    """
    # TODO(you): loop over each chunk and its frequency; for every adjacent pair (a, b)
    # in the chunk, add `freq` to counts[(a, b)]. Hint: zip(ids, ids[1:]).
    counts = Counter()
    for ids, freq in words.items():
        for a, b in zip(ids, ids[1:]):
            counts[(a, b)] += freq
    return counts


def merge(ids: list[int] | tuple[int, ...], pair: Pair, new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of `pair` (left to right) with `new_id`.

    >>> merge([1, 2, 3, 1, 2], (1, 2), 9)
    [9, 3, 9]
    >>> merge([5, 5, 5], (5, 5), 7)
    [7, 5]
    """
    # TODO(you): walk through `ids` with an index i. If ids[i], ids[i+1] equals `pair`,
    # append new_id and jump i by 2; otherwise append ids[i] and step by 1.
    # Careful at the last position (there is no ids[i+1]).
    out = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)  # found the pair: write the new id
            i += 2  # skip how many positions?
        else:
            out.append(ids[i])  # not the pair: copy this number
            i += 1
    return out


class BPETokenizer:
    def __init__(self, special_tokens: Iterable[str] = ("<|endoftext|>",)) -> None:
        self.merges: dict[Pair, int] = {}  # (a, b) -> new id, in the order learned
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_tokens: dict[str, int] = {}
        self._special_names = list(special_tokens)
        self._register_special_tokens()

    # ---- training -----------------------------------------------------------------------

    def train(self, texts: Iterable[str], vocab_size: int, verbose: bool = False) -> None:
        n_merges = vocab_size - 256 - len(self._special_names)
        if n_merges < 0:
            raise ValueError(f"vocab_size must be >= {256 + len(self._special_names)}")

        # Pre-tokenize and count distinct chunks once.
        chunk_counts: Counter[str] = Counter()
        for text in texts:
            chunk_counts.update(pretokenize(text))
        words: dict[tuple[int, ...], int] = {
            tuple(chunk.encode("utf-8")): freq for chunk, freq in chunk_counts.items()
        }

        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i in range(n_merges):
            new_id = 256 + i
            # TODO(you): the merge loop. Four steps:
            #   1. counts = count_pairs(words); if it's empty, break (nothing left to merge)
            #   2. best = the most frequent pair (ties -> first seen; max(counts, key=...))
            #   3. record it: self.merges[best] = new_id
            #                 self.vocab[new_id] = self.vocab[a] + self.vocab[b]
            #   4. rebuild `words` with `best` merged inside every chunk. Two different
            #      chunks can't become identical after a merge, but summing freqs into the
            #      new dict is the safe way to write it anyway.
            # Optional: if verbose, print i, best, the new token (self.pretty(new_id)), count.
            counts = count_pairs(words)
            if not counts:
                break
            best = max(counts, key=counts.get)  # the most frequent pair
            a, b = best
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
            new_words = {}
            for chunk, freq in words.items():
                merged = tuple(merge(chunk, best, new_id))
                new_words[merged] = new_words.get(merged, 0) + freq
            words = new_words
            if verbose:
                print(i, best, self.pretty(new_id), counts[best])

        self._register_special_tokens()

    def _register_special_tokens(self) -> None:
        # Special tokens get ids AFTER all merges. They are never produced by merging;
        # encode() matches them as literal strings.
        base = 256 + len(self.merges)
        self.special_tokens = {name: base + k for k, name in enumerate(self._special_names)}
        for name, idx in self.special_tokens.items():
            self.vocab[idx] = name.encode("utf-8")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ---- encode / decode ----------------------------------------------------------------

    def _encode_chunk(self, chunk: str) -> list[int]:
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # Of all adjacent pairs, apply the one that was learned EARLIEST (lowest id).
            # Pairs that were never learned get rank infinity.
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        if self.special_tokens:
            pattern = "(" + "|".join(regex.escape(s) for s in self.special_tokens) + ")"
            parts = regex.split(pattern, text)
        else:
            parts = [text]
        for part in parts:
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            elif part:
                for chunk in pretokenize(part):
                    ids.extend(self._encode_chunk(chunk))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        # errors="replace": a slice of ids can end in the middle of a multi-byte character.
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def pretty(self, idx: int) -> str:
        """Human-readable token: decoded text if valid UTF-8, else the raw bytes."""
        raw = self.vocab[idx]
        try:
            return repr(raw.decode("utf-8"))
        except UnicodeDecodeError:
            return str(raw)

    # ---- persistence --------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        data = {
            "special_tokens": self._special_names,
            "merges": [[a, b, idx] for (a, b), idx in self.merges.items()],
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        data = json.loads(Path(path).read_text())
        tok = cls(special_tokens=data["special_tokens"])
        for a, b, idx in data["merges"]:
            tok.merges[(a, b)] = idx
            tok.vocab[idx] = tok.vocab[a] + tok.vocab[b]
        tok._register_special_tokens()
        return tok
