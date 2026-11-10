"""Near-duplicate removal with MinHash + LSH, written from scratch with numpy.

Why: Twitter data is full of retweets and copy-pasted forwards; web data is full of
boilerplate. If the model sees the same text 500 times it memorizes it, and duplicates
that land in both train and val make the val loss look better than it really is.

The idea in four steps
----------------------
1. Shingles. Turn each document into a SET of overlapping character 5-grams
   ("mera naam" -> {"mera ", "era n", "ra na", ...}). Two documents are similar if their
   sets overlap a lot. Overlap is measured with Jaccard similarity: |A ∩ B| / |A ∪ B|.

2. MinHash. Comparing big sets is slow, so we squash each set into a short signature of
   P numbers. For each of P random hash functions h_i, store min(h_i(s) for s in set).
   Magic fact: P(min-hash of A == min-hash of B) = Jaccard(A, B). So the fraction of
   positions where two signatures agree is an estimate of their Jaccard similarity.

3. LSH banding. Comparing every pair of signatures is O(N^2), impossible for millions of
   docs. Instead cut each signature into b bands of r rows. Docs that are IDENTICAL in at
   least one band become "candidate pairs". For Jaccard s, P(candidate) = 1 - (1 - s^r)^b,
   an S-shaped curve that jumps from ~0 to ~1 near s ≈ (1/b)^(1/r). We use b=16, r=4
   (curve centre ~0.5): generous on purpose, so a pair with s=0.8 is found with
   probability 0.9998. Finding identical bands is just sorting, which is fast.

4. Union-find. Candidates are verified with the signature estimate (>= 0.7), so the
   generous banding doesn't cause false merges. Verified pairs are merged into
   clusters ("A~B and B~C" means A, B, C are one cluster). We keep the first document in
   each cluster and drop the rest.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import regex

_NOT_WORDISH = regex.compile(r"[^\p{L}\p{M}\p{N}\s]+")  # \p{M} keeps Devanagari vowel signs
_WS = regex.compile(r"\s+")

_POLY_BASE = np.uint64(1_000_003)
_MASK32 = np.uint64(0xFFFFFFFF)


def dedup_normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace, so trivial edits don't hide dupes."""
    text = _NOT_WORDISH.sub(" ", text.lower())
    return _WS.sub(" ", text).strip()


def exact_hash(text: str) -> int:
    """64-bit fingerprint of the normalized text (catches exact copies cheaply)."""
    digest = hashlib.blake2b(dedup_normalize(text).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def _mix64(x: np.ndarray) -> np.ndarray:
    """splitmix64 finalizer: scrambles bits so similar n-grams get unrelated hashes."""
    x = x ^ (x >> np.uint64(30))
    x = x * np.uint64(0xBF58476D1CE4E5B9)
    x = x ^ (x >> np.uint64(27))
    x = x * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def shingle_hashes(text: str, ngram: int = 5, max_chars: int = 4000) -> np.ndarray:
    """Return the SET (unique uint64 hashes) of character n-grams of the normalized text."""
    norm = dedup_normalize(text)[:max_chars]
    codes = np.frombuffer(norm.encode("utf-32-le"), dtype=np.uint32).astype(np.uint64)
    if len(codes) == 0:
        codes = np.zeros(1, dtype=np.uint64)
    n = min(ngram, len(codes))
    # Polynomial rolling hash of every window, computed for all windows at once:
    # h = c[i]*B^(n-1) + c[i+1]*B^(n-2) + ... + c[i+n-1]   (uint64 wrap-around is fine)
    windows = len(codes) - n + 1
    h = np.zeros(windows, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(n):
            h = h * _POLY_BASE + codes[j : j + windows]
        return np.unique(_mix64(h))


@dataclass
class MinHasher:
    num_perm: int = 64
    ngram: int = 5
    max_chars: int = 4000
    seed: int = 1

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        # "Multiply-shift" hashing: h_i(x) = high 32 bits of (a_i * x + b_i) mod 2^64.
        # a_i must be odd. Each (a_i, b_i) pair is one random hash function.
        self.a = rng.integers(1, 2**63, size=self.num_perm, dtype=np.uint64) | np.uint64(1)
        self.b = rng.integers(0, 2**63, size=self.num_perm, dtype=np.uint64)

    def signature(self, text: str) -> np.ndarray:
        x = shingle_hashes(text, self.ngram, self.max_chars)  # (k,)
        with np.errstate(over="ignore"):
            hashed = (x[:, None] * self.a[None, :] + self.b[None, :]) >> np.uint64(32)
        return (hashed & _MASK32).min(axis=0).astype(np.uint32)  # (num_perm,)


def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    return float(np.mean(sig_a == sig_b))


def band_keys(signatures: np.ndarray, bands: int) -> np.ndarray:
    """(N, P) signatures -> (N, bands) one uint64 key per band."""
    n, p = signatures.shape
    if p % bands:
        raise ValueError(f"num_perm={p} must be divisible by bands={bands}")
    rows = p // bands
    sig = signatures.astype(np.uint64).reshape(n, bands, rows)
    keys = np.zeros((n, bands), dtype=np.uint64)
    with np.errstate(over="ignore"):
        for r in range(rows):
            keys = keys * _POLY_BASE + sig[:, :, r]
    return _mix64(keys)


class UnionFind:
    """Disjoint sets. The root of each set is always its SMALLEST index (= first seen doc)."""

    def __init__(self, n: int) -> None:
        self.parent = np.arange(n)

    def find(self, i: int) -> int:
        parent = self.parent
        while parent[i] != i:
            parent[i] = parent[parent[i]]  # path halving keeps trees shallow
            i = parent[i]
        return int(i)

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            lo, hi = min(ri, rj), max(ri, rj)
            self.parent[hi] = lo


def _adjacent_equal_pairs(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort keys; every run of equal keys yields pairs (a, b) of neighbouring indices."""
    order = np.argsort(keys, kind="stable")
    same = keys[order][1:] == keys[order][:-1]
    idx = np.nonzero(same)[0]
    return order[idx], order[idx + 1]


def find_duplicates(
    signatures: np.ndarray,
    exact_hashes: np.ndarray,
    bands: int = 16,
    jaccard_threshold: float = 0.7,
) -> np.ndarray:
    """Return a boolean `keep` mask of length N (True = first doc of its cluster)."""
    n = len(signatures)
    uf = UnionFind(n)

    a, b = _adjacent_equal_pairs(exact_hashes)
    for i, j in zip(a.tolist(), b.tolist(), strict=True):
        uf.union(i, j)

    keys = band_keys(signatures, bands)
    for band in range(bands):
        a, b = _adjacent_equal_pairs(keys[:, band])
        for i, j in zip(a.tolist(), b.tolist(), strict=True):
            # LSH only PROPOSES pairs; check the signature estimate before merging.
            if estimate_jaccard(signatures[i], signatures[j]) >= jaccard_threshold:
                uf.union(i, j)

    return np.array([uf.find(i) == i for i in range(n)], dtype=bool)
