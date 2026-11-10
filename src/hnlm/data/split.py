"""Deterministic train/val split.

We decide each document's split from a hash of its text, not from a random number.
That makes the split stable: re-running the pipeline (or adding more data) never moves
a document from val into train, which would silently leak val data into training.
"""

from __future__ import annotations

import hashlib


def assign_split(text: str, val_fraction: float, seed: int = 0) -> str:
    digest = hashlib.blake2b(f"{seed}:{text}".encode(), digest_size=8).digest()
    bucket = int.from_bytes(digest, "little") / 2**64  # uniform in [0, 1)
    return "val" if bucket < val_fraction else "train"
