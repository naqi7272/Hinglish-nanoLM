"""Checkpoints that survive a Colab/Kaggle disconnect.

A checkpoint holds everything needed to continue as if nothing happened: model weights,
optimizer state (AdamW's running averages), the fp16 GradScaler, the step counter, and the
W&B run id. It's written to a temp file first and then renamed, so a disconnect in the
middle of saving can never leave a corrupt `latest.pt` behind.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic on the same filesystem


def load_checkpoint(path: Path, map_location="cpu") -> dict | None:
    if not path.exists():
        return None
    return torch.load(path, map_location=map_location, weights_only=False)
