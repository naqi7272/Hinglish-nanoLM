"""Random training windows from several token files, mixed by weight.

Each batch row picks a source (weighted), then a random 513-token window in it:
x = window[:-1], y = window[1:]  (y is x shifted by one: "predict the next token").

The randomness comes from an rng seeded with (seed, step), so batch number N is always the
same batch. That makes resuming after a disconnect exact without saving any loader state.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class MixtureLoader:
    def __init__(
        self,
        bins_dir: Path,
        sources: list[str],
        split: str = "train",
        weights: dict[str, float] | None = None,
    ) -> None:
        self.names = list(sources)
        self.data = {
            s: np.memmap(Path(bins_dir) / f"{s}.{split}.bin", dtype=np.uint16, mode="r")
            for s in self.names
        }
        # Default: proportional to size, i.e. every token of the corpus is equally likely.
        w = np.array([(weights or {}).get(s, len(self.data[s])) for s in self.names], float)
        self.weights = w / w.sum()

    def total_tokens(self) -> int:
        return sum(len(d) for d in self.data.values())

    def get_batch(self, rng: np.random.Generator, batch: int, ctx: int, device) -> tuple:
        rows = rng.choice(len(self.names), size=batch, p=self.weights)
        xs = np.empty((batch, ctx + 1), dtype=np.int64)
        for i, r in enumerate(rows):
            d = self.data[self.names[r]]
            start = rng.integers(0, len(d) - ctx - 1)
            xs[i] = d[start : start + ctx + 1]
        t = torch.from_numpy(xs)
        if torch.device(device).type == "cuda":
            t = t.pin_memory().to(device, non_blocking=True)
        else:
            t = t.to(device)
        return t[:, :-1], t[:, 1:]
