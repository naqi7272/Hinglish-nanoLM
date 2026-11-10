"""Metrics go to local CSV files ALWAYS (the plots are made from these), and to Weights &
Biases too when it's installed and logged in. W&B is optional: a missing key or no internet
never stops training."""

from __future__ import annotations

import csv
import os
from pathlib import Path


class RunLogger:
    def __init__(
        self,
        run_dir: Path,
        project: str | None,
        run_name: str,
        config: dict,
        wandb_id: str | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.wandb = None
        self.wandb_id = wandb_id
        if project and (os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_MODE")):
            try:
                import wandb

                self.wandb = wandb.init(
                    project=project,
                    name=run_name,
                    id=wandb_id,
                    resume="allow",
                    config=config,
                    dir=str(run_dir),
                )
                self.wandb_id = self.wandb.id
            except Exception as e:  # never let logging kill a training run
                print(f"W&B disabled: {e}")

    def log(self, kind: str, row: dict) -> None:
        """kind: 'train' or 'eval' -> appended to {run_dir}/{kind}.csv"""
        path = self.run_dir / f"{kind}.csv"
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        if self.wandb is not None:
            self.wandb.log(
                {f"{kind}/{k}": v for k, v in row.items() if k != "step"}, step=row["step"]
            )

    def truncate_after(self, step: int) -> None:
        """After resuming from step N, drop CSV rows logged after N (they'll be redone)."""
        for kind in ("train", "eval"):
            path = self.run_dir / f"{kind}.csv"
            if not path.exists():
                continue
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
            keep = [r for r in rows if int(r["step"]) <= step]
            if len(keep) != len(rows):
                with open(path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(keep)

    def finish(self) -> None:
        if self.wandb is not None:
            self.wandb.finish()
