"""Train (or resume) one run. Re-running the same command after a disconnect continues
from the last checkpoint.

    python scripts/train.py --config configs/train/hinglish16k-15m.yaml \
        --data-dir /content/data --runs-dir /content/drive/MyDrive/hinglish-nanolm/runs
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hnlm.train.trainer import TrainConfig, train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--max-steps", type=int, default=None, help="stop early (smoke tests)")
    ap.add_argument("--device", default=None, help="auto | cuda | cpu | mps")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    cfg = TrainConfig.from_yaml(args.config, device=args.device)
    if args.no_wandb:
        cfg.wandb_project = None
    train(cfg, Path(args.data_dir), Path(args.runs_dir), args.max_steps)


if __name__ == "__main__":
    main()
