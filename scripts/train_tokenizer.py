"""Train the production tokenizers listed in configs/tokenizer.yaml.

uv run python scripts/train_tokenizer.py --data-dir /content/drive/MyDrive/hinglish-nanolm/data
uv run python scripts/train_tokenizer.py --only english16k
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import yaml

from hnlm.tokenizer.train_hf import sample_texts, train_tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tokenizer.yaml")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--only", default=None, help="comma-separated tokenizer names")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    data_dir = Path(args.data_dir or cfg["data_dir"])
    out_dir = data_dir / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, spec in cfg["tokenizers"].items():
        if args.only and name not in args.only.split(","):
            continue
        streams = [
            sample_texts(data_dir / "splits" / f"{src}.train.jsonl", mb, cfg["seed"])
            for src, mb in spec["sample_mb"].items()
        ]
        print(f"training {name} on {spec['sample_mb']} MB ...")
        tok = train_tokenizer(
            itertools.chain(*streams),
            cfg["vocab_size"],
            cfg["special_tokens"],
            cfg["min_frequency"],
        )
        path = out_dir / f"{name}.json"
        tok.save(str(path))
        print(f"saved {path} (vocab {tok.get_vocab_size()})")


if __name__ == "__main__":
    main()
