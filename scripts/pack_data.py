"""Tokenize every split with a trained tokenizer into data/bins/{tokenizer}/*.bin.

uv run python scripts/pack_data.py --tokenizer hinglish16k --data-dir /content/data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hnlm.data.pack import pack_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="hinglish16k")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    meta = pack_all(
        data_dir / "tokenizers" / f"{args.tokenizer}.json",
        data_dir / "splits",
        data_dir / "bins" / args.tokenizer,
        args.force,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
