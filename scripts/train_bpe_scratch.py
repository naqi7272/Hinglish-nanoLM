"""Train the from-scratch BPE and watch which merges it learns.

    uv run python scripts/train_bpe_scratch.py                      # tiny fixture corpus
    uv run python scripts/train_bpe_scratch.py --input data/splits/hingcorpus.train.jsonl \
        --max-docs 20000 --vocab-size 1000
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import islice
from pathlib import Path

from hnlm.tokenizer.bpe_scratch import BPETokenizer


def read_texts(path: Path, max_docs: int) -> list[str]:
    with open(path, encoding="utf-8") as f:
        lines = islice(f, max_docs)
        if path.suffix == ".jsonl":
            return [json.loads(line)["text"] for line in lines]
        return [line.rstrip("\n") for line in lines if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="tests/fixtures/tiny_corpus.txt")
    ap.add_argument("--max-docs", type=int, default=10_000)
    ap.add_argument("--vocab-size", type=int, default=512)
    ap.add_argument("--out", default=None, help="save merges to this .json file")
    args = ap.parse_args()

    texts = read_texts(Path(args.input), args.max_docs)
    tok = BPETokenizer()
    t0 = time.time()
    tok.train(texts, args.vocab_size, verbose=True)
    print(f"\ntrained {len(tok.merges)} merges in {time.time() - t0:.1f}s")

    sample = "Yaar, मैं कल Delhi जा रहा हूँ, tum bhi chaloge?"
    ids = tok.encode(sample)
    print(f"\n{sample!r}\n-> {len(ids)} tokens (vs {len(sample.encode())} bytes):")
    print(" | ".join(tok.pretty(i) for i in ids))
    if args.out:
        tok.save(args.out)


if __name__ == "__main__":
    main()
