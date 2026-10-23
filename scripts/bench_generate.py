"""How much faster is generation with a KV cache? Speed doesn't depend on what the weights
are, so this uses randomly initialized models of each size (runs on CPU or GPU).

    python scripts/bench_generate.py --out results/kv_cache_speed.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from hnlm.device import get_device
from hnlm.model.config import SIZES, GPTConfig
from hnlm.model.generate import generate
from hnlm.model.gpt import GPT


def time_generate(model, n_new: int, use_cache: bool, device) -> float:
    prompt = torch.randint(0, model.cfg.vocab_size, (1, 16), device=device)
    generate(model, prompt, 8, temperature=0, use_cache=use_cache)  # warm-up
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    generate(model, prompt, n_new, temperature=0, use_cache=use_cache)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return n_new / (time.perf_counter() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="5m,15m,30m")
    ap.add_argument("--lengths", default="64,128,256,480")
    ap.add_argument("--out", default="results/kv_cache_speed.json")
    args = ap.parse_args()

    dev = get_device()
    results = []
    for size in args.sizes.split(","):
        torch.manual_seed(0)
        model = GPT(GPTConfig(**SIZES[size])).to(dev.device).eval()
        for n in map(int, args.lengths.split(",")):
            row = {"size": size, "new_tokens": n, "device": dev.describe()}
            for use_cache in (True, False):
                key = "cache" if use_cache else "no_cache"
                row[f"tok_per_s_{key}"] = round(time_generate(model, n, use_cache, dev.device), 1)
            row["speedup"] = round(row["tok_per_s_cache"] / row["tok_per_s_no_cache"], 2)
            print(row)
            results.append(row)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
