"""Generate text from a trained run and write results/samples_{run}.md.

python scripts/sample.py --run runs/hinglish16k-30m --data-dir data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from hnlm.device import get_device
from hnlm.model.generate import generate
from hnlm.train.trainer import load_model

PROMPTS = [
    "Aaj office mein",
    "Yaar kal ka match",
    "भारत की राजधानी",
    "मुझे लगता है कि",
    "Mummy ne bola ki",
    "The best way to learn",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory with final.pt or latest.pt")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()

    run = Path(args.run)
    ckpt = run / "final.pt" if (run / "final.pt").exists() else run / "latest.pt"
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    tok = Tokenizer.from_file(
        str(Path(args.data_dir) / "tokenizers" / f"{state['tokenizer']}.json")
    )
    dev = get_device()
    model = load_model(ckpt, dev.device)
    eot = tok.token_to_id("<|endoftext|>")
    g = torch.Generator(device=dev.device).manual_seed(0)

    lines = [
        f"# Samples from `{run.name}`",
        "",
        f"temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}",
        "",
    ]
    for prompt in PROMPTS:
        ids = torch.tensor([tok.encode(prompt, add_special_tokens=False).ids], device=dev.device)
        out = generate(
            model,
            ids,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.top_p,
            stop_token=eot,
            generator=g,
        )
        text = tok.decode(out[0].tolist(), skip_special_tokens=True)
        print(f"\n>>> {text}")
        lines += [f"**{prompt}** → {text[len(prompt) :].strip()}", ""]
    out_path = Path(args.out_dir) / f"samples_{run.name}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
