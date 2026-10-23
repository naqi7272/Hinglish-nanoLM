"""Compare our tokenizers with GPT-2, GPT-4o, Llama 3 and Qwen3. Writes:
results/tokenizer_comparison.json   all numbers
results/tokenizer_comparison.md     fertility table
results/segmentations.md            how each tokenizer cuts example sentences
plots/fertility.png, plots/chars_per_token.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from hnlm.tokenizer import compare
from hnlm.tokenizer.adapters import from_file, from_reference


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tokenizer.yaml")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--plots-dir", default="plots")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    data_dir = Path(args.data_dir or cfg["data_dir"])
    ccfg = cfg["compare"]

    tokenizers = [from_file(n, data_dir / cfg["out_dir"] / f"{n}.json") for n in cfg["tokenizers"]]
    for spec in ccfg["references"]:
        try:
            tokenizers.append(from_reference(spec))
        except Exception as e:  # keep going without it, but say so loudly
            print(f"WARNING: skipping {spec['name']}: {e}")

    results = []
    for label, rel in ccfg["eval_sets"].items():
        texts = compare.load_eval_texts(data_dir / rel, ccfg["max_docs"])
        for tok in tokenizers:
            m = compare.measure(tok, label, texts)
            results.append(m)
            print(
                f"{label:20s} {tok.name:12s} fertility={m.fertility:.2f} "
                f"bytes/token={m.bytes_per_token:.2f} chars/token={m.chars_per_token:.2f}"
            )

    res_dir, plot_dir = Path(args.results_dir), Path(args.plots_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "tokenizer_comparison.json").write_text(
        json.dumps([m.to_dict() for m in results], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    table = compare.results_markdown(results)
    (res_dir / "tokenizer_comparison.md").write_text(table + "\n", encoding="utf-8")
    (res_dir / "segmentations.md").write_text(
        compare.segmentation_markdown(tokenizers, ccfg["examples"]), encoding="utf-8"
    )
    compare.plot_metric(
        results,
        "fertility",
        "Tokens per word (lower is better)",
        "tokens / word",
        plot_dir / "fertility.png",
    )
    compare.plot_metric(
        results,
        "chars_per_token",
        "Characters per token (higher is better)",
        "chars / token",
        plot_dir / "chars_per_token.png",
    )
    print("\n" + table)


if __name__ == "__main__":
    main()
