"""All experiment plots, made from the runs' CSV logs (no W&B needed).

    python scripts/make_plots.py --runs-dir runs --out plots
Writes plots/loss_vs_tokens.png, loss_vs_size.png, bpb_tokenizers.png, kv_cache_speed.png
(each only if its inputs exist) and results/experiments.md with the numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from hnlm import plotting as P

SIZE_RUNS = ["hinglish16k-5m", "hinglish16k-15m", "hinglish16k-30m"]
BASELINE_PAIR = ["hinglish16k-15m", "english16k-15m"]
VAL_LABELS = {
    "hingcorpus": "Roman Hinglish (tweets)",
    "sangraha_hin": "Hindi (Devanagari)",
    "fineweb_edu_en": "English",
    "phinc": "Roman Hinglish (PHINC)",
}


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def load_runs(runs_dir: Path) -> dict:
    runs = {}
    for d in sorted(runs_dir.iterdir()) if runs_dir.exists() else []:
        if (d / "eval.csv").exists() and (d / "config.json").exists():
            runs[d.name] = {
                "eval": read_csv(d / "eval.csv"),
                "config": json.loads((d / "config.json").read_text()),
            }
    return runs


def mean_train_val_bpb(row: dict) -> float:
    keys = [f"bpb_{s}" for s in ("hingcorpus", "sangraha_hin", "fineweb_edu_en")]
    return sum(row[k] for k in keys) / len(keys)


def plot_loss_vs_tokens(runs: dict, out: Path) -> None:
    if not any(n in runs for n in SIZE_RUNS):
        return
    fig, ax = P.new_figure(7, 4.2)
    for color, name in zip(P.SERIES, SIZE_RUNS):
        if name not in runs:
            continue
        rows = runs[name]["eval"]
        xs = [r["tokens"] / 1e6 for r in rows]
        ys = [mean_train_val_bpb(r) for r in rows]
        params = runs[name]["config"]["non_embedding_params"] / 1e6
        ax.plot(xs, ys, color=color, linewidth=2, label=f"{params:.1f}M params")
        ax.annotate(
            f"{params:.0f}M",
            (xs[-1], ys[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=9,
            color=P.TEXT_PRIMARY,
            va="center",
        )
    P.style_axes(
        ax,
        "tokens seen (millions)",
        "validation bits per byte",
        "Validation loss vs tokens seen (lower is better)",
    )
    ax.legend(frameon=False, fontsize=9)
    P.save(fig, out)


def plot_loss_vs_size(runs: dict, out: Path) -> list[dict]:
    pts = [
        (runs[n]["config"]["non_embedding_params"], mean_train_val_bpb(runs[n]["eval"][-1]), n)
        for n in SIZE_RUNS
        if n in runs
    ]
    if not pts:
        return []
    fig, ax = P.new_figure(5.5, 4)
    ax.plot([p[0] / 1e6 for p in pts], [p[1] for p in pts], color=P.NEUTRAL, linewidth=2)
    for n_params, bpb, name in pts:
        color = P.SERIES[SIZE_RUNS.index(name)]  # color follows the model, not its rank
        ax.scatter(
            [n_params / 1e6], [bpb], color=color, s=64, zorder=3, edgecolors=P.SURFACE, linewidths=2
        )
        ax.annotate(
            f"{bpb:.3f}",
            (n_params / 1e6, bpb),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            color=P.TEXT_PRIMARY,
        )
    ax.set_xscale("log")
    P.style_axes(
        ax,
        "non-embedding parameters (millions, log scale)",
        "final validation bits per byte",
        "Bigger model, lower loss",
    )
    P.save(fig, out)
    return [{"run": n, "params": p, "val_bpb": round(b, 4)} for p, b, n in pts]


def plot_bpb_tokenizers(runs: dict, out: Path) -> list[dict]:
    if not all(n in runs for n in BASELINE_PAIR):
        return []
    final = {n: runs[n]["eval"][-1] for n in BASELINE_PAIR}
    sets = [s for s in VAL_LABELS if f"bpb_{s}" in final[BASELINE_PAIR[0]]]
    fig, ax = P.new_figure(7.5, 4)
    width = 0.36
    labels = {"hinglish16k-15m": "hinglish16k (ours)", "english16k-15m": "english16k (baseline)"}
    colors = {"hinglish16k-15m": P.OURS, "english16k-15m": P.NEUTRAL}
    for i, name in enumerate(BASELINE_PAIR):
        xs = [j + (i - 0.5) * (width + 0.02) for j in range(len(sets))]
        vals = [final[name][f"bpb_{s}"] for s in sets]
        ax.bar(xs, vals, width=width, color=colors[name], label=labels[name])
        for x, v in zip(xs, vals):
            ax.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=P.TEXT_PRIMARY)
    ax.set_xticks(range(len(sets)), [VAL_LABELS[s] for s in sets], fontsize=8)
    P.style_axes(
        ax, "", "bits per byte (lower is better)", "Same 15M model, two tokenizers: bits per byte"
    )
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=9)
    P.save(fig, out)
    return [{"val_set": s, **{n: final[n][f"bpb_{s}"] for n in BASELINE_PAIR}} for s in sets]


def plot_kv_speed(path: Path, out: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = json.loads(path.read_text())
    size = "30m" if any(r["size"] == "30m" for r in rows) else rows[0]["size"]
    rows = [r for r in rows if r["size"] == size]
    fig, ax = P.new_figure(6, 4)
    xs = [r["new_tokens"] for r in rows]
    for color, key, label in (
        (P.SERIES[0], "tok_per_s_cache", "with KV cache"),
        (P.SERIES[1], "tok_per_s_no_cache", "without KV cache"),
    ):
        ax.plot(
            xs,
            [r[key] for r in rows],
            color=color,
            linewidth=2,
            marker="o",
            markersize=6,
            label=label,
        )
    P.style_axes(
        ax,
        "tokens generated",
        "tokens per second",
        f"Generation speed, {size} model ({rows[0]['device']})",
    )
    ax.legend(frameon=False, fontsize=9)
    P.save(fig, out)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out", default="plots")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    runs, out, res = load_runs(Path(args.runs_dir)), Path(args.out), Path(args.results)
    print("runs found:", list(runs))

    plot_loss_vs_tokens(runs, out / "loss_vs_tokens.png")
    summary = {
        "sizes": plot_loss_vs_size(runs, out / "loss_vs_size.png"),
        "tokenizers_bpb": plot_bpb_tokenizers(runs, out / "bpb_tokenizers.png"),
        "kv_cache": plot_kv_speed(res / "kv_cache_speed.json", out / "kv_cache_speed.png"),
    }
    res.mkdir(parents=True, exist_ok=True)
    (res / "experiments.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
