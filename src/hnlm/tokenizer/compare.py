"""Compare tokenizers on English, Devanagari Hindi and Roman Hinglish.

Metrics (computed over a whole eval set, not averaged per document):
  * fertility       = tokens / words      (words = whitespace-separated). Lower is better:
                      1.0 would mean every word is a single token.
  * bytes per token = UTF-8 bytes / tokens. Higher = more text packed into each token.
                      This is the "compression" view, and it's fair across scripts only
                      with care: one Devanagari character is 3 bytes, a Latin one is 1.
  * chars per token = characters / tokens. Script-neutral version of the above.

Why it matters: more tokens for the same sentence means higher API cost, slower
generation, and less real text fitting into the model's context window.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path

from hnlm.tokenizer.adapters import TokenizerAdapter


@dataclass
class Metrics:
    tokenizer: str
    eval_set: str
    vocab_size: int
    docs: int
    words: int
    tokens: int
    bytes: int
    chars: int

    @property
    def fertility(self) -> float:
        return self.tokens / max(self.words, 1)

    @property
    def bytes_per_token(self) -> float:
        return self.bytes / max(self.tokens, 1)

    @property
    def chars_per_token(self) -> float:
        return self.chars / max(self.tokens, 1)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(
            fertility=round(self.fertility, 3),
            bytes_per_token=round(self.bytes_per_token, 3),
            chars_per_token=round(self.chars_per_token, 3),
        )
        return d


def load_eval_texts(path: Path, max_docs: int) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in islice(f, max_docs)]


def measure(tok: TokenizerAdapter, eval_set: str, texts: list[str]) -> Metrics:
    m = Metrics(tok.name, eval_set, tok.vocab_size, len(texts), 0, 0, 0, 0)
    for text in texts:
        m.words += len(text.split())
        m.tokens += len(tok.encode(text))
        m.bytes += len(text.encode("utf-8"))
        m.chars += len(text)
    return m


PIPE = "\\|"  # an escaped pipe, so token pieces don't break the markdown table


def segmentation_markdown(tokenizers: list[TokenizerAdapter], examples: list[str]) -> str:
    """How each tokenizer cuts the same sentences. '·' marks a space, '�' a partial byte."""
    out = [
        "# Example segmentations",
        "",
        "`·` = space, `�` = token holding part of a character.",
        "",
    ]
    for text in examples:
        out += [f"## {text}", "", "| tokenizer | tokens | pieces |", "|---|---:|---|"]
        for tok in tokenizers:
            pieces = [p.replace(" ", "·").replace("|", PIPE) for p in tok.pieces(text)]
            out.append(f"| {tok.name} | {len(pieces)} | {(' ' + PIPE + ' ').join(pieces)} |")
        out.append("")
    return "\n".join(out)


def results_markdown(results: list[Metrics]) -> str:
    sets = list(dict.fromkeys(r.eval_set for r in results))
    names = list(dict.fromkeys(r.tokenizer for r in results))
    by = {(r.tokenizer, r.eval_set): r for r in results}
    out = ["| tokenizer | vocab | " + " | ".join(f"{s} fertility" for s in sets) + " |"]
    out.append("|---|---:|" + "---:|" * len(sets))
    for n in names:
        row = [f"{by[(n, s)].fertility:.2f}" if (n, s) in by else "–" for s in sets]
        vocab = next(r.vocab_size for r in results if r.tokenizer == n)
        out.append(f"| {n} | {vocab:,} | " + " | ".join(row) + " |")
    return "\n".join(out)


# ---- plots -----------------------------------------------------------------------------

# Reference palette (light mode): one highlight hue for OUR tokenizers, neutral for the rest,
# so the eye goes straight to the comparison that matters. Identity is never color-only:
# every bar is labelled with its tokenizer name on the y axis.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
OURS = "#2a78d6"
OTHERS = "#b4b2ab"


def plot_metric(
    results: list[Metrics],
    metric: str,
    title: str,
    xlabel: str,
    out_path: Path,
    ours: tuple[str, ...] = ("hinglish16k",),
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sets = list(dict.fromkeys(r.eval_set for r in results))
    names = list(dict.fromkeys(r.tokenizer for r in results))
    by = {(r.tokenizer, r.eval_set): getattr(r, metric) for r in results}
    xmax = max(by.values()) * 1.18

    fig, axes = plt.subplots(
        1, len(sets), figsize=(4.2 * len(sets), 0.5 * len(names) + 1.6), sharey=True
    )
    fig.patch.set_facecolor(SURFACE)
    for ax, s in zip(axes, sets):
        ax.set_facecolor(SURFACE)
        y = list(range(len(names)))[::-1]  # first tokenizer at the top
        vals = [by.get((n, s), 0.0) for n in names]
        colors = [OURS if n in ours else OTHERS for n in names]
        ax.barh(y, vals, height=0.6, color=colors)
        for yi, v in zip(y, vals):
            ax.text(v + xmax * 0.01, yi, f"{v:.2f}", va="center", fontsize=9, color=TEXT_PRIMARY)
        ax.set_yticks(y, names, fontsize=10, color=TEXT_PRIMARY)
        ax.set_xlim(0, xmax)
        ax.set_title(s, fontsize=11, color=TEXT_PRIMARY, loc="left")
        ax.set_xlabel(xlabel, fontsize=9, color=TEXT_SECONDARY)
        ax.tick_params(axis="x", colors=TEXT_SECONDARY, labelsize=8)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=TEXT_PRIMARY)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
