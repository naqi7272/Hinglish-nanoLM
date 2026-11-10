"""One visual style for every plot in the project (light theme, static PNGs for the README).

Colors come from a validated reference palette. Rules we follow: categorical colors are
assigned in a fixed order and follow the entity (the 30M model is always the same color),
grids and axes stay quiet, and identity is never color-only (labels or a legend always).
"""

from __future__ import annotations

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#b4b2ab"  # "everything else" in highlight charts
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua: all-pairs colorblind-safe

OURS = SERIES[0]


def plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    return pyplot


def style_axes(ax, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    ax.set_facecolor(SURFACE)
    if title:
        ax.set_title(title, fontsize=11, color=TEXT_PRIMARY, loc="left")
    ax.set_xlabel(xlabel, fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def new_figure(width: float, height: float, ncols: int = 1, **kw):
    fig, axes = plt().subplots(1, ncols, figsize=(width, height), **kw)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def save(fig, path) -> None:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt().close(fig)
