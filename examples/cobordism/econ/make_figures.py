"""Comparison figures for the kill-experiment report (tessera#602).

Reads out/leak_history.csv (conductance metric) and produces:

* ``out/comparison.png`` — per year-pair: the observed irreducible
  fraction of the change vs the IPF size-recomposition null (top), and
  the magnitude baselines (bottom); recession pairs shaded.
* ``out/recession_ranks.png`` — where each statistic ranks the four
  recession pairs among all 27 transitions (rank 1 = largest). A
  statistic that separates recessions puts them in the shaded top-4
  zone; the magnitude baselines get closer than the topological leak.

Static PNGs on a light surface; palette slots from the validated
reference set (blue/red/aqua/violet), identity carried by legend +
direct labels, grid recessive.
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, AQUA, VIOLET, RED = "#2a78d6", "#1baf7a", "#4a3aa7", "#e34948"
INK, MUTED, GRID = "#1f2430", "#5a6270", "#d9dde3"
RECESSION_SHADE = "#e34948"

OUT = pathlib.Path("out")


def _style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def _shade_recessions(ax, df):
    for i, rec in enumerate(df["recession"]):
        if rec:
            ax.axvspan(i - 0.45, i + 0.45, color=RECESSION_SHADE,
                       alpha=0.10, zorder=0)


def comparison_figure(df: pd.DataFrame) -> None:
    x = np.arange(len(df))
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.2), sharex=True)

    ax = axes[0]
    _shade_recessions(ax, df)
    ax.plot(x, df["leak_frac"], color=BLUE, linewidth=2, marker="o",
            markersize=5, label="observed transition", zorder=3)
    ax.plot(x, df["leak_frac_null"], color=MUTED, linewidth=2,
            linestyle=(0, (4, 3)), marker="o", markersize=4,
            label="IPF null (size recomposition only)", zorder=2)
    ax.set_ylabel("irreducible fraction of the change", color=INK, fontsize=9)
    ax.set_title(
        "Topological leak vs the null — the observed transition is no less "
        "absorbable in recessions (shaded)",
        color=INK, fontsize=10, loc="left")
    ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK)
    ax.text(x[-1] + 0.3, df["leak_frac"].iloc[-1], "observed",
            color=BLUE, fontsize=8, va="center")
    ax.text(x[-1] + 0.3, df["leak_frac_null"].iloc[-1], "IPF null",
            color=MUTED, fontsize=8, va="center")
    _style(ax)

    ax = axes[1]
    _shade_recessions(ax, df)
    ax.plot(x, df["frobenius"], color=AQUA, linewidth=2, marker="o",
            markersize=5, label="Frobenius distance", zorder=3)
    ax.plot(x, df["leontief"], color=VIOLET, linewidth=2, marker="o",
            markersize=5, label="Leontief-inverse distance", zorder=3)
    ax.set_ylabel("relative distance", color=INK, fontsize=9)
    ax.set_title(
        "Magnitude baselines — these DO elevate in the recession pairs",
        color=INK, fontsize=10, loc="left")
    ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK)
    ax.text(x[-1] + 0.3, df["frobenius"].iloc[-1], "Frobenius",
            color=AQUA, fontsize=8, va="center")
    ax.text(x[-1] + 0.3, df["leontief"].iloc[-1], "Leontief",
            color=VIOLET, fontsize=8, va="center")
    _style(ax)

    ax.set_xticks(x)
    ax.set_xticklabels(df["pair"], rotation=60, fontsize=7, color=MUTED)
    ax.set_xlim(-0.6, len(df) - 0.4 + 2.2)
    fig.suptitle("Held-fixed leak vs baselines, BEA summary 1997–2024 "
                 "(shaded = recession year-pairs)", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "comparison.png", dpi=160)
    plt.close(fig)


def ranks_figure(df: pd.DataFrame) -> None:
    stats = [
        ("leak_frac_excess", "period-leak excess over IPF null\n(the register statistic)"),
        ("leak_frac", "irreducible fraction (raw)"),
        ("frobenius", "Frobenius distance\n(naive magnitude baseline)"),
        ("leontief", "Leontief-inverse distance\n(propagation baseline)"),
    ]
    pair_colors = dict(zip(
        df.loc[df["recession"], "pair"], (BLUE, RED, AQUA, VIOLET)))
    n = len(df)

    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.axvspan(0.5, 4.5, color=AQUA, alpha=0.10, zorder=0)
    ax.text(2.5, len(stats) - 0.42, "top-4 zone\n(= separation)",
            ha="center", color=MUTED, fontsize=8)

    for row, (col, label) in enumerate(stats):
        y = len(stats) - 1 - row
        ranks = df[col].rank(ascending=False)
        ax.hlines(y, 1, n, color=GRID, linewidth=1, zorder=1)
        for pair, color in pair_colors.items():
            r = float(ranks[df["pair"] == pair].iloc[0])
            ax.plot(r, y, "o", color=color, markersize=10, zorder=3,
                    markeredgecolor="white", markeredgewidth=1.5)
        ax.text(0.2, y, label, ha="right", va="center", color=INK, fontsize=9)

    for pair, color in pair_colors.items():
        ax.plot([], [], "o", color=color, markersize=8, label=pair)
    ax.legend(loc="lower right", fontsize=8, frameon=False, ncol=4,
              labelcolor=INK, title="recession pair", title_fontsize=8)

    ax.set_xlim(-9.5, n + 1)
    ax.set_ylim(-0.9, len(stats) - 0.1)
    ax.set_yticks([])
    ax.set_xticks([1, 5, 10, 15, 20, 25, 27])
    ax.set_xlabel("rank among the 27 year-pair transitions (1 = largest signal)",
                  color=INK, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_title(
        "Where each statistic ranks the recessions — the baselines separate "
        "them better than the register statistic",
        color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "recession_ranks.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    df = pd.read_csv(OUT / "leak_history.csv")
    comparison_figure(df)
    ranks_figure(df)
    print(f"wrote {OUT/'comparison.png'} and {OUT/'recession_ranks.png'}")
