"""Render the interaction-branching-simplex figure from a result JSON.

Left panel: cell composition (genuine 4-volume / geometrically frustrated /
disconnected) for the open and closed cell, per Schwinger m/g.
Right panel: the fraction of frustrated-open cells cured to a 4-volume by
the t+2dt closure.

Usage::

    python examples/quantum/plot_interaction_branching_simplex.py \\
        --in-json /tmp/interaction-branching/result.json \\
        --out-png docs/source/quantum-experiments/figures/interaction_branching_simplex.png
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VOLUME = "#2c7fb8"
FRUSTRATED = "#d95f02"
DISCONNECTED = "#bdbdbd"
CURED = "#1b9e77"


def composition(branch: dict) -> tuple[float, float, float]:
    """(4-volume, frustrated, disconnected) as fractions of ALL cells."""
    disc = branch["frac_disconnected"]
    conn = 1.0 - disc
    vol = branch["det_gram"]["frac_positive"] * conn
    fru = branch["det_gram"]["frac_negative"] * conn
    return vol, fru, disc


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-json", default="/tmp/interaction-branching/result.json")
    p.add_argument("--out-png",
                   default="docs/source/quantum-experiments/figures/"
                           "interaction_branching_simplex.png")
    args = p.parse_args()

    with open(args.in_json) as f:
        records = json.load(f)["records"]

    mgs = [r["m_over_g"] for r in records]
    x = np.arange(len(mgs))
    w = 0.38

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))

    for off, regime, hatch in ((-w / 2, "open", None), (w / 2, "closed", "//")):
        vol, fru, disc = zip(*(composition(r[regime]) for r in records))
        vol, fru, disc = np.array(vol), np.array(fru), np.array(disc)
        ax0.bar(x + off, vol, w, color=VOLUME, hatch=hatch, edgecolor="white",
                label=f"{regime}: 4-volume")
        ax0.bar(x + off, fru, w, bottom=vol, color=FRUSTRATED, hatch=hatch,
                edgecolor="white", label=f"{regime}: frustrated")
        ax0.bar(x + off, disc, w, bottom=vol + fru, color=DISCONNECTED,
                hatch=hatch, edgecolor="white", label=f"{regime}: disconnected")
    ax0.set_xticks(x)
    ax0.set_xticklabels([f"{m:g}" for m in mgs])
    ax0.set_xlabel(r"Schwinger $m/g$")
    ax0.set_ylabel("fraction of cells")
    ax0.set_ylim(0, 1)
    ax0.set_title("Cell composition: open (solid) vs closed (hatched)")
    ax0.legend(fontsize=7, ncol=2, loc="upper center")

    cured = [r["transition"]["frac_frustrated_cured_by_closure"]
             for r in records]
    n_fru = [r["transition"]["n_frustrated_open"] for r in records]
    bars = ax1.bar(x, cured, 0.5, color=CURED, edgecolor="white")
    for xi, c, n in zip(x, cured, n_fru):
        ax1.text(xi, c + 0.02, f"{c:.2f}\n(n={n})", ha="center", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{m:g}" for m in mgs])
    ax1.set_xlabel(r"Schwinger $m/g$")
    ax1.set_ylabel("fraction cured")
    ax1.set_ylim(0, 1.1)
    ax1.set_title("Frustrated-open cells cured to 4-volume by the closure")

    fig.suptitle("Interaction-branching simplex: the t+2dt closure rotates "
                 "the cell out of the plane", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out_png, dpi=130)
    print(f"[wrote] {args.out_png}")


if __name__ == "__main__":
    main()
