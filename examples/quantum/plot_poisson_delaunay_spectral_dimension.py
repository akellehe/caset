"""Render the Poisson-Delaunay spectral-dimension figure.

The result JSON is an N x m/g sweep. Left panel: peak D_S vs N, one line
per Schwinger m/g (mean over Poisson layouts, +/- std band). Right panel:
the Ambjorn-Loll D_inf fit vs N, same grouping.

Usage::

    python examples/quantum/plot_poisson_delaunay_spectral_dimension.py \\
        --in-json /tmp/interaction-branching/spectral_dimension.json \\
        --out-png docs/source/quantum-experiments/figures/poisson_delaunay_spectral_dimension.png
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#1b9e77", "#7570b3", "#d95f02", "#e7298a"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-json",
                   default="/tmp/interaction-branching/spectral_dimension.json")
    p.add_argument("--out-png",
                   default="docs/source/quantum-experiments/figures/"
                           "poisson_delaunay_spectral_dimension.png")
    args = p.parse_args()

    with open(args.in_json) as f:
        records = json.load(f)["records"]

    # group by m/g -> sorted list of (N, peak_mean, peak_std, dinf_mean, dinf_std)
    by_mg = defaultdict(list)
    for rec in records:
        s = rec["summary"]
        by_mg[rec["m_over_g"]].append((
            rec["N"],
            s["peak_dS"]["mean"], s["peak_dS"]["std"],
            s["D_infinity"]["mean"], s["D_infinity"]["std"],
        ))
    for mg in by_mg:
        by_mg[mg].sort()

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))

    for (mg, rows), color in zip(sorted(by_mg.items()), COLORS):
        ns = np.array([r[0] for r in rows])
        peak_m = np.array([r[1] for r in rows])
        peak_s = np.array([r[2] for r in rows])
        dinf_m = np.array([r[3] for r in rows])
        dinf_s = np.array([r[4] for r in rows])
        label = f"m/g = {mg:g}"
        ax0.plot(ns, peak_m, "o-", color=color, label=label)
        ax0.fill_between(ns, peak_m - peak_s, peak_m + peak_s,
                         color=color, alpha=0.18)
        ax1.plot(ns, dinf_m, "o-", color=color, label=label)
        ax1.fill_between(ns, dinf_m - dinf_s, dinf_m + dinf_s,
                         color=color, alpha=0.18)

    for ax in (ax0, ax1):
        ax.set_xlabel("N (Schwinger sites = Poisson points)")
        ax.axhline(2.0, ls=":", color="grey", lw=0.8)
        ax.axhline(4.0, ls=":", color="grey", lw=0.8)
        ax.legend(fontsize=8)
    ax0.set_ylabel(r"peak $D_S$")
    ax0.set_title(r"Peak spectral dimension vs $N$")
    ax1.set_ylabel(r"Ambjorn-Loll $D_\infty$")
    ax1.set_title(r"$D_\infty$ fit vs $N$")

    fig.suptitle("Coned Poisson-Delaunay MI complex: spectral dimension "
                 "(temporally-connected parameters)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out_png, dpi=130)
    print(f"[wrote] {args.out_png}")


if __name__ == "__main__":
    main()
