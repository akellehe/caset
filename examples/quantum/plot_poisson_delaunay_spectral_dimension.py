"""Render the Poisson-Delaunay spectral-dimension figure.

Left panel: D_S(sigma) per Schwinger m/g -- mean over Poisson layouts
with a +/- std band. Right panel: peak D_S per m/g with error bars.

Usage::

    python examples/quantum/plot_poisson_delaunay_spectral_dimension.py \\
        --in-json /tmp/interaction-branching/spectral_dimension.json \\
        --out-png docs/source/quantum-experiments/figures/poisson_delaunay_spectral_dimension.png
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#1b9e77", "#7570b3", "#d95f02"]


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

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))

    peak_means, peak_stds, labels = [], [], []
    for rec, color in zip(records, COLORS):
        sigmas = np.array(rec["sigmas"])
        curves = np.array([r["dS_smoothed"] for r in rec["runs"]],
                          dtype=np.float64)
        curves = np.where(np.isfinite(curves), curves, np.nan)
        mean = np.nanmean(curves, axis=0)
        std = np.nanstd(curves, axis=0)
        label = f"m/g = {rec['m_over_g']:g}"
        ax0.plot(sigmas, mean, "-", color=color, label=label)
        ax0.fill_between(sigmas, mean - std, mean + std, color=color,
                         alpha=0.18)
        peak_means.append(rec["summary"]["peak_dS"]["mean"])
        peak_stds.append(rec["summary"]["peak_dS"]["std"])
        labels.append(f"{rec['m_over_g']:g}")

    ax0.set_xscale("log")
    ax0.set_xlabel(r"$\sigma$ (diffusion time)")
    ax0.set_ylabel(r"$D_S(\sigma)$")
    ax0.set_title("Spectral dimension of the coned Poisson-Delaunay MI complex")
    ax0.axhline(2.0, ls=":", color="grey", lw=0.8)
    ax0.axhline(4.0, ls=":", color="grey", lw=0.8)
    ax0.legend(fontsize=8)

    x = np.arange(len(records))
    ax1.bar(x, peak_means, 0.5, yerr=peak_stds, capsize=4,
            color=COLORS[:len(records)], edgecolor="white")
    for xi, m, s in zip(x, peak_means, peak_stds):
        ax1.text(xi, m + s + 0.1, f"{m:.2f}", ha="center", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel(r"Schwinger $m/g$")
    ax1.set_ylabel(r"peak $D_S$")
    ax1.set_title("Peak spectral dimension")
    ax1.axhline(2.0, ls=":", color="grey", lw=0.8)
    ax1.axhline(4.0, ls=":", color="grey", lw=0.8)
    ax1.set_ylim(0, max(4.4, max(peak_means) + max(peak_stds) + 0.4))

    fig.suptitle("Poisson-Delaunay coned complex: heat-kernel spectral "
                 "dimension", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out_png, dpi=130)
    print(f"[wrote] {args.out_png}")


if __name__ == "__main__":
    main()
