"""Render the interaction-history β-scan figure.

Left panel: peak spectral dimension vs β (mean ± std over Poisson
layouts), with the D_S = 4 target line. Right panel: the mean
interaction count accepted vs β — the growth-vs-suppression axis.

Usage::

    python examples/quantum/plot_interaction_history_spectral_dimension.py \\
        --in-json /tmp/interaction-history/result.json \\
        --out-png docs/source/quantum-experiments/figures/interaction_history_spectral_dimension.png
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-json",
                   default="/tmp/interaction-history/result.json")
    p.add_argument("--out-png",
                   default="docs/source/quantum-experiments/figures/"
                           "interaction_history_spectral_dimension.png")
    args = p.parse_args()

    with open(args.in_json) as f:
        data = json.load(f)
    records = data["records"]

    betas = np.array([r["beta"] for r in records])
    peak_mean = np.array([r["peak_dS_mean"] for r in records])
    peak_std = np.array([r["peak_dS_std"] for r in records])
    counts = np.array([r["mean_interaction_count"] for r in records])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax0.errorbar(betas, peak_mean, yerr=peak_std, fmt="o-", color="#2c7fb8",
                 capsize=3, label="peak $D_S$")
    ax0.axhline(4.0, ls="--", color="#d95f02", label="$D_S = 4$ target")
    ax0.axhline(2.0, ls=":", color="grey", lw=0.8)
    cross = data.get("dS_eq_4_beta")
    if cross is not None:
        ax0.axvline(cross, ls="-.", color="#d95f02",
                    label=f"$D_S=4$ at β≈{cross:.2e}")
    ax0.set_xscale("log")
    ax0.set_xlabel(r"$\beta$ (inverse temperature)")
    ax0.set_ylabel(r"peak $D_S$")
    ax0.set_title("Emergent spectral dimension vs β")
    ax0.legend(fontsize=8)

    ax1.semilogx(betas, counts, "s-", color="#1b9e77")
    ax1.set_xlabel(r"$\beta$ (inverse temperature)")
    ax1.set_ylabel("mean accepted interactions")
    ax1.set_title("Growth vs suppression")

    fig.suptitle("Interaction-history Monte Carlo: locating $D_S = 4$",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out_png, dpi=130)
    print(f"[wrote] {args.out_png}")


if __name__ == "__main__":
    main()
