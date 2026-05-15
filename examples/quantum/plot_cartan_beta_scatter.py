"""Plot the Cartan/local-frame β-scatter scan.

Loads /tmp/interaction-history/cartan_beta_scatter.json and produces a
two-panel figure:
  • top: peak D_S vs β scatter (one dot per seed), with the per-β mean
    and ±1σ band, and the phase boundaries shaded.
  • bottom: σ_peak vs β scatter (the scale at which peak D_S occurs),
    revealing the extended-vs-localized phase split.

Output: docs/source/quantum-experiments/figures/cartan_beta_scatter.png
"""
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

JSON_PATH = Path("/tmp/interaction-history/cartan_beta_scatter.json")
REPO = Path(__file__).resolve().parents[2]
OUT_PATH = (REPO / "docs/source/quantum-experiments/figures"
            / "cartan_beta_scatter.png")


def main():
    with open(JSON_PATH) as f:
        data = json.load(f)

    records = data["records"]
    betas = sorted({r["beta"] for r in records})

    sigma_max = max(r["peak_sigma"] for r in records)
    rows = []
    for b in betas:
        rs = [r for r in records if r["beta"] == b]
        peaks = np.array([r["peak_dS"] for r in rs], dtype=float)
        sigs  = np.array([r["peak_sigma"] for r in rs], dtype=float)
        cells = np.array([r["cells"] for r in rs], dtype=float)
        sat   = sigs >= sigma_max * 0.999  # σ-saturated mask
        rows.append({"beta": b, "peaks": peaks, "sigs": sigs,
                     "cells": cells, "sat": sat})

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5),
                                   sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})

    # --- top: peak D_S vs β ---
    # Plot saturated (lower-bound) points hollow, true-peak points filled.
    for r in rows:
        bs = [r["beta"]] * len(r["peaks"])
        notsat = ~r["sat"]
        if notsat.any():
            ax1.scatter(np.array(bs)[notsat], r["peaks"][notsat],
                        s=22, alpha=0.55, color="#1f77b4",
                        edgecolor="none",
                        label="true peak (D_S(σ) turned over)"
                        if r is rows[0] else None)
        if r["sat"].any():
            ax1.scatter(np.array(bs)[r["sat"]], r["peaks"][r["sat"]],
                        s=22, alpha=0.55,
                        facecolor="none", edgecolor="#1f77b4",
                        linewidths=1.2,
                        marker="^",
                        label="σ-saturated (lower bound on true peak)"
                        if r is rows[0] else None)
    # mean line
    bs = np.array([r["beta"] for r in rows])
    ms = np.array([r["peaks"].mean() for r in rows])
    sd = np.array([r["peaks"].std()  for r in rows])
    ax1.plot(bs, ms, color="#d62728", lw=1.6, label="per-β mean")
    ax1.fill_between(bs, ms - sd, ms + sd, color="#d62728", alpha=0.15,
                     label="±1σ")
    ax1.axhline(4.0, color="gray", lw=0.8, ls="--",
                label="target (D_S = 4)")
    ax1.axhline(0.635, color="black", lw=0.6, ls=":",
                label="marginal-model ceiling (0.635)")

    # Approximate phase windows derived from observation:
    ax1.axvspan(1e-4, 3e-4,  color="#bbbbbb", alpha=0.15)
    ax1.axvspan(3e-4, 1.5e-3, color="#ffd180", alpha=0.18)
    ax1.axvspan(1.5e-3, 5e-3, color="#bbdefb", alpha=0.18)

    ax1.set_xscale("log")
    ax1.set_ylabel("peak $D_S$ (one dot per seed)")
    ax1.set_title("Cartan / local-frame model — peak spectral dimension "
                  "vs β\n(N=8, T=3000, 10 seeds per β)")
    ax1.grid(True, which="both", ls=":", alpha=0.4)
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=9)

    # --- bottom: σ_peak vs β ---
    for r in rows:
        ax2.scatter([r["beta"]] * len(r["sigs"]), r["sigs"],
                    s=18, alpha=0.45, color="#2ca02c", edgecolor="none")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("β (inverse temperature)")
    ax2.set_ylabel("σ at peak $D_S$ (heat-kernel scale)")
    ax2.grid(True, which="both", ls=":", alpha=0.4)
    ax2.axhline(1e6, color="gray", lw=0.6, ls="--",
                label="σ-grid ceiling (under-measurement above)")

    # Phase bands repeated:
    ax2.axvspan(1e-4, 3e-4,  color="#bbbbbb", alpha=0.15)
    ax2.axvspan(3e-4, 1.5e-3, color="#ffd180", alpha=0.18)
    ax2.axvspan(1.5e-3, 5e-3, color="#bbdefb", alpha=0.18)

    # Phase labels at top of σ axis
    ymax = ax2.get_ylim()[1]
    for x, txt in [(np.sqrt(1e-4 * 3e-4), "zero-D"),
                   (np.sqrt(3e-4 * 1.5e-3),
                    "extended-D\n(D_S ≈ 1)"),
                   (np.sqrt(1.5e-3 * 5e-3),
                    "high-D / σ-saturated\n(D_S ≥ 4)")]:
        ax2.text(x, ymax * 0.5, txt, ha="center", va="top",
                 fontsize=9, color="#555555")
    ax2.legend(loc="lower right", framealpha=0.9, fontsize=9)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"[wrote] {OUT_PATH}")

    # Also print summary
    print("\nPer-β summary:")
    print(f"  {'β':>9}  {'cells_mean':>10}  {'D_S mean ± std':>18}  "
          f"{'D_S max':>8}")
    for r in rows:
        print(f"  {r['beta']:9.3e}  {r['cells'].mean():10.0f}  "
              f"{r['peaks'].mean():7.3f} ± {r['peaks'].std():5.3f}      "
              f"{r['peaks'].max():6.3f}")


if __name__ == "__main__":
    main()
