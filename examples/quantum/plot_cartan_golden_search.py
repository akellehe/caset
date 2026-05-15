"""Plot the golden-section search trajectory.

Loads /tmp/interaction-history/cartan_golden_search.json. Two-panel:

  • top: mean ± std peak D_S at each evaluated β, with individual seed
    dots overlaid and the iteration sequence shown by colour/order.
  • bottom: the D_S(σ) curves at the four highest-mean-D_S evaluations,
    to see whether the heat-kernel dimension actually plateaus near 4
    or just keeps climbing.

Output: docs/source/quantum-experiments/figures/cartan_golden_search.png
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis

JSON_PATH = Path("/tmp/interaction-history/cartan_golden_search.json")
REPO = Path(__file__).resolve().parents[2]
OUT_PATH = (REPO / "docs/source/quantum-experiments/figures"
            / "cartan_golden_search.png")


def main():
    with open(JSON_PATH) as f:
        data = json.load(f)

    history = data["history"]
    sigmas = data["sigmas"]
    n_evals = len(history)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9))

    # --- top: trajectory ---
    colors = viridis(np.linspace(0, 1, n_evals))
    for k, h in enumerate(history):
        beta = h["beta"]
        peaks = np.array([r["peak_dS"] for r in h["rows"]], dtype=float)
        ax1.scatter([beta] * len(peaks), peaks, s=16,
                    color=colors[k], alpha=0.45, edgecolor="none")
        ax1.errorbar([beta], [h["mean_peak_dS"]],
                     yerr=[h["std_peak_dS"]],
                     fmt="o", color=colors[k],
                     markersize=9, markeredgecolor="black",
                     markeredgewidth=0.8, lw=1.6,
                     label=f"eval #{h['eval_idx']:>2}  β={beta:.3e}",
                     capsize=4)

    ax1.axhline(4.0, color="gray", lw=0.8, ls="--",
                label="target (D_S = 4)")
    ax1.axhline(0.635, color="black", lw=0.6, ls=":",
                label="marginal-model ceiling (0.635)")
    ax1.set_xscale("log")
    ax1.set_xlabel("β (inverse temperature)")
    ax1.set_ylabel("peak $D_S$ at extended σ-grid (σ_max = 10¹⁰)")
    ax1.set_title("Golden-section search for peak $D_S$ "
                  "— Cartan/local-frame model (10 seeds per eval)")
    ax1.grid(True, which="both", ls=":", alpha=0.4)
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=8, ncol=2)

    # --- bottom: D_S(σ) curves at top-4 by mean ---
    top4 = sorted(history, key=lambda h: h["mean_peak_dS"],
                  reverse=True)[:4]
    for h in top4:
        # average D_S(σ) across seeds (handle non-finite)
        m = np.array([r["dS"] for r in h["rows"]], dtype=float)
        finite = np.isfinite(m).all(axis=0)
        ax2.plot(np.array(sigmas), m.mean(axis=0),
                 lw=1.7,
                 label=f"β={h['beta']:.3e}  "
                       f"mean peak D_S={h['mean_peak_dS']:.2f}")
        # shaded ±1σ
        ax2.fill_between(np.array(sigmas),
                         m.mean(axis=0) - m.std(axis=0),
                         m.mean(axis=0) + m.std(axis=0),
                         alpha=0.15)

    ax2.axhline(4.0, color="gray", lw=0.8, ls="--")
    ax2.set_xscale("log")
    ax2.set_xlabel("σ (heat-kernel diffusion time)")
    ax2.set_ylabel("$D_S(\\sigma)$ (mean over 10 seeds)")
    ax2.set_title("$D_S(\\sigma)$ at the four highest-mean-peak β values")
    ax2.grid(True, which="both", ls=":", alpha=0.4)
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=9)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"[wrote] {OUT_PATH}")

    # Console summary
    best = data.get("best", {})
    print(f"\nBest evaluation: β={best.get('beta', '?'):.4e}  "
          f"mean peak D_S = {best.get('mean_peak_dS', '?'):.3f} "
          f"± {best.get('std_peak_dS', '?'):.3f}")
    print(f"\nAll evaluations (by eval order):")
    for h in history:
        print(f"  #{h['eval_idx']:>2}  β={h['beta']:.4e}  "
              f"mean={h['mean_peak_dS']:.3f}±{h['std_peak_dS']:.3f}  "
              f"sat={h['n_saturated']}/10  cells_mean={h['mean_cells']:.0f}")


if __name__ == "__main__":
    main()
