"""Generate a focused single-panel plot of the T-scan asymptotic
approach to D_S = 4."""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SRC = Path("/tmp/interaction-history/issue10_finite_size.json")
OUT = Path("/home/andrew/tessera/docs/source/quantum-experiments"
           "/figures/v02_t_scan_asymptote.png")


def main():
    with open(SRC) as f:
        data = json.load(f)
    B = [r for r in data["records"] if r["scan"] == "B_T_scan"]
    Ts = sorted({r["T"] for r in B})
    means = []
    stds = []
    for T in Ts:
        ps = np.array([r["peak_dS"] for r in B if r["T"] == T])
        means.append(ps.mean())
        stds.append(ps.std())
    means = np.array(means)
    stds = np.array(stds)

    # Geometric extrapolation: deltas are -0.177, -0.132, -0.067; ratio
    # ~0.5. Asymptotic value = last + final_delta / (1 - ratio).
    deltas = np.diff(means)
    ratio = float(np.mean(deltas[1:] / deltas[:-1]))
    asymptote = means[-1] + deltas[-1] / (1.0 - ratio)

    fig, ax = plt.subplots(figsize=(9, 6))

    for T in Ts:
        ps = np.array([r["peak_dS"] for r in B if r["T"] == T])
        ax.scatter([T] * len(ps), ps,
                   s=24, alpha=0.4, color="#d62728", edgecolor="none")

    ax.errorbar(Ts, means, yerr=stds,
                fmt="o-", color="#d62728", lw=2.2,
                capsize=5, markersize=10,
                label="measured (N = 8, β = 3×10⁻⁴, 10 seeds per T)")

    # Asymptote line.
    T_grid = np.array([Ts[0] * 0.5, 1e6])
    ax.plot(T_grid, [asymptote, asymptote],
            color="#1f77b4", lw=1.2, ls="-.",
            label=f"geometric extrapolation → D_S(T → ∞) ≈ {asymptote:.2f}")
    ax.axhline(4.0, color="gray", lw=1.0, ls="--",
               label="H_DS4 target: D_S = 4")

    ax.set_xscale("log")
    ax.set_xlim(Ts[0] * 0.7, 1e6)
    ax.set_ylim(3.9, 4.75)
    ax.set_xlabel("T (cells)", fontsize=12)
    ax.set_ylabel("peak D_S  (mean ± std)", fontsize=12)
    ax.set_title("Charged-Cartan v0.2 plateau: T-scaling asymptote\n"
                 "(N = 8, β = 3×10⁻⁴, γ_CP = 0)",
                 fontsize=12)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)

    # Annotate each measured point with its mean value.
    for T, m in zip(Ts, means):
        ax.annotate(f"{m:.3f}",
                    xy=(T, m), xytext=(8, 6),
                    textcoords="offset points",
                    fontsize=9, color="#7f1d1d")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"[wrote] {OUT}")


if __name__ == "__main__":
    main()
