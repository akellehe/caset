"""Plot the Choi sanity scan: peak D_S and Q-conservation
side-by-side for choi off vs choi on."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SRC = Path("/tmp/interaction-history/choi_sanity_scan.json")
OUT = Path("/home/andrew/tessera/docs/source/quantum-experiments/figures"
           "/choi_sigma_ab_sanity.png")


def main():
    with open(SRC) as f:
        data = json.load(f)
    rs = data["records"]

    groups = defaultdict(list)
    for r in rs:
        groups[(r["label"], r["beta"])].append(r)
    betas = sorted({r["beta"] for r in rs})

    fig, (axD, axQ) = plt.subplots(1, 2, figsize=(13, 5.5))

    style = {
        "choi_off": ("#1f77b4", "Choi OFF (v0.2 baseline)"),
        "choi_on":  ("#d62728", "Choi ON (#16)"),
    }

    # peak D_S vs β
    for key, color_label in style.items():
        color, label = color_label
        peaks_per_b = [[r["peak_dS"] for r in groups[(key, b)]]
                       for b in betas]
        means = np.array([np.mean(p) for p in peaks_per_b])
        stds  = np.array([np.std(p)  for p in peaks_per_b])
        axD.plot(betas, means, "o-", color=color, lw=1.8,
                 markersize=8, label=label)
        axD.fill_between(betas, means - stds, means + stds,
                          color=color, alpha=0.15)
        for b, ps in zip(betas, peaks_per_b):
            axD.scatter([b] * len(ps), ps, s=20, alpha=0.4,
                        color=color, edgecolor="none")
    axD.axhline(4.0, color="gray", lw=0.8, ls="--",
                 label="H_DS4 target")
    axD.set_xscale("log"); axD.set_yscale("log")
    axD.set_xlabel("β (inverse temperature)")
    axD.set_ylabel("peak $D_S$ (mean ± std, 5 seeds)")
    axD.set_title("Plateau preserved by Choi fix\n"
                  "(N=8, T=2500, γ_CP=0)")
    axD.grid(True, which="both", ls=":", alpha=0.4)
    axD.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # |Q_global| vs β (log scale to show the 4-orders-of-magnitude shrink)
    for key, color_label in style.items():
        color, label = color_label
        for b in betas:
            qs = [abs(r["q_global"]) for r in groups[(key, b)]]
            # Clip to 1e-16 floor for log display
            qs = [max(q, 1e-16) for q in qs]
            axQ.scatter([b] * len(qs), qs, s=30, alpha=0.55,
                        color=color, edgecolor="none",
                        label=label if b == betas[0] else None)
    axQ.set_xscale("log"); axQ.set_yscale("log")
    axQ.set_xlabel("β (inverse temperature)")
    axQ.set_ylabel("|Q_global|  (one dot per seed)")
    axQ.set_title("Q-conservation: integer drift ($\sim$1–20)\n"
                  "→ float noise ($\sim$10⁻⁵)")
    axQ.axhline(1.0, color="gray", lw=0.6, ls=":",
                 label="|Q| = 1 (integer drift)")
    axQ.axhline(1e-6, color="gray", lw=0.6, ls="--",
                 label="|Q| = 10⁻⁶ (test threshold)")
    axQ.set_ylim(1e-16, 1e2)
    axQ.grid(True, which="both", ls=":", alpha=0.4)
    axQ.legend(loc="lower left", fontsize=8, framealpha=0.9)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"[wrote] {OUT}")


if __name__ == "__main__":
    main()
