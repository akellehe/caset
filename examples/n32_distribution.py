#!/usr/bin/env python3
# MIT License -- Copyright (c) 2025 Andrew Kelleher
"""
Distribution of N_4^{(3,2)} at fixed N_4^{(4,1)}.

Reproduces Figure 2 from:
  Ambjorn, Jurkiewicz, Loll, "Reconstructing the Universe",
  Phys. Rev. D 72 (2005) [hep-th/0505154]

In 4D CDT there are two types of four-simplices: (4,1)-type (with 4
vertices at time tau and 1 at tau+1) and (3,2)-type (with 3 at tau and
2 at tau+1).  The volume-fixing term in the action constrains
N_4^{(4,1)}, but the number of (3,2)-simplices N_4^{(3,2)} fluctuates
freely.  Figure 2 shows that the distribution of N_4^{(3,2)} is sharply
peaked, demonstrating that the two types of simplices are strongly
correlated in the de Sitter phase.

Parameters: k0 = 2.2, Delta = 0.6.
Estimated runtime: ~1-3 minutes.
"""
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt

import caset


def main():
    parser = argparse.ArgumentParser(
        description="N_4^{(3,2)} distribution at fixed N_4^{(4,1)} "
                    "(Fig 2 of hep-th/0505154)")
    parser.add_argument("--n-therm", type=int, default=50)
    parser.add_argument("--n-meas", type=int, default=200)
    parser.add_argument("--meas-interval", type=int, default=3)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    # Run at multiple target volumes
    target_n41_values = [200, 500, 1000]
    colors = ["black", "red", "green", "blue"]

    fig, ax = plt.subplots(figsize=(10, 7))

    for idx, target_n41 in enumerate(target_n41_values):
        # Build with roughly 2x target since N41 ~ half of total
        n_build = target_n41 * 2
        print(f"\nTarget N41 ~ {target_n41} (build={n_build})...")

        sig = caset.Signature(4, caset.Lorentzian)
        metric = caset.Metric(True, sig)
        st = caset.Spacetime(metric, caset.CDT, 1.0, 1.0, caset.PREFERRED,
                             caset.Toroid())
        st.build(n_build)

        target = st.getSimplexCount()
        cdt = caset.CDTSimulation(st, 2.2, 0.5, 0.6, 0.02, target)

        # Thermalize
        for _ in range(args.n_therm):
            cdt.sweep()

        # Collect N32 samples
        n32_samples = []
        n41_samples = []
        for _ in range(args.n_meas):
            for _ in range(args.meas_interval):
                cdt.sweep()
            n32_samples.append(st.getN32())
            n41_samples.append(st.getN41())

        n32_arr = np.array(n32_samples, dtype=float)
        n41_arr = np.array(n41_samples, dtype=float)

        print(f"  N41: mean={n41_arr.mean():.0f}, std={n41_arr.std():.0f}")
        print(f"  N32: mean={n32_arr.mean():.0f}, std={n32_arr.std():.0f}")

        # Plot histogram (normalized)
        if len(n32_arr) > 10 and n32_arr.std() > 0:
            hist, edges = np.histogram(n32_arr, bins=30, density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            ax.plot(centers, hist, "-o", markersize=3,
                    color=colors[idx % len(colors)],
                    label=fr"$\tilde{{N}}_4 \approx {int(n41_arr.mean())}$")

    ax.set_xlabel(r"$N_4^{(3,2)}$", fontsize=14)
    ax.set_ylabel(r"$P(N_4^{(3,2)})$", fontsize=14)
    ax.set_title(
        r"Distribution of $N_4^{(3,2)}$ at fixed $N_4^{(4,1)}$"
        "\n"r"($\kappa_0=2.2$, $\Delta=0.6$; cf. Fig 2, Ambjorn et al. 2005)")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
