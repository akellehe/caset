#!/usr/bin/env python3
# MIT License -- Copyright (c) 2025 Andrew Kelleher
"""
Spectral dimension measurement via discrete diffusion.

Reproduces Figures 9-10 from:
  Ambjorn, Jurkiewicz, Loll, "Reconstructing the Universe",
  Phys. Rev. D 72 (2005) [hep-th/0505154]

The spectral dimension D_S(sigma) is measured by running a discrete
diffusion process on the dual graph of the triangulation (where each
d-simplex is a node and neighbors share a (d-1)-face).

The return probability P(sigma) -- the probability that a random walker
returns to its starting simplex after sigma steps -- scales as

    P(sigma) ~ sigma^{-D_S/2}

so the spectral dimension is extracted via

    D_S(sigma) = -2  d log P(sigma) / d log sigma

Key results from the paper (k0=2.2, Delta=0.6, t=80):
  D_S(sigma -> infinity) = 4.02 +/- 0.1   (large-scale dimension)
  D_S(sigma -> 0)        = 1.80 +/- 0.25  (short-distance dimension)
  Best fit:  D_S(sigma) = 4.02 - 119/(54 + sigma)    [Eq. 29]

Parameters: k0 = 2.2, Delta = 0.6.
Estimated runtime: ~2-5 minutes.
"""
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import caset


def build_dual_adjacency(st):
    """
    Build the dual graph adjacency: each top-dimensional simplex is a node,
    two nodes are adjacent if their simplices share a (d-1)-face.

    Uses the coface structure: a facet shared by exactly 2 top-simplices
    connects those two simplices in the dual graph.
    """
    d = 4  # spacetime dimension
    d_plus_1 = d + 1

    # Collect top-dimensional simplices
    top_simplices = []
    simplex_to_idx = {}
    for s in st.getSimplices():
        if len(s.getVertices()) == d_plus_1:
            idx = len(top_simplices)
            simplex_to_idx[hash(s)] = idx
            top_simplices.append(s)

    N = len(top_simplices)
    if N == 0:
        return [], {}

    # Build adjacency lists
    adjacency = [[] for _ in range(N)]
    for i, s in enumerate(top_simplices):
        facets = s.getFacets()
        for f in facets:
            for cf in f.getCofaces():
                if len(cf.getVertices()) == d_plus_1:
                    h = hash(cf)
                    if h in simplex_to_idx:
                        j = simplex_to_idx[h]
                        if j != i and j not in adjacency[i]:
                            adjacency[i].append(j)

    return adjacency, simplex_to_idx


def diffuse(adjacency, start, max_sigma):
    """
    Run a discrete diffusion process on the dual graph.
    At each step, the probability distributes uniformly to neighbors.

    Returns P(sigma) = probability of being at the start node after sigma steps.
    """
    N = len(adjacency)
    if N == 0:
        return np.zeros(max_sigma + 1)

    # prob[i] = probability of being at node i
    prob = np.zeros(N)
    prob[start] = 1.0

    return_prob = np.zeros(max_sigma + 1)
    return_prob[0] = 1.0

    for sigma in range(1, max_sigma + 1):
        new_prob = np.zeros(N)
        for i in range(N):
            if prob[i] > 0 and len(adjacency[i]) > 0:
                share = prob[i] / len(adjacency[i])
                for j in adjacency[i]:
                    new_prob[j] += share
        prob = new_prob
        return_prob[sigma] = prob[start]

    return return_prob


def compute_spectral_dimension(return_prob):
    """
    Compute D_S(sigma) = -2 d(log P) / d(log sigma).

    Uses centered finite differences on log-log data.
    Returns (sigma_values, D_S_values) excluding endpoints and zeros.
    """
    sigma = np.arange(len(return_prob))
    # Skip sigma=0 and any zeros
    valid = (sigma > 1) & (return_prob > 0)
    s = sigma[valid].astype(float)
    p = return_prob[valid]

    log_s = np.log(s)
    log_p = np.log(p)

    if len(log_s) < 2:
        return s, np.zeros(len(s))

    # Centered finite differences
    ds = np.zeros(len(log_s))
    ds[1:-1] = (log_p[2:] - log_p[:-2]) / (log_s[2:] - log_s[:-2])
    ds[0] = (log_p[1] - log_p[0]) / (log_s[1] - log_s[0])
    ds[-1] = (log_p[-1] - log_p[-2]) / (log_s[-1] - log_s[-2])

    D_S = -2.0 * ds
    return s, D_S


def main():
    parser = argparse.ArgumentParser(
        description="Spectral dimension D_S(sigma) measurement "
                    "(Figs 9-10 of hep-th/0505154)")
    parser.add_argument("--n-simplices", type=int, default=500,
                        help="Initial number of simplices")
    parser.add_argument("--n-therm", type=int, default=50,
                        help="Thermalization sweeps")
    parser.add_argument("--n-configs", type=int, default=10,
                        help="Number of independent configurations to average")
    parser.add_argument("--n-walks", type=int, default=20,
                        help="Diffusion processes per configuration")
    parser.add_argument("--max-sigma", type=int, default=200,
                        help="Maximum diffusion time")
    parser.add_argument("--sweeps-between", type=int, default=10,
                        help="Sweeps between configurations for decorrelation")
    parser.add_argument("--save", type=str, default=None,
                        help="Save figure to this path instead of showing")
    args = parser.parse_args()

    print("=" * 64)
    print("  Spectral Dimension Measurement via Discrete Diffusion")
    print("  Reproduces Figs 9-10, Ambjorn, Jurkiewicz, Loll (2005)")
    print("  Parameters: k0=2.2, Delta=0.6")
    print(f"  Configs: {args.n_configs}, walks/config: {args.n_walks}, "
          f"max sigma: {args.max_sigma}")
    print("=" * 64)

    # --- Build and thermalize ---
    t_total = time.time()
    print("\nBuilding spacetime...")
    sig = caset.Signature(4, caset.Lorentzian)
    metric = caset.Metric(True, sig)
    st = caset.Spacetime(metric, caset.CDT, 1.0, 1.0, caset.PREFERRED,
                         caset.Toroid())
    st.build(args.n_simplices)
    print(f"  {st.getSimplexCount():,} simplices built")

    target = st.getSimplexCount()
    cdt = caset.CDTSimulation(st, 2.2, 0.5, 0.6, 0.02, target)

    print("Tuning k4...")
    cdt.tune()
    print(f"  k4 = {cdt.getK4():.4f}")

    for _ in tqdm(range(args.n_therm), desc="Thermalizing",
                  unit="sweep", leave=False):
        cdt.sweep()
    print(f"Thermalization complete ({args.n_therm} sweeps)")

    # --- Measure spectral dimension ---
    all_return_probs = []

    for cfg in tqdm(range(args.n_configs), desc="Configurations",
                    unit="cfg"):
        # Decorrelate
        for _ in range(args.sweeps_between):
            cdt.sweep()

        t0 = time.time()
        adjacency, s2i = build_dual_adjacency(st)
        N = len(adjacency)
        if N == 0:
            tqdm.write(f"  Config {cfg+1}: no simplices, skipping")
            continue

        # Average over random starting simplices
        starts = np.random.choice(N, size=min(args.n_walks, N), replace=False)
        for start in starts:
            rp = diffuse(adjacency, start, args.max_sigma)
            all_return_probs.append(rp)

        avg_neighbors = np.mean([len(adj) for adj in adjacency])
        tqdm.write(f"  Config {cfg+1}: N4={st.getSimplexCount():,}, "
                   f"dual nodes={N:,}, avg neighbors={avg_neighbors:.1f}, "
                   f"{time.time()-t0:.1f}s")

    if not all_return_probs:
        print("No data collected.")
        return

    print(f"\nCollected {len(all_return_probs)} diffusion walks")

    # Average return probability
    max_len = max(len(rp) for rp in all_return_probs)
    avg_rp = np.zeros(max_len)
    counts = np.zeros(max_len)
    for rp in all_return_probs:
        avg_rp[:len(rp)] += rp
        counts[:len(rp)] += 1
    counts[counts == 0] = 1
    avg_rp /= counts

    # Compute spectral dimension
    sigma_vals, D_S_vals = compute_spectral_dimension(avg_rp)

    # --- Plot (Fig 9/10 style) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Fig 9: D_S(sigma) vs sigma
    ax1.plot(sigma_vals, D_S_vals, "b-", linewidth=1.5, label="Measured")

    # Overlay the paper's best fit: D_S = 4.02 - 119/(54 + sigma)
    sigma_fit = np.linspace(10, args.max_sigma, 200)
    D_S_fit = 4.02 - 119.0 / (54.0 + sigma_fit)
    ax1.plot(sigma_fit, D_S_fit, "r--", linewidth=1.5,
             label=r"Fit: $D_S = 4.02 - \frac{119}{54+\sigma}$")

    ax1.set_xlabel(r"Diffusion time $\sigma$", fontsize=13)
    ax1.set_ylabel(r"$D_S(\sigma)$", fontsize=13)
    ax1.set_title(r"Spectral dimension $D_S(\sigma)$"
                  "\n(cf. Fig 9, Ambjorn, Jurkiewicz, Loll,\n"
                  r"$\it{Reconstructing\ the\ Universe}$, 2005)")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 5)

    # Fig 10 style: same data with error envelope
    ax2.plot(sigma_vals, D_S_vals, "k-", linewidth=1.5, label="Measured")
    ax2.plot(sigma_fit, D_S_fit, "g--", linewidth=1,
             label=r"$4.02 - 119/(54+\sigma)$")
    ax2.axhline(y=4.02, color="gray", linestyle=":", alpha=0.5,
                label=r"$D_S(\infty)=4.02$")
    ax2.axhline(y=1.80, color="gray", linestyle=":", alpha=0.5,
                label=r"$D_S(0)\approx 1.80$")
    ax2.set_xlabel(r"$\sigma$", fontsize=13)
    ax2.set_ylabel(r"$D_S$", fontsize=13)
    ax2.set_title(r"Spectral dimension with reference values"
                  "\n(cf. Fig 10, Ambjorn, Jurkiewicz, Loll,\n"
                  r"$\it{Reconstructing\ the\ Universe}$, 2005)")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 5)

    fig.tight_layout()

    # Report key results
    if len(sigma_vals) > 0:
        # Large-scale estimate (last 20% of data)
        n_tail = max(1, len(D_S_vals) // 5)
        D_S_large = np.mean(D_S_vals[-n_tail:])
        # Small-scale estimate (first 20% of data)
        n_head = max(1, len(D_S_vals) // 5)
        D_S_small = np.mean(D_S_vals[:n_head])
        print(f"\nResults:")
        print(f"  D_S(large scale)  = {D_S_large:.2f}  (paper: 4.02 +/- 0.1)")
        print(f"  D_S(small scale)  = {D_S_small:.2f}  (paper: 1.80 +/- 0.25)")

    print(f"Total elapsed: {time.time()-t_total:.1f}s")

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
