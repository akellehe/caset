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
"""
import argparse
import multiprocessing
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from tqdm import tqdm

import caset


# ---------------------------------------------------------------------------
# Dual-graph construction
# ---------------------------------------------------------------------------

def build_transition_matrix(st):
    """Build the sparse row-stochastic transition matrix for diffusion on
    the dual graph of the triangulation.

    Each top-dimensional simplex is a node.  Two nodes are adjacent if their
    simplices share a (d-1)-face.  The transition probability is uniform over
    neighbours: T[j,i] = 1/deg(i) for every neighbour j of i.

    Returns (T, N) where T is a CSC sparse matrix and N the number of nodes.
    """
    d_plus_1 = 5  # 4D

    top_simplices = []
    simplex_to_idx = {}
    for s in st.getSimplices():
        if len(s.getVertices()) == d_plus_1:
            simplex_to_idx[hash(s)] = len(top_simplices)
            top_simplices.append(s)

    N = len(top_simplices)
    if N == 0:
        return None, 0

    rows, cols = [], []
    for i, s in enumerate(top_simplices):
        for f in s.getFacets():
            for cf in f.getCofaces():
                if len(cf.getVertices()) == d_plus_1:
                    h = hash(cf)
                    if h in simplex_to_idx:
                        j = simplex_to_idx[h]
                        if j != i:
                            rows.append(j)
                            cols.append(i)

    if not rows:
        return None, N

    # Build adjacency, remove duplicate edges, then normalise
    A = sparse.csc_matrix((np.ones(len(rows)), (rows, cols)),
                          shape=(N, N))
    # Eliminate duplicates (summed) – set all nonzeros to 1
    A.data[:] = 1.0
    # Row-stochastic: divide each column by its sum (= degree of that node)
    deg = np.array(A.sum(axis=0)).ravel()
    deg[deg == 0] = 1.0
    T = A @ sparse.diags(1.0 / deg)
    return T.tocsc(), N


def diffuse_sparse(T, starts, max_sigma):
    """Run diffusion for multiple starting nodes simultaneously using sparse
    matrix–vector products.

    Returns an array of shape (len(starts), max_sigma+1) with P(sigma) for
    each starting node.
    """
    N = T.shape[0]
    n_walks = len(starts)
    return_probs = np.zeros((n_walks, max_sigma + 1))
    return_probs[:, 0] = 1.0

    # prob[:, w] is the probability vector for walk w
    prob = np.zeros((N, n_walks))
    for w, s in enumerate(starts):
        prob[s, w] = 1.0

    for sigma in range(1, max_sigma + 1):
        prob = T @ prob                       # sparse mat × dense mat
        for w, s in enumerate(starts):
            return_probs[w, sigma] = prob[s, w]

    return return_probs


# ---------------------------------------------------------------------------
# Worker for parallel configurations
# ---------------------------------------------------------------------------

def _worker(args_tuple):
    """Run one independent configuration: build spacetime, thermalize,
    build dual graph, run diffusion walks.  Returns list of return-prob arrays.
    """
    (cfg_id, n_simplices, n_therm, sweeps_between,
     n_walks, max_sigma) = args_tuple

    sig = caset.Signature(4, caset.Lorentzian)
    metric = caset.Metric(True, sig)
    st = caset.Spacetime(metric, caset.CDT, 1.0, 1.0, caset.PREFERRED,
                         caset.Toroid())
    st.build(n_simplices)
    target = st.getSimplexCount()
    cdt = caset.CDTSimulation(st, 2.2, 0.5, 0.6, 0.02, target)
    cdt.tune()

    for _ in range(n_therm):
        cdt.sweep()

    # Decorrelate
    for _ in range(sweeps_between):
        cdt.sweep()

    t0 = time.time()
    T, N = build_transition_matrix(st)
    if T is None or N == 0:
        return cfg_id, [], 0, 0.0, 0.0

    starts = np.random.choice(N, size=min(n_walks, N), replace=False)
    rp = diffuse_sparse(T, starts, max_sigma)

    deg = np.array(T.sum(axis=0)).ravel()
    avg_nbr = deg.mean()
    elapsed = time.time() - t0
    return cfg_id, rp.tolist(), N, avg_nbr, elapsed


# ---------------------------------------------------------------------------
# Spectral dimension extraction
# ---------------------------------------------------------------------------

def compute_spectral_dimension(return_prob):
    """Compute D_S(sigma) = -2 d(log P) / d(log sigma).

    Uses centered finite differences on log-log data.
    Returns (sigma_values, D_S_values) excluding endpoints and zeros.
    """
    sigma = np.arange(len(return_prob))
    valid = (sigma > 1) & (return_prob > 0)
    s = sigma[valid].astype(float)
    p = return_prob[valid]

    log_s = np.log(s)
    log_p = np.log(p)

    if len(log_s) < 2:
        return s, np.zeros(len(s))

    ds = np.zeros(len(log_s))
    ds[1:-1] = (log_p[2:] - log_p[:-2]) / (log_s[2:] - log_s[:-2])
    ds[0] = (log_p[1] - log_p[0]) / (log_s[1] - log_s[0])
    ds[-1] = (log_p[-1] - log_p[-2]) / (log_s[-1] - log_s[-2])

    D_S = -2.0 * ds
    return s, D_S


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    parser.add_argument("--workers", type=int,
                        default=min(os.cpu_count() or 1, 8),
                        help="Parallel worker processes (default: min(cpus, 8))")
    parser.add_argument("--save", type=str, default=None,
                        help="Save figure to this path instead of showing")
    args = parser.parse_args()

    n_workers = max(1, args.workers)

    print("=" * 64)
    print("  Spectral Dimension Measurement via Discrete Diffusion")
    print("  Reproduces Figs 9-10, Ambjorn, Jurkiewicz, Loll (2005)")
    print("  Parameters: k0=2.2, Delta=0.6")
    print(f"  Configs: {args.n_configs}, walks/config: {args.n_walks}, "
          f"max sigma: {args.max_sigma}")
    print(f"  Workers: {n_workers}")
    print("=" * 64)

    t_total = time.time()

    # Each worker gets its own independent spacetime + simulation
    tasks = [
        (cfg, args.n_simplices, args.n_therm, args.sweeps_between,
         args.n_walks, args.max_sigma)
        for cfg in range(args.n_configs)
    ]

    all_return_probs = []

    if n_workers == 1:
        # Sequential fallback
        for task in tqdm(tasks, desc="Configurations", unit="cfg"):
            cfg_id, rps, N, avg_nbr, elapsed = _worker(task)
            if rps:
                all_return_probs.extend(rps)
                print(f"  Config {cfg_id+1}: N4~{N:,}, "
                      f"avg neighbors={avg_nbr:.1f}, {elapsed:.1f}s")
            else:
                print(f"  Config {cfg_id+1}: no simplices, skipping")
    else:
        with multiprocessing.Pool(n_workers) as pool:
            results = pool.imap_unordered(_worker, tasks)
            done = 0
            for cfg_id, rps, N, avg_nbr, elapsed in results:
                done += 1
                if rps:
                    all_return_probs.extend(rps)
                    print(f"  Config {cfg_id+1}: N4~{N:,}, "
                          f"avg neighbors={avg_nbr:.1f}, "
                          f"{elapsed:.1f}s  [{done}/{args.n_configs}]")
                else:
                    print(f"  Config {cfg_id+1}: no simplices  "
                          f"[{done}/{args.n_configs}]")

    if not all_return_probs:
        print("No data collected.")
        return

    all_return_probs = [np.array(rp) for rp in all_return_probs]
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

    ax1.plot(sigma_vals, D_S_vals, "b-", linewidth=1.5, label="Measured")
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

    if len(sigma_vals) > 0:
        n_tail = max(1, len(D_S_vals) // 5)
        D_S_large = np.mean(D_S_vals[-n_tail:])
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
