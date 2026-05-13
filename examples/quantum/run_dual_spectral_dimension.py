"""Dual-lattice (bond-cut) emergent spectral dimension on a Schwinger TDVP state.

Builds the van Raamsdonk-style graph whose vertices are (bond, snapshot)
labels — one per bipartition of the chain at each time step — with edge
weights set by tripartite information

  I(bond_n, bond_m) = S(A) + S(C) − S(B),

where A = [1..n], B = [n+1..m], C = [m+1..N] (1-based cut positions).
Temporal edges connect (bond_n, t) to (bond_n, t+1) with weight equal to
the median spatial-MI value in the source snapshot (so the temporal step
costs ~one "typical" spatial hop).

The downstream pipeline reuses ``EmergentGraph`` from
``tessera.quantum.holography``: return probability via Krylov-Lanczos heat
kernel, then Savitzky-Golay-smoothed spectral dimension and the
Ambjorn-Loll three-parameter fit.

Usage:

    python examples/quantum/run_dual_spectral_dimension.py \\
        --N 8 --m-over-g 0.5 --T 1.0 --dt 0.25 \\
        --out-json /tmp/dual-holography/mg_0.5.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from statistics import median

import numpy as np

from tessera.quantum import SchwingerQuench, TDVPConfig
from tessera.quantum.holography import (
    AmbjornLollFit,
    EmergentGraph,
)


def build_dual_graph(snapshots, epsilon_I):
    """Return (n_vertices, edges) for the (bond, snapshot) graph.

    Spatial edges (within snapshot t): tripartite-info entries above
    epsilon_I from the snapshot's ``bondMutualInformation`` matrix.
    Temporal edges (between t and t+1): unit-bond connections with weight
    set to the median of the spatial-MI values at snapshot t.
    """
    K = len(snapshots)
    if K == 0:
        raise ValueError("dual graph: at least one snapshot required")

    first_bm = snapshots[0].bondMutualInformation
    if not first_bm:
        raise ValueError(
            "dual graph: snapshots have no bondMutualInformation — set "
            "TDVPConfig.recordBondMutualInformation = True before evolve()")
    B = int(round(math.sqrt(len(first_bm))))
    if B * B != len(first_bm):
        raise ValueError(
            f"dual graph: bondMutualInformation length {len(first_bm)} is "
            f"not a perfect square; expected (N-1)² for some N")

    n_vertices = B * K
    edges = []

    def vertex(bond, snap):
        return snap * B + bond

    # Spatial edges per snapshot.
    for t, snap in enumerate(snapshots):
        bm = np.array(snap.bondMutualInformation,
                       dtype=np.float64).reshape(B, B)
        for n in range(B):
            for m in range(n + 1, B):
                w = float(bm[n, m])
                if w > epsilon_I:
                    edges.append((vertex(n, t), vertex(m, t), w))

    # Temporal edges (consecutive snapshots only). Weight = median
    # spatial MI in the source snapshot — a structural anchor that puts
    # temporal hops on the same scale as a typical bond-pair edge.
    for t in range(K - 1):
        bm = np.array(snapshots[t].bondMutualInformation,
                       dtype=np.float64).reshape(B, B)
        # Median of strictly-upper-triangle values above the cutoff.
        triu_vals = bm[np.triu_indices(B, k=1)]
        triu_above = triu_vals[triu_vals > epsilon_I]
        w_temporal = float(median(triu_above)) if triu_above.size > 0 else 0.0
        if w_temporal <= 0.0:
            continue
        for n in range(B):
            edges.append((vertex(n, t), vertex(n, t + 1), w_temporal))

    return n_vertices, edges


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=8)
    p.add_argument("--a", type=float, default=1.0)
    p.add_argument("--g", type=float, default=1.0)
    p.add_argument("--m-over-g", type=float, default=0.5)
    p.add_argument("--L0", type=float, default=0.0)
    p.add_argument("--i0", type=int, default=3)
    p.add_argument("--d",  type=int, default=3)
    p.add_argument("--dt", type=float, default=0.25)
    p.add_argument("--T",  type=float, default=1.0)
    p.add_argument("--max-bond-dim", type=int, default=80)
    p.add_argument("--dmrg-max-bond-dim", type=int, default=64)
    p.add_argument("--dmrg-n-sweeps",     type=int, default=12)
    p.add_argument("--sigma-min",   type=float, default=1e-2)
    p.add_argument("--sigma-max",   type=float, default=1e3)
    p.add_argument("--sigma-count", type=int,   default=48)
    p.add_argument("--epsilon-i",   type=float, default=1e-8)
    p.add_argument("--krylov-dim",  type=int,   default=30)
    p.add_argument("--out-json", default="/tmp/dual-holography/result.json")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    cfg = TDVPConfig()
    cfg.N = args.N
    cfg.a = args.a
    cfg.g = args.g
    cfg.m = args.m_over_g * args.g
    cfg.L0 = args.L0
    cfg.dmrgMaxBondDim = args.dmrg_max_bond_dim
    cfg.dmrgNSweeps    = args.dmrg_n_sweeps
    cfg.dmrgKrylovDim  = 4
    cfg.dmrgCutoff     = 1e-12
    cfg.i0 = args.i0
    cfg.d  = args.d
    cfg.quenchEnforceParity = True
    cfg.dt = args.dt
    cfg.T  = args.T
    cfg.snapshotEvery = 1
    cfg.maxBondDim = args.max_bond_dim
    cfg.cutoff     = 1e-10
    cfg.krylovDim  = 12
    cfg.quiet      = True
    cfg.conserveQns = True
    cfg.recordBondMutualInformation = True

    print(f"[setup] N={args.N}, m/g={args.m_over_g}, T={args.T}, dt={args.dt}",
          flush=True)
    t0 = time.perf_counter()
    res = SchwingerQuench(cfg).evolve()
    print(f"[tdvp] {len(res.snapshots)} snapshots in "
          f"{time.perf_counter() - t0:.1f} s", flush=True)

    n_vertices, edges = build_dual_graph(res.snapshots, args.epsilon_i)
    print(f"[graph] |V|={n_vertices}, |E|={len(edges)}", flush=True)
    if len(edges) == 0:
        print("[abort] no spatial edges above epsilon_I — check input")
        sys.exit(1)

    g = EmergentGraph.fromWeightedEdges(n_vertices, edges)

    sigmas_log = np.linspace(math.log(args.sigma_min),
                              math.log(args.sigma_max),
                              args.sigma_count).tolist()
    sigmas = [math.exp(x) for x in sigmas_log]

    t1 = time.perf_counter()
    P = g.returnProbability(sigmas, args.krylov_dim)
    print(f"[heat-kernel] {args.sigma_count} sigmas in "
          f"{time.perf_counter() - t1:.1f} s", flush=True)

    dS  = EmergentGraph.spectralDimension(sigmas, P)
    dSs = EmergentGraph.spectralDimensionSmoothed(sigmas, P, 5, 2)
    fit = AmbjornLollFit.fit(sigmas, dSs)

    peak_dS = max((d for d in dSs if math.isfinite(d)), default=float("nan"))
    peak_idx = next((i for i, d in enumerate(dSs)
                      if math.isfinite(d) and d == peak_dS), None)
    sigma_peak = sigmas[peak_idx] if peak_idx is not None else None

    sigma_peak_str = f"{sigma_peak:.4f}" if sigma_peak is not None else "NA"
    print(f"[result] peak D_S = {peak_dS:.4f} at sigma ≈ {sigma_peak_str}",
          flush=True)
    print(f"[result] D_infinity (Ambjorn-Loll fit) = {fit.dInfinity:.4f}",
          flush=True)

    out = {
        "config": {
            "N": args.N, "a": args.a, "g": args.g,
            "m_over_g": args.m_over_g, "L0": args.L0,
            "i0": args.i0, "d": args.d,
            "dt": args.dt, "T": args.T,
            "max_bond_dim": args.max_bond_dim,
            "sigma_count": args.sigma_count,
            "epsilon_I": args.epsilon_i,
            "krylov_dim": args.krylov_dim,
        },
        "graph": {
            "n_vertices": n_vertices,
            "n_edges": len(edges),
            "n_snapshots": len(res.snapshots),
            "n_bonds": args.N - 1,
        },
        "sigmas": sigmas,
        "P": P,
        "dS_raw": dS,
        "dS_smoothed": dSs,
        "ambjorn_loll": {
            "D_infinity": fit.dInfinity,
            "C": fit.C,
            "B": fit.B,
            "chi_squared": fit.chiSquared,
        },
        "peak_dS": peak_dS,
        "sigma_peak": sigma_peak,
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[wrote] {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
