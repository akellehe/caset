"""Temporally-connected entangled spacetime: a dual lattice whose temporal
edges are mutual information, not causal-set forward propagation.

Setup (van Raamsdonk dual lattice)
----------------------------------

We work on the same (bond, snapshot) label set as
``examples/quantum/run_dual_spectral_dimension.py``: each vertex is the
bipartition cut of the Schwinger chain at bond ``n`` and snapshot ``t``,
and edge weights are mutual-information values on those bipartitions.

In that script the temporal sector was a causet ``causal step``: bond
``n`` at time ``t`` is connected to *only* the same bond ``n`` at time
``t+1``, weighted by a global ``typical-hop`` proxy (the median spatial
MI in the source snapshot). That is the ``information transferred from
one causal event to the next`` reading of forward propagation. Spatial
connectivity, by contrast, is MI-based and degree-unbounded.

What changes here
-----------------

We replace causet-style forward propagation with **mutual information
across forward snapshot pairs**: bond ``n`` at time ``t`` is connected
to *every* bond ``m`` at time ``t'`` whose cross-snapshot MI exceeds
``--epsilon-i``, with ``t'`` ranging over *all* later snapshots by
default (set ``--max-temporal-stride`` to a positive value to cap).
The temporal sector is no longer a single causal step per node —
every pair of bonds that share information across any pair of
snapshots gets an edge, and the ``epsilon-I`` cutoff alone decides
what counts as ``shares information``. Vertex degrees are driven by
the entanglement structure rather than by the lattice topology, and
they grow with the snapshot count ``K``.

Cross-snapshot bond-bond MI
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A clean ``MI(bond n at t : bond m at t')`` would come from the Choi
state of the TDVP propagator on bipartition variables. We do not have
that observable on bonds; the tessera Choi machinery operates on
sites. We use the simplest computable surrogate consistent with the
``shares information across the two endpoints`` reading:

.. math::

    \\widetilde I(n, t ; m, t') = \\tfrac12 \\bigl(
        I_t(n, m) + I_{t'}(n, m)
    \\bigr),

where :math:`I_t` is the per-snapshot tripartite-information bond-MI
matrix (``TDVPSnapshot.bondMutualInformation``). The surrogate
symmetrises in ``(t, t')`` and reduces to the spatial weight when
``t = t'``. It does **not** capture pure-time correlations that
develop between snapshots through the unitary evolution — it captures
the part of the cross-time MI that is consistent with the bond MI at
both endpoints. This is documented openly so the comparison with the
causet baseline is interpretable.

Spectral dimension
------------------

We feed the resulting weighted graph into the same Krylov-Lanczos heat
kernel + Savitzky-Golay spectral-dimension pipeline as the existing
holography scripts, and fit ``D_S(sigma) = D_inf - C / (B + sigma)``
(Ambjorn-Loll). At ``N`` around 50 the causet baseline lands near the
expected ``D_S ~= 2`` lattice plateau; the falsifiable question this
script asks is whether the MI-temporal graph preserves that signal or
collapses it (e.g. by smearing temporal locality into a small-world
limit too aggressively).

Threading
---------

This is a long-running TDVP workload. On a shared box set the thread
caps at launch::

    OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 \\
    MKL_NUM_THREADS=10 BLIS_NUM_THREADS=10 \\
        python examples/quantum/temporally_connected_entangled_spacetime.py \\
            --N 50 --T 1.0 --dt 0.25 \\
            --out-json /tmp/temporal-entangled/result.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

from tessera.quantum import SchwingerQuench, TDVPConfig
from tessera.quantum.holography import (
    AmbjornLollFit,
    EmergentGraph,
)


def _bond_matrices(snapshots):
    """Stack per-snapshot bond-MI matrices into a (K, B, B) array."""
    K = len(snapshots)
    if K == 0:
        raise ValueError("at least one snapshot required")
    first = snapshots[0].bondMutualInformation
    if not first:
        raise ValueError(
            "snapshots carry no bondMutualInformation. Set "
            "TDVPConfig.recordBondMutualInformation = True before evolve().")
    B = int(round(math.sqrt(len(first))))
    if B * B != len(first):
        raise ValueError(
            f"bondMutualInformation length {len(first)} is not a perfect "
            f"square; expected (N-1)^2.")
    out = np.empty((K, B, B), dtype=np.float64)
    for t, snap in enumerate(snapshots):
        out[t] = np.array(snap.bondMutualInformation,
                           dtype=np.float64).reshape(B, B)
        np.fill_diagonal(out[t], 0.0)
    return out


def build_temporally_connected_graph(snapshots, epsilon_i, max_stride):
    """Return (n_vertices, edges, stats, mi_values) for the temporally-
    connected (bond, snapshot) graph.

    Spatial edges: for each snapshot ``t`` and bond pair ``(n, m)``, an
    edge ``((n, t), (m, t))`` with weight ``bondMI[t][n, m]`` whenever
    that exceeds ``epsilon_i``.

    Temporal edges: for each snapshot pair ``(t, t')`` with
    ``1 <= t' - t <= max_stride`` and each bond pair ``(n, m)``, an
    edge ``((n, t), (m, t'))`` with weight
    ``0.5 * (bondMI[t][n, m] + bondMI[t'][n, m])`` whenever that
    exceeds ``epsilon_i``. ``max_stride <= 0`` means ``K - 1`` (all
    forward pairs).

    ``mi_values`` is a ``(spatial_all, temporal_all)`` tuple of flat
    numpy arrays carrying *all* candidate weights *before* the
    ``epsilon_i`` cutoff — for downstream histogram analysis.
    """
    bond_mi = _bond_matrices(snapshots)
    K, B, _ = bond_mi.shape
    n_vertices = K * B

    if max_stride <= 0:
        max_stride = K - 1

    def vertex(bond, snap):
        return snap * B + bond

    edges = []

    spatial_all = []
    spatial_count = 0
    for t in range(K):
        W = bond_mi[t]
        ii, jj = np.triu_indices(B, k=1)
        wij = W[ii, jj]
        spatial_all.append(wij)
        keep = wij > epsilon_i
        for n, m, w in zip(ii[keep], jj[keep], wij[keep]):
            edges.append((vertex(int(n), t), vertex(int(m), t), float(w)))
            spatial_count += 1

    temporal_all = []
    temporal_count = 0
    for t in range(K - 1):
        s_max = min(max_stride, K - 1 - t)
        for s in range(1, s_max + 1):
            avg = 0.5 * (bond_mi[t] + bond_mi[t + s])
            # All bond pairs — including (n, n) — are eligible across
            # snapshots; n=m on the temporal axis is the analogue of
            # the causet baseline.
            ii, jj = np.indices(avg.shape)
            wij = avg.ravel()
            temporal_all.append(wij)
            mask = (avg > epsilon_i)
            # Avoid degenerate self-loops within the same vertex; vertices
            # are (bond, snap) so (n, t) and (n, t+s) are distinct.
            for n, m in zip(ii[mask], jj[mask]):
                w = float(avg[n, m])
                edges.append((vertex(int(n), t),
                               vertex(int(m), t + s),
                               w))
                temporal_count += 1

    stats = {
        "n_bonds":         B,
        "n_snapshots":     K,
        "max_stride":      max_stride,
        "n_edges_spatial":  spatial_count,
        "n_edges_temporal": temporal_count,
    }
    mi_values = (
        np.concatenate(spatial_all)  if spatial_all  else np.empty(0),
        np.concatenate(temporal_all) if temporal_all else np.empty(0),
    )
    return n_vertices, edges, stats, mi_values


def _mi_histogram(values, n_bins=60):
    """Log-binned histogram summary suitable for JSON storage."""
    values = np.asarray(values, dtype=np.float64)
    positive = values[values > 0]
    if positive.size == 0:
        return {
            "n_total":    int(values.size),
            "n_positive": 0,
            "n_zero":     int(values.size),
            "min":        0.0,
            "max":        0.0,
            "median":     0.0,
            "edges":      [],
            "counts":     [],
        }
    lo = float(positive.min())
    hi = float(positive.max())
    if hi <= lo:
        hi = lo * 10.0
    edges = np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)
    counts, _ = np.histogram(positive, bins=edges)
    return {
        "n_total":    int(values.size),
        "n_positive": int(positive.size),
        "n_zero":     int(values.size - positive.size),
        "min":        lo,
        "max":        hi,
        "median":     float(np.median(positive)),
        "edges":      edges.tolist(),
        "counts":     counts.astype(int).tolist(),
    }


def _degree_summary(n_vertices, edges):
    """Mean / max / median degree across vertices."""
    deg = np.zeros(n_vertices, dtype=np.int64)
    for u, v, _ in edges:
        deg[u] += 1
        deg[v] += 1
    if deg.size == 0:
        return {"mean": 0.0, "max": 0, "median": 0.0, "p95": 0.0}
    return {
        "mean":   float(deg.mean()),
        "max":    int(deg.max()),
        "median": float(np.median(deg)),
        "p95":    float(np.percentile(deg, 95)),
    }


def _diameter_summary(n_vertices, edges):
    """Exact unweighted (hop) diameter and avg shortest-path length on
    the largest connected component. Uses scipy.sparse.csgraph for the
    BFS sweep (compiled C), so cost stays under a few seconds even at
    |V| ~ 1000 and |E| ~ 500k.
    """
    if n_vertices == 0 or not edges:
        return {"hop_diameter": 0, "hop_avg": 0.0,
                "lcc_size": 0, "n_components": 0}

    import scipy.sparse as sp
    import scipy.sparse.csgraph as csg

    rows = np.fromiter((u for u, _, _ in edges), dtype=np.int64,
                        count=len(edges))
    cols = np.fromiter((v for _, v, _ in edges), dtype=np.int64,
                        count=len(edges))
    data = np.ones(len(edges), dtype=np.int8)
    A = sp.coo_matrix(
        (np.concatenate([data, data]),
         (np.concatenate([rows, cols]),
          np.concatenate([cols, rows]))),
        shape=(n_vertices, n_vertices)).tocsr()

    n_components, comp_labels = csg.connected_components(
        A, directed=False, return_labels=True)
    unique, counts = np.unique(comp_labels, return_counts=True)
    lcc_id = int(unique[int(np.argmax(counts))])
    lcc_mask = (comp_labels == lcc_id)
    lcc_vertices = np.flatnonzero(lcc_mask)
    lcc_size = int(lcc_vertices.size)

    sub = A[lcc_mask][:, lcc_mask]
    dist = csg.shortest_path(sub, method="auto", unweighted=True,
                              directed=False)
    finite = dist[np.isfinite(dist) & (dist > 0)]
    diameter   = int(finite.max()) if finite.size else 0
    hop_avg    = float(finite.mean()) if finite.size else 0.0
    return {
        "hop_diameter":   diameter,
        "hop_avg":        hop_avg,
        "lcc_size":       lcc_size,
        "n_components":   int(n_components),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--N",  type=int, default=50,
                    help="Staggered sites. Matches the regime where the "
                         "causet-dual-lattice script lands near D_S=2.")
    p.add_argument("--a",  type=float, default=1.0)
    p.add_argument("--g",  type=float, default=1.0)
    p.add_argument("--m-over-g", type=float, default=0.5)
    p.add_argument("--L0", type=float, default=0.0)
    p.add_argument("--i0", type=int,   default=3)
    p.add_argument("--d",  type=int,   default=3)
    p.add_argument("--dt", type=float, default=0.25)
    p.add_argument("--T",  type=float, default=1.0)
    p.add_argument("--max-bond-dim",      type=int, default=80)
    p.add_argument("--dmrg-max-bond-dim", type=int, default=64)
    p.add_argument("--dmrg-n-sweeps",     type=int, default=12)
    p.add_argument("--sigma-min",   type=float, default=1e-2)
    p.add_argument("--sigma-max",   type=float, default=1e3)
    p.add_argument("--sigma-count", type=int,   default=48)
    p.add_argument("--epsilon-i",   type=float, default=1e-8)
    p.add_argument("--krylov-dim",  type=int,   default=30)
    p.add_argument("--max-temporal-stride", type=int, default=0,
                    help="Max |t' - t| on the temporal axis. 0 (default) = "
                         "unlimited — every forward snapshot pair "
                         "contributes temporal edges and the epsilon-I "
                         "cutoff alone prunes them. Set a positive value "
                         "to cap the stride explicitly.")
    p.add_argument("--out-json",
                    default="/tmp/temporal-entangled/result.json")
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

    print(f"[setup] N={args.N}, m/g={args.m_over_g}, T={args.T}, "
          f"dt={args.dt}, max_stride={args.max_temporal_stride}",
          flush=True)
    t0 = time.perf_counter()
    res = SchwingerQuench(cfg).evolve()
    print(f"[tdvp]  {len(res.snapshots)} snapshots in "
          f"{time.perf_counter() - t0:.1f} s", flush=True)

    n_vertices, edges, gstats, mi_values = build_temporally_connected_graph(
        res.snapshots, args.epsilon_i, args.max_temporal_stride)
    print(f"[graph] |V|={n_vertices}, |E|={len(edges)}  "
          f"(spatial={gstats['n_edges_spatial']}, "
          f"temporal={gstats['n_edges_temporal']})", flush=True)
    if len(edges) == 0:
        print("[abort] no edges above epsilon_I — check input")
        sys.exit(1)

    deg = _degree_summary(n_vertices, edges)
    print(f"[degree] mean={deg['mean']:.2f}  median={deg['median']:.2f}  "
          f"p95={deg['p95']:.1f}  max={deg['max']}", flush=True)

    t_diam = time.perf_counter()
    diam = _diameter_summary(n_vertices, edges)
    print(f"[diam] hop_diameter={diam['hop_diameter']}  "
          f"avg_path={diam['hop_avg']:.2f}  "
          f"LCC={diam['lcc_size']}/{n_vertices}  "
          f"comps={diam['n_components']}  "
          f"({time.perf_counter() - t_diam:.1f}s)", flush=True)

    spatial_hist  = _mi_histogram(mi_values[0])
    temporal_hist = _mi_histogram(mi_values[1])
    print(f"[mi] spatial:  n={spatial_hist['n_positive']}/{spatial_hist['n_total']} "
          f"min={spatial_hist['min']:.2e} max={spatial_hist['max']:.2e} "
          f"median={spatial_hist['median']:.2e}", flush=True)
    print(f"[mi] temporal: n={temporal_hist['n_positive']}/{temporal_hist['n_total']} "
          f"min={temporal_hist['min']:.2e} max={temporal_hist['max']:.2e} "
          f"median={temporal_hist['median']:.2e}", flush=True)

    g = EmergentGraph.fromWeightedEdges(n_vertices, edges)

    sigmas_log = np.linspace(math.log(args.sigma_min),
                              math.log(args.sigma_max),
                              args.sigma_count).tolist()
    sigmas = [math.exp(x) for x in sigmas_log]

    t1 = time.perf_counter()
    P = g.returnProbability(sigmas, args.krylov_dim)
    print(f"[heat]  {args.sigma_count} sigmas in "
          f"{time.perf_counter() - t1:.1f} s", flush=True)

    dS  = EmergentGraph.spectralDimension(sigmas, P)
    dSs = EmergentGraph.spectralDimensionSmoothed(sigmas, P, 5, 2)
    fit = AmbjornLollFit.fit(sigmas, dSs)

    peak_dS = max((d for d in dSs if math.isfinite(d)), default=float("nan"))
    peak_idx = next((i for i, d in enumerate(dSs)
                      if math.isfinite(d) and d == peak_dS), None)
    sigma_peak = sigmas[peak_idx] if peak_idx is not None else None

    sigma_peak_str = f"{sigma_peak:.4f}" if sigma_peak is not None else "NA"
    print(f"[result] peak D_S = {peak_dS:.4f} at sigma ~= {sigma_peak_str}",
          flush=True)
    print(f"[result] D_inf (Ambjorn-Loll fit) = {fit.dInfinity:.4f}",
          flush=True)

    out = {
        "config": {
            "N": args.N, "a": args.a, "g": args.g,
            "m_over_g": args.m_over_g, "L0": args.L0,
            "i0": args.i0, "d": args.d,
            "dt": args.dt, "T": args.T,
            "max_bond_dim":     args.max_bond_dim,
            "sigma_count":       args.sigma_count,
            "epsilon_I":         args.epsilon_i,
            "krylov_dim":        args.krylov_dim,
            "max_temporal_stride": args.max_temporal_stride,
        },
        "graph": {
            "n_vertices":     n_vertices,
            "n_edges":        len(edges),
            "n_edges_spatial":  gstats["n_edges_spatial"],
            "n_edges_temporal": gstats["n_edges_temporal"],
            "n_snapshots":    gstats["n_snapshots"],
            "n_bonds":        gstats["n_bonds"],
            "max_stride":     gstats["max_stride"],
            "degree":         deg,
            "diameter":       diam,
        },
        "sigmas":      sigmas,
        "P":           P,
        "dS_raw":      dS,
        "dS_smoothed": dSs,
        "ambjorn_loll": {
            "D_infinity":  fit.dInfinity,
            "C":           fit.C,
            "B":           fit.B,
            "chi_squared": fit.chiSquared,
        },
        "peak_dS":    peak_dS,
        "sigma_peak": sigma_peak,
        "mi_distributions": {
            "spatial":  spatial_hist,
            "temporal": temporal_hist,
        },
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[wrote] {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
