"""Spectral dimension of the coned Poisson-Delaunay mutual-information complex.

Companion to
``examples/quantum/interaction_branching_simplex.py``. The single-cell
experiment there asks whether one branching cell has a 4-volume. This
script asks the graph-level question: what spectral dimension does the
*whole* coned Poisson-Delaunay complex carry, with every edge length
set by Schwinger mutual information?

Construction
------------

* **Initial layer.** Poisson-distribute ``N`` points, Delaunay
  triangulate. Points are Schwinger sites; Delaunay edges are the
  ``t = 0`` spatial adjacency. (Same layer as the single-cell script.)

* **Cone forward.** One layer per TDVP snapshot. The coning connects
  each site to its own forward copy and to the forward copies of its
  Delaunay neighbours -- the CDT time-extrusion of the Delaunay complex.

* **Weights from mutual information.** Spatial edges within snapshot
  ``k`` carry the site-site MI of that snapshot; temporal edges between
  consecutive snapshots carry the Choi-propagator temporal MI over
  ``dt``. Edges below the MI floor are dropped. The weighted graph
  Laplacian uses ``W = I`` directly, per
  emergent-spectral-dimension-schwinger-tdvp.md §3.4.

* **No thermalization** -- the Poisson-Delaunay layer already supplies
  the randomized connectivity.

Observable
----------

The heat-kernel spectral dimension ``D_S(sigma) = -2 d log P / d log
sigma`` of the resulting weighted graph, with the three-parameter
Ambjorn-Loll fit ``D_S = D_inf - C / (B + sigma)``. Reported per
Schwinger ``m/g``, averaged over independent Poisson layouts.

Run::

    OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 \\
    MKL_NUM_THREADS=10 BLIS_NUM_THREADS=10 \\
        python examples/quantum/poisson_delaunay_spectral_dimension.py \\
            --N 14 --T 2.0 --layers 12 \\
            --out-json /tmp/interaction-branching/spectral_dimension.json
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
from scipy.spatial import Delaunay

from tessera.quantum import SchwingerQuench, TDVPConfig
from tessera.quantum.holography import (
    AmbjornLollFit,
    ChoiPropagator,
    ChoiTDVPSettings,
    EmergentGraph,
    SchwingerParams,
)


# --- the Poisson-Delaunay initial layer -------------------------------------

def poisson_delaunay_layer(n_sites: int, rng: np.random.Generator):
    """Poisson points in the unit square + Delaunay triangulation.

    Returns (edges, neighbours): edges is the set of frozenset pairs of
    site indices joined by a Delaunay edge; neighbours[i] is the set of
    sites Delaunay-adjacent to site i.
    """
    points = rng.uniform(0.0, 1.0, size=(n_sites, 2))
    tri = Delaunay(points)
    edges = set()
    for simplex in tri.simplices:
        i, j, k = (int(x) for x in simplex)
        edges.update({frozenset((i, j)), frozenset((j, k)), frozenset((i, k))})
    neighbours = {i: set() for i in range(n_sites)}
    for e in edges:
        a, b = tuple(e)
        neighbours[a].add(b)
        neighbours[b].add(a)
    return edges, neighbours


# --- the Schwinger quantum data ---------------------------------------------

def schwinger_snapshots(n_sites: int, m_over_g: float, dt: float, total_t: float,
                        max_bond_dim: int = 64) -> dict:
    """Ground state + TDVP evolution to total_t; collect every per-snapshot
    site-site MI matrix and the one-step Choi temporal MI matrix.

    Returns:
      spatial    -- list of K (N x N) snapshot MI matrices
      temporal   -- (N x N) Choi temporal MI over a single step dt
      energies   -- per-snapshot energy
    """
    cfg = TDVPConfig()
    cfg.N = n_sites
    cfg.a = 1.0
    cfg.g = 1.0
    cfg.m = m_over_g * cfg.g
    cfg.L0 = 0.0
    cfg.dmrgMaxBondDim = 64
    cfg.dmrgNSweeps = 12
    cfg.dmrgKrylovDim = 4
    cfg.dmrgCutoff = 1e-12
    cfg.i0 = 3
    cfg.d = 3
    cfg.quenchEnforceParity = True
    cfg.dt = dt
    cfg.T = total_t
    cfg.snapshotEvery = 1
    cfg.maxBondDim = max_bond_dim
    cfg.cutoff = 1e-10
    cfg.krylovDim = 24
    cfg.quiet = True
    cfg.conserveQns = True
    cfg.recordMutualInformation = True

    result = SchwingerQuench(cfg).evolve()
    snaps = result.snapshots
    spatial = [np.array(s.mutualInformation, dtype=np.float64)
               .reshape(n_sites, n_sites) for s in snaps]

    params = SchwingerParams()
    params.N = n_sites
    params.a = cfg.a
    params.g = cfg.g
    params.m = cfg.m
    params.L0 = cfg.L0
    choi = ChoiTDVPSettings()
    choi.dt = dt
    choi.maxBondDim = max_bond_dim
    choi.cutoff = 1e-10
    choi.krylovDim = 24
    choi.quiet = True
    temporal = np.array(
        ChoiPropagator.temporalMutualInformation(params, dt, choi),
        dtype=np.float64)

    return {
        "spatial": spatial,
        "temporal": temporal,
        "energies": [float(s.energy) for s in snaps],
    }


# --- the coned graph --------------------------------------------------------

def build_coned_graph(data: dict, edges: set, neighbours: dict,
                       n_sites: int, epsilon: float):
    """Assemble the (site, snapshot) weighted graph.

    Spatial edges: Delaunay edges within each snapshot, weight = snapshot
    site-site MI. Temporal edges: each site to its own forward copy and
    to the forward copies of its Delaunay neighbours, weight = Choi
    temporal MI over dt. Edges with weight <= epsilon are dropped.

    Returns (n_vertices, weighted_edges, stats).
    """
    spatial = data["spatial"]
    temporal = data["temporal"]
    n_snap = len(spatial)
    n_vertices = n_sites * n_snap

    def vertex(site, snap):
        return snap * n_sites + site

    weighted = []
    n_spatial = n_temporal = 0

    for k in range(n_snap):
        w = spatial[k]
        for e in edges:
            i, j = tuple(e)
            weight = float(w[i, j])
            if weight > epsilon:
                weighted.append((vertex(i, k), vertex(j, k), weight))
                n_spatial += 1

    for k in range(n_snap - 1):
        for i in range(n_sites):
            forward = {i} | neighbours[i]
            for j in forward:
                weight = float(temporal[i, j])
                if weight > epsilon:
                    weighted.append((vertex(i, k), vertex(j, k + 1), weight))
                    n_temporal += 1

    stats = {
        "n_snapshots": n_snap,
        "n_sites": n_sites,
        "n_edges_spatial": n_spatial,
        "n_edges_temporal": n_temporal,
    }
    return n_vertices, weighted, stats


# --- the experiment ---------------------------------------------------------

def run_one(data: dict, n_sites: int, rng: np.random.Generator,
            sigmas: list, krylov_dim: int, epsilon: float) -> dict:
    """One Poisson layout: build the coned graph, take its spectral
    dimension."""
    edges, neighbours = poisson_delaunay_layer(n_sites, rng)
    n_vertices, weighted, stats = build_coned_graph(
        data, edges, neighbours, n_sites, epsilon)
    if not weighted:
        return None

    graph = EmergentGraph.fromWeightedEdges(n_vertices, weighted)
    p = graph.returnProbability(sigmas, krylov_dim)
    d_s = EmergentGraph.spectralDimension(sigmas, p)
    d_s_smooth = EmergentGraph.spectralDimensionSmoothed(sigmas, p, 5, 2)
    fit = AmbjornLollFit.fit(sigmas, d_s_smooth)

    finite = [d for d in d_s_smooth if math.isfinite(d)]
    peak = max(finite) if finite else float("nan")
    peak_idx = next((i for i, d in enumerate(d_s_smooth)
                     if math.isfinite(d) and d == peak), None)
    return {
        "n_vertices": n_vertices,
        "n_edges": len(weighted),
        "stats": stats,
        "peak_dS": peak,
        "sigma_peak": sigmas[peak_idx] if peak_idx is not None else None,
        "D_infinity": fit.dInfinity,
        "fit_chi_squared": fit.chiSquared,
        "dS_smoothed": list(d_s_smooth),
    }


def summarise(runs: list) -> dict:
    peaks = np.array([r["peak_dS"] for r in runs if math.isfinite(r["peak_dS"])])
    d_inf = np.array([r["D_infinity"] for r in runs
                      if math.isfinite(r["D_infinity"])])
    return {
        "n_layouts": len(runs),
        "peak_dS": {
            "mean": float(peaks.mean()) if peaks.size else float("nan"),
            "std": float(peaks.std()) if peaks.size else float("nan"),
            "min": float(peaks.min()) if peaks.size else float("nan"),
            "max": float(peaks.max()) if peaks.size else float("nan"),
        },
        "D_infinity": {
            "mean": float(d_inf.mean()) if d_inf.size else float("nan"),
            "std": float(d_inf.std()) if d_inf.size else float("nan"),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # Defaults match examples/quantum/temporally_connected_entangled_spacetime.py
    p.add_argument("--N", type=int, nargs="+", default=[10, 20, 30, 40],
                   help="Schwinger sites = Poisson points (sweep)")
    p.add_argument("--m-over-g", type=float, nargs="+",
                   default=[0.125, 0.25, 0.5])
    p.add_argument("--dt", type=float, default=0.25)
    p.add_argument("--T", type=float, default=1.0,
                   help="total TDVP time; snapshot count K = T/dt + 1")
    p.add_argument("--max-bond-dim", type=int, default=80)
    p.add_argument("--layers", type=int, default=12,
                   help="independent Poisson layouts per (N, m/g)")
    p.add_argument("--sigma-min", type=float, default=1e-2)
    p.add_argument("--sigma-max", type=float, default=1e3)
    p.add_argument("--sigma-count", type=int, default=48)
    p.add_argument("--krylov-dim", type=int, default=30)
    p.add_argument("--epsilon", type=float, default=1e-8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json",
                   default="/tmp/interaction-branching/spectral_dimension.json")
    args = p.parse_args()

    sigmas_log = np.linspace(math.log(args.sigma_min), math.log(args.sigma_max),
                             args.sigma_count)
    sigmas = [math.exp(x) for x in sigmas_log]

    rng = np.random.default_rng(args.seed)
    print(f"[setup] N={args.N}, m/g={args.m_over_g}, dt={args.dt}, T={args.T} "
          f"(K={int(round(args.T / args.dt)) + 1} snapshots), "
          f"{args.layers} Poisson layouts each", flush=True)

    records = []
    for n_sites in args.N:
        for m_over_g in args.m_over_g:
            data = schwinger_snapshots(n_sites, m_over_g, args.dt, args.T,
                                       args.max_bond_dim)
            runs = []
            for _ in range(args.layers):
                r = run_one(data, n_sites, rng, sigmas, args.krylov_dim,
                            args.epsilon)
                if r is not None:
                    runs.append(r)
            summary = summarise(runs)
            rec = {
                "N": n_sites,
                "m_over_g": m_over_g,
                "energies": data["energies"],
                "summary": summary,
                "runs": runs,
                "sigmas": sigmas,
            }
            records.append(rec)
            ps = summary["peak_dS"]
            di = summary["D_infinity"]
            ex = runs[0]
            print(f"[N={n_sites:>2} m/g={m_over_g}] "
                  f"|V|={ex['n_vertices']} |E|~{ex['n_edges']}  "
                  f"peak D_S = {ps['mean']:.3f} +/- {ps['std']:.3f}  "
                  f"(range {ps['min']:.2f}-{ps['max']:.2f})   "
                  f"D_inf = {di['mean']:.3f} +/- {di['std']:.3f}", flush=True)

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"config": vars(args), "records": records}, f, indent=2)
    print(f"[wrote] {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
