"""Interaction-branching simplex: does the t+2dt closure rotate the cell
out of the plane?

The construction
----------------

1. **Initial layer.** Poisson-distribute N points in a 2D patch and Delaunay
   triangulate them. The points are the N staggered sites of a Schwinger
   chain; the Delaunay edges are the spatial adjacencies at t = 0. This is
   the randomized 2-simplicial complex -- and the Delaunay triangulation is
   the Voronoi dual.

2. **The cell.** Pick a Delaunay triangle. Its three vertices are sites
   ``a``, ``b``, ``m``. The branching cell ``{A, B, A', AB, B'}`` sits on
   the ``a-b`` Delaunay edge:

     A  = (site a, t=0)      A' = (site a, t=dt)
     B  = (site b, t=0)      B' = (site b, t=dt)
                             AB = (site m, t=dt)   -- the triangle apex,
                                                      the interaction product

   Two vertices at t=0, three at t=dt: a CDT ``(2,3)`` 4-simplex.

3. **Lengths from mutual information.** Every edge length is the van
   Raamsdonk distance ``d = -log I``. The ten edges split by origin:

     * ``A-B``                              -- the Delaunay edge (t=0 spatial MI)
     * 6 edges from e1 (the first TDVP step):
       ``A-A' A-AB B-AB B-B'``  (temporal, Choi MI over dt)
       ``A'-AB B'-AB``          (spatial, snapshot-1 MI)
     * 3 closure edges supplied only by the t+2dt step:
       ``A'-B'``  (spatial)  and  ``A-B' B-A'``  (temporal)

   Spatial MI comes from ``TDVPSnapshot.mutualInformation``; temporal MI
   from the Choi state of the propagator (``ChoiPropagator``).

The test
--------

  * **open** cell -- the closure edges are read from the t=dt data only
    (snapshot 1, Choi over dt). This is the cell before the t+2dt event.
  * **closed** cell -- the closure edges are read from the t+2dt data
    (snapshot 2, Choi over 2*dt). This is the cell the next interaction
    has closed.

Hand the ten edge lengths to the tessera simplicial machinery as a
``(2,3)`` Lorentzian 4-simplex and read off:

  * ``det G`` -- Gram determinant, proportional to the squared 4-volume.
    ``det G = 0`` is the degenerate (coplanar) 4-simplex; ``det G > 0`` is
    a genuine 4-volume -- the cell rotated out of the plane; ``det G < 0``
    means the MI lengths admit no Euclidean embedding at all.
  * ``area`` of the spatial triangle ``{A', AB, B'}`` -- zero means the
    t=dt slice is a degenerate 1D path; nonzero means it has become 2D.

Sweeping every Delaunay triangle of the Poisson layer, across Schwinger
``m/g`` values, gives the statistics: does extending the evolution through
the t+2dt closure turn degenerate cells into ones with 4-volume?

Run::

    OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 \\
    MKL_NUM_THREADS=10 BLIS_NUM_THREADS=10 \\
        python examples/quantum/interaction_branching_simplex.py \\
            --N 12 --out-json /tmp/interaction-branching/result.json
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
from scipy.spatial import Delaunay

from tessera import Lorentzian, Metric, NONE, REGGE, Signature, Spacetime
from tessera.quantum import MutualInformation, SchwingerQuench, TDVPConfig
from tessera.quantum.holography import (
    ChoiPropagator,
    ChoiTDVPSettings,
    SchwingerParams,
)
import cmath

LOG2 = math.log(2.0)
I_MAX = 2.0 * LOG2   # algebraic maximum of the MI between two single sites

# The ten edges of the {A,B,A',AB,B'} 4-simplex, with CDT disposition.
# Same time slice -> spacelike (squared length > 0); across slices ->
# timelike (squared length < 0).
EDGES = [
    ("A",  "B",  "spacelike"),   # the Delaunay edge of the initial layer
    ("A",  "A'", "timelike"),    # -- 6 edges from e1 --
    ("A",  "AB", "timelike"),
    ("B",  "AB", "timelike"),
    ("B",  "B'", "timelike"),
    ("A'", "AB", "spacelike"),
    ("B'", "AB", "spacelike"),
    ("A'", "B'", "spacelike"),   # -- 3 closure edges (need t+2dt) --
    ("A",  "B'", "timelike"),
    ("B",  "A'", "timelike"),
]
CLOSURE_EDGES = {("A'", "B'"), ("A", "B'"), ("B", "A'")}


# --- the Poisson-Delaunay initial layer -------------------------------------

def poisson_delaunay_layer(n_sites: int, rng: np.random.Generator):
    """Poisson-distribute n_sites points in the unit square, Delaunay
    triangulate, and return (points, triangles).

    The point count is fixed at n_sites (a Poisson process conditioned on
    its count is n_sites i.i.d. uniform points); positions are uniform.
    triangles is the (n_tri, 3) array of site-index triples.
    """
    points = rng.uniform(0.0, 1.0, size=(n_sites, 2))
    tri = Delaunay(points)
    return points, tri.simplices


# --- the Schwinger quantum data ---------------------------------------------

def schwinger_mi_data(n_sites: int, m_over_g: float, dt: float,
                      max_bond_dim: int = 64) -> dict:
    """Run the Schwinger ground state + two TDVP steps, and collect every
    mutual-information matrix the cell construction needs.

    Returns a dict with:
      spatial[k]  -- N x N site-site MI at snapshot k (k = 0, 1, 2)
      temporal_1  -- N x N Choi temporal MI over a single step dt
      temporal_2  -- N x N Choi temporal MI over two steps 2*dt
    """
    cfg = TDVPConfig()
    cfg.N = n_sites
    cfg.a = 1.0
    cfg.g = 1.0
    cfg.m = m_over_g * cfg.g
    cfg.L0 = 0.0
    cfg.dmrgMaxBondDim = min(64, max_bond_dim)
    cfg.dmrgNSweeps = 10
    cfg.dmrgKrylovDim = 4
    cfg.dmrgCutoff = 1e-12
    # q-qbar quench: i0 on the Up sublattice (odd, 1-based), d odd.
    cfg.i0 = 3
    cfg.d = 3
    cfg.quenchEnforceParity = True
    cfg.dt = dt
    cfg.T = 2.0 * dt
    cfg.snapshotEvery = 1
    cfg.maxBondDim = max_bond_dim
    cfg.cutoff = 1e-10
    cfg.krylovDim = 24
    cfg.quiet = True
    cfg.conserveQns = True
    cfg.recordMutualInformation = True

    result = SchwingerQuench(cfg).evolve()
    snaps = result.snapshots
    if len(snaps) < 3:
        raise RuntimeError(f"expected 3 snapshots, got {len(snaps)}")
    spatial = [np.array(s.mutualInformation, dtype=np.float64)
               .reshape(n_sites, n_sites) for s in snaps[:3]]

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
    temporal_1 = np.array(
        ChoiPropagator.temporalMutualInformation(params, dt, choi),
        dtype=np.float64)
    temporal_2 = np.array(
        ChoiPropagator.temporalMutualInformation(params, 2.0 * dt, choi),
        dtype=np.float64)

    return {
        "spatial": spatial,
        "temporal_1": temporal_1,
        "temporal_2": temporal_2,
        "energies": [float(s.energy) for s in snaps[:3]],
    }


# --- cell edge lengths from the quantum data --------------------------------

def edge_length(info: float, epsilon: float) -> float:
    """van Raamsdonk distance, normalised so d >= 0: d = -log(I / I_max),
    floored when I drops below epsilon."""
    return MutualInformation.edgeLength(max(info, 0.0) / I_MAX, epsilon)


def cell_mutual_information(data: dict, a: int, b: int, m: int,
                            regime: str) -> dict:
    """The ten pairwise mutual informations of the cell on sites (a, b, m).

    regime is "open" (closure edges from the t=dt data) or "closed"
    (closure edges from the t+2dt data).
    """
    sp = data["spatial"]
    t1 = data["temporal_1"]
    t2 = data["temporal_2"]

    mi = {
        ("A",  "B"):  sp[0][a, b],          # Delaunay edge, t=0 spatial
        ("A",  "A'"): t1[a, a],             # -- e1 --
        ("A",  "AB"): t1[a, m],
        ("B",  "AB"): t1[b, m],
        ("B",  "B'"): t1[b, b],
        ("A'", "AB"): sp[1][a, m],
        ("B'", "AB"): sp[1][b, m],
    }
    if regime == "open":
        mi[("A'", "B'")] = sp[1][a, b]      # closure edges, pre-closure data
        mi[("A",  "B'")] = t1[a, b]
        mi[("B",  "A'")] = t1[b, a]
    elif regime == "closed":
        mi[("A'", "B'")] = sp[2][a, b]      # closure edges, t+2dt data
        mi[("A",  "B'")] = t2[a, b]
        mi[("B",  "A'")] = t2[b, a]
    else:
        raise ValueError(f"regime must be open|closed, got {regime!r}")
    return mi


# --- simplicial assembly ----------------------------------------------------

def assemble_cell(mi: dict, epsilon: float):
    """Build the (2,3) Lorentzian 4-simplex from the MI edge lengths.

    Returns (det_gram, spatial_area, max_edge_length, lengths).
    """
    sig = Signature(4, Lorentzian)
    metric = Metric(True, sig)                      # coordinate-free
    st = Spacetime(metric, REGGE, 1.0, 1.0, NONE, None)

    times = {"A": [0.0], "B": [0.0], "A'": [1.0], "AB": [1.0], "B'": [1.0]}
    vert = {name: st.createVertex(i, c)
            for i, (name, c) in enumerate(times.items())}

    lengths = {}
    for u, v, kind in EDGES:
        d = edge_length(mi[(u, v)], epsilon)
        lengths[f"{u}-{v}"] = d
        sq = d * d
        if kind == "timelike":
            sq = -sq
        st.createEdge(vert[u], vert[v], complex(sq))

    # An edge whose MI fell below the floor is infinitely long: the cell is
    # disconnected and has no Gram determinant.
    if not all(math.isfinite(d) for d in lengths.values()):
        return float("nan"), float("nan"), float("inf"), lengths

    simplex, _ = st.createSimplex([vert[n] for n in times])
    gram = np.array(simplex.gramMatrix(),
                    dtype=np.float64).reshape(4, 4)
    with np.errstate(invalid="ignore"):
        det_gram = float(np.linalg.det(gram))

    spatial, _ = st.createSimplex([vert["A'"], vert["AB"], vert["B'"]])
    spatial_area = float(spatial.area())

    return det_gram, spatial_area, max(lengths.values()), lengths


# --- the sweep --------------------------------------------------------------

def run_layer(data: dict, triangles: np.ndarray, epsilon: float) -> list:
    """Every Delaunay triangle, in both regimes. Each triangle gives three
    cells -- one per choice of which edge is the A-B Delaunay edge."""
    cells = []
    for tri in triangles:
        i, j, k = (int(x) for x in tri)
        for a, b, m in ((i, j, k), (j, k, i), (k, i, j)):
            row = {"a": a, "b": b, "m": m}
            for regime in ("open", "closed"):
                mi = cell_mutual_information(data, a, b, m, regime)
                det_g, area, max_len, lengths = assemble_cell(mi, epsilon)
                row[regime] = {
                    "det_gram": det_g,
                    "spatial_area": area,
                    "max_edge_length": max_len,
                }
            cells.append(row)
    return cells


def summarise(cells: list, regime: str) -> dict:
    """Aggregate the open or closed branch over all cells.

    det G > 0 : a genuine 4-volume -- the cell is rotated out of the plane.
    det G < 0 : geometrically frustrated -- the MI lengths admit no
                Euclidean embedding as a 4-simplex.
    det G ~ 0 : degenerate -- coplanar.
    disconnected : an edge fell below the MI floor (infinite length).
    """
    det = np.array([c[regime]["det_gram"] for c in cells])
    area = np.array([c[regime]["spatial_area"] for c in cells])
    n = len(cells)
    connected = np.isfinite(det)
    det_c = det[connected]
    tol = 1e-9
    return {
        "n_cells": n,
        "frac_disconnected": float((~connected).mean()) if n else 0.0,
        "det_gram": {
            "median": float(np.median(det_c)) if det_c.size else float("nan"),
            "frac_positive": float((det_c > tol).mean()) if det_c.size else 0.0,
            "frac_zero": float((np.abs(det_c) <= tol).mean())
                         if det_c.size else 0.0,
            "frac_negative": float((det_c < -tol).mean())
                             if det_c.size else 0.0,
        },
        "spatial_area": {
            "median": float(np.median(area[np.isfinite(area)]))
                      if np.isfinite(area).any() else float("nan"),
            "frac_nonzero": float((area > tol).mean()),
        },
    }


def transition_summary(cells: list) -> dict:
    """Paired open -> closed transitions for the same cell.

    The key number: of cells that are frustrated (det G < 0) when open,
    what fraction acquire a genuine 4-volume (det G > 0) once closed.
    """
    tol = 1e-9

    def klass(x):
        if not math.isfinite(x):
            return "disconnected"
        if x > tol:
            return "volume"
        if x < -tol:
            return "frustrated"
        return "degenerate"

    counts = {}
    for c in cells:
        key = (klass(c["open"]["det_gram"]), klass(c["closed"]["det_gram"]))
        counts[key] = counts.get(key, 0) + 1
    frustrated_open = [c for c in cells
                       if klass(c["open"]["det_gram"]) == "frustrated"]
    cured = [c for c in frustrated_open
             if klass(c["closed"]["det_gram"]) == "volume"]
    return {
        "transitions": {f"{a}->{b}": n for (a, b), n in sorted(counts.items())},
        "n_frustrated_open": len(frustrated_open),
        "frac_frustrated_cured_by_closure": (
            len(cured) / len(frustrated_open) if frustrated_open else 0.0),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--N", type=int, default=12,
                   help="Schwinger sites = Poisson points in the layer")
    p.add_argument("--m-over-g", type=float, nargs="+",
                   default=[0.125, 0.25, 0.5],
                   help="Schwinger mass ratios to sweep")
    p.add_argument("--dt", type=float, default=0.25,
                   help="TDVP step; the closure is the second step")
    p.add_argument("--max-bond-dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=8,
                   help="independent Poisson layouts per m/g")
    p.add_argument("--epsilon", type=float, default=1e-12,
                   help="mutual-information floor for the -log edge length")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json",
                   default="/tmp/interaction-branching/result.json")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"[setup] N={args.N} sites, m/g={args.m_over_g}, dt={args.dt}, "
          f"{args.layers} Poisson layers each, seed={args.seed}", flush=True)

    records = []
    for m_over_g in args.m_over_g:
        data = schwinger_mi_data(args.N, m_over_g, args.dt, args.max_bond_dim)
        print(f"[tdvp] m/g={m_over_g}: energies={[round(e,4) for e in data['energies']]}",
              flush=True)
        cells = []
        for _ in range(args.layers):
            _, triangles = poisson_delaunay_layer(args.N, rng)
            cells.extend(run_layer(data, triangles, args.epsilon))
        rec = {
            "m_over_g": m_over_g,
            "n_cells": len(cells),
            "open": summarise(cells, "open"),
            "closed": summarise(cells, "closed"),
            "transition": transition_summary(cells),
        }
        records.append(rec)
        for regime in ("open", "closed"):
            s = rec[regime]
            dg = s["det_gram"]
            ar = s["spatial_area"]
            print(f"   [{regime:>6}] det G: +vol={dg['frac_positive']:.3f}  "
                  f"degenerate={dg['frac_zero']:.3f}  "
                  f"frustrated={dg['frac_negative']:.3f}  "
                  f"disconnected={s['frac_disconnected']:.3f}  "
                  f"slice-2D={ar['frac_nonzero']:.3f}", flush=True)
        tr = rec["transition"]
        print(f"   [closure] frustrated-open cells cured to 4-volume: "
              f"{tr['frac_frustrated_cured_by_closure']:.3f} "
              f"(of {tr['n_frustrated_open']})", flush=True)

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"config": vars(args), "records": records}, f, indent=2)
    print(f"[wrote] {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
