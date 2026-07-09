"""Tessera-native economic cobordism build — Proton semantics, our action.

The pattern is the proton drive verbatim (grow → relax → verdict, dual
period residuals, surgery probes, restart floors), with two substitutions
sanctioned by the machinery itself:

* **Action.** The C++ objective (Regge + Γ·r_U) is not injectable, so —
  as `RealizabilityOracle` does internally — the loop is driven from
  Python: verdicts come from `MultiCobordism.r_state` (least-squares fit
  of the target against the emergent holes' cycle periods), and the
  action term is S_econ, the gauge energy of the carried harmonic
  representative (see gauge_dictionary.md §10).
* **Boundary treatment.** The entire year-t economy is the pinned
  initial geometry: both cylinder layers carry the year-t weights (the
  pinned economy carried forward); only surgery — never re-weighting of
  the pinned cells — changes what the geometry can carry.

The bulk is the **pure 2-complex cylinder** over the year-t register:
layer-0 fills, layer-1 fills, and two tube triangles per boundary edge
(every edge of the bulk lies in a triangle, so tessera's top-cell
readers — `Spacetime.fromCells`, `ChainComplex.fromSpacetime`,
`HodgeLaplacian` — represent it exactly).

The topological theorem that powers the verdict: in the plain cylinder
the two copies of any register loop are homologous, and a harmonic flow
has equal periods on homologous loops. Therefore

* the **anchor** (target = the year-t register itself) must be carried
  at residual ≈ 0 — the paper's T1/identity test;
* any transition that moves the register is **exactly unrealizable** on
  the intact cylinder — `r_state` floors at the register displacement;
* feasibility is restored only by **surgery** on tube cells (severing
  netting relations through time), and the cut set localizes the
  structural change. Surgery candidates are ranked by our action, not
  by Regge.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from tessera import cobordism, spacetime

from econ_register import Register


@dataclasses.dataclass
class TesseraBulk:
    st: "spacetime.Spacetime"
    reg: Register
    n_base: int
    tube_cells: list[tuple[int, int, int]]   # tube triangles, surgery candidates
    weight_of_edge: dict[frozenset, float]   # pinned year-t weights (both layers)


def _edge_weight_map(reg: Register, metric_mode: str = "conductance") -> dict:
    w = {}
    for k, (i, j) in enumerate(reg.edges):
        val = float(reg.gross[k])
        w[frozenset((i, j))] = (1.0 / val) if metric_mode == "length" else val
    return w


def build_cylinder_spacetime(reg: Register,
                             metric_mode: str = "conductance") -> TesseraBulk:
    """Pure-2D cylinder bulk as a tessera Spacetime, fully pinned.

    Weight semantics: `squaredLength` per edge. In the conductance
    reading big flows are short/easy directions, so squaredLength =
    1/gross; `length` mode uses squaredLength = gross (the length ∝
    value convention). Vertical and diagonal (time-direction) edges
    inherit the incident relationship's weight — the pinned economy
    carried forward.
    """
    n = len(reg.vertices)
    up = lambda v: v + n
    cells: list[list[int]] = []
    tube_cells: list[tuple[int, int, int]] = []
    for t in reg.triangles:
        cells.append(list(t))                          # layer-0 fill (pinned)
        cells.append([up(v) for v in t])               # layer-1 fill
    for (i, j) in reg.edges:
        for tri in ((i, j, up(j)), (i, up(j), up(i))):
            cells.append(list(tri))
            tube_cells.append(tri)

    st = spacetime.Spacetime.fromCells(2, cells)

    wmap = _edge_weight_map(reg, metric_mode)
    scale = float(np.median(list(wmap.values())))

    def weight_for(u: int, v: int) -> float:
        bu, bv = u % n, v % n
        key = frozenset((bu, bv))
        if key in wmap:
            return wmap[key] / scale
        # vertical edge (bu == bv): carry the vertex's median relationship
        vals = [w for e, w in wmap.items() if bu in e]
        return (float(np.median(vals)) if vals else 1.0) / scale

    n_set = 0
    for e in st.getEdgeList().toVector():
        u, v = e.getSource().getId(), e.getTarget().getId()
        e.setSquaredLength(weight_for(u, v))
        n_set += 1
    assert n_set > 0, "no edges found on the bulk spacetime"
    return TesseraBulk(st, reg, n, tube_cells, wmap)


def bulk_harmonics(bulk: TesseraBulk, tol: float = 1e-9):
    """Harmonic 1-cochains of the built bulk, via tessera's HodgeLaplacian,
    as (Psi, edge_cols): Psi is E_bulk x h real, edge_cols the ChainComplex
    column order [u, v] (u < v). Falls back to the combinatorial metric if
    the geometric one degenerates (economic weights need not satisfy
    triangle inequalities)."""
    st = bulk.st
    cc = cobordism.ChainComplex.fromSpacetime(st)
    edge_cols = [tuple(e) for e in cc.kSimplexVertices(1)]
    hl = cobordism.HodgeLaplacian(st)
    for metric in (True, False):
        try:
            harm = hl.harmonics(1, tol) if metric else None
            if not metric:
                L = np.asarray(hl.laplacian(1, metric=False)).reshape(
                    len(edge_cols), len(edge_cols)).real
                w, v = np.linalg.eigh(L)
                Psi = v[:, w < tol * max(w.max(), 1.0)]
                return Psi, edge_cols
            Psi = np.array([np.asarray(h.coeffs()).real
                            if callable(getattr(h, "coeffs", None))
                            else np.asarray(h.coeffs).real
                            for h in harm]).T
            if Psi.size == 0:
                Psi = np.zeros((len(edge_cols), 0))
            if not np.all(np.isfinite(Psi)):
                raise FloatingPointError("degenerate metric harmonics")
            return Psi, edge_cols
        except Exception:
            continue
    raise RuntimeError("no harmonic basis obtainable")


def register_periods(reg: Register, f: np.ndarray) -> np.ndarray:
    """True cycle-integral periods of a flow's harmonic content.

    The columns of reg.harmonics satisfy d1 @ H = 0 exactly — they are
    cycles — so z^T psi is a homology-invariant pairing on closed
    cochains. The register content of a raw flow is the integral of its
    R-harmonic component: p = Z^T (H H^T R f) with Z = H."""
    h = reg.harmonics @ (reg.harmonics.T @ (reg.metric * f))
    return reg.harmonics.T @ h


def _period_operator(bulk: TesseraBulk, edge_cols: list, layer: int) -> np.ndarray:
    """P (b1_base x E_bulk): the year-t register CYCLE INTEGRALS read on
    one layer of the bulk. Bare integrals (no metric factor) — the
    pairing z^T psi is invariant under z -> z + boundary on closed psi
    and under psi -> psi + gradient, which is what makes the intact
    cylinder force equal periods on the two layers."""
    reg = bulk.reg
    n = bulk.n_base
    col_of = {frozenset(e): (k, e) for k, e in enumerate(edge_cols)}
    P = np.zeros((reg.b1, len(edge_cols)))
    off = layer * n
    for k_base, (i, j) in enumerate(reg.edges):
        key = frozenset((i + off, j + off))
        if key not in col_of:
            continue
        col, (u, v) = col_of[key]
        sign = +1.0 if (u, v) == (i + off, j + off) else -1.0
        P[:, col] = sign * reg.harmonics[k_base, :]
    return P


def chart_verdict(bulk: TesseraBulk, p0_target: np.ndarray,
                  p1_target: np.ndarray, cache: dict | None = None) -> dict:
    """Chart-based r_U: can one harmonic state of the built bulk carry
    p0_target on the layer-0 register loops and p1_target on layer-1?

    min_c ||P0 Psi c - p0||^2 + ||P1 Psi c - p1||^2 over harmonic
    coefficients c — the r_U semantics without the relabeling search,
    since the year-t chart fixes the loop order. Returns the relative
    residual (the floor when > 0), plus S_econ of the carrying state.
    """
    if cache is not None and "Psi" in cache:
        Psi, edge_cols = cache["Psi"], cache["edge_cols"]
    else:
        Psi, edge_cols = bulk_harmonics(bulk)
        if cache is not None:
            cache["Psi"], cache["edge_cols"] = Psi, edge_cols
    P0 = _period_operator(bulk, edge_cols, 0) @ Psi
    P1 = _period_operator(bulk, edge_cols, 1) @ Psi
    A = np.vstack([P0, P1])
    b = np.concatenate([p0_target, p1_target])
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = A @ c - b
    scale = float(b @ b)
    psi = Psi @ c
    # S_econ of the carrying state: gauge energy in the pinned metric
    wmap_default = float(np.median(list(bulk.weight_of_edge.values())))
    R = np.array([bulk.weight_of_edge.get(
        frozenset((u % bulk.n_base, v % bulk.n_base)), wmap_default)
        for (u, v) in edge_cols])
    return {
        "r": float(resid @ resid) / max(scale, 1e-300),
        "r_abs": float(resid @ resid),
        "harmonic_dim": Psi.shape[1],
        "s_econ": float(np.sum(R * psi * psi)),
        "coeffs": c,
    }


def surgery_step(bulk: TesseraBulk, p0_target: np.ndarray,
                 p1_target: np.ndarray, n_candidates: int = 8,
                 seed: int = 0) -> dict:
    """One directed surgery probe, proton-style: try removing candidate
    tube cells, keep the cut that most lowers the chart verdict.

    Tube cells are the only candidates — layer cells are the pinned
    boundary data and are never touched (the full primal pin).
    """
    rng = np.random.default_rng(seed)
    base = chart_verdict(bulk, p0_target, p1_target)
    # move = open the tube over one base edge (remove BOTH tube cells).
    # A single tube cell is homologically redundant in a dense tube —
    # the layer identification routes around it — so the pair is the
    # smallest move with any chance of changing what the bulk carries.
    # Candidates are evaluated on FRESH materializations of the cell set
    # (never mutate-and-restore: removeSimplex/createSimplex round-trips
    # do not reproduce the complex — cf. tessera#587).
    by_edge: dict[tuple, list] = {}
    for tri in bulk.tube_cells:
        base_pair = tuple(sorted({v % bulk.n_base for v in tri}))
        by_edge.setdefault(base_pair, []).append(tri)
    candidates = list(by_edge.items())
    order = rng.permutation(len(candidates))[:n_candidates]
    best = None
    for idx in order:
        pair, tris = candidates[idx]
        cut = set(map(tuple, tris))
        trial = _variant(bulk, cut)
        after = chart_verdict(trial, p0_target, p1_target)
        gain = base["r"] - after["r"]
        if best is None or gain > best["gain"]:
            best = {"edge": pair, "cells": sorted(cut), "gain": gain,
                    "after": after, "trial": trial}
    accepted = best is not None and best["gain"] > 1e-12
    if accepted:
        removed = set(map(tuple, best["cells"]))
        bulk.st = best["trial"].st
        bulk.tube_cells = [t for t in bulk.tube_cells if tuple(t) not in removed]
    return {"base": base, "best": best, "accepted": accepted}


def _variant(bulk: TesseraBulk, cut: set) -> TesseraBulk:
    """Fresh TesseraBulk with `cut` tube cells absent — immutable move
    evaluation. Layer cells (the pinned boundary data) are never cut."""
    reg = bulk.reg
    n = bulk.n_base
    up = lambda v: v + n
    cells: list[list[int]] = []
    for t in reg.triangles:
        cells.append(list(t))
        cells.append([up(v) for v in t])
    tube_cells = [t for t in bulk.tube_cells if tuple(t) not in cut]
    cells.extend(list(t) for t in tube_cells)

    st = spacetime.Spacetime.fromCells(2, cells)
    scale = float(np.median(list(bulk.weight_of_edge.values())))
    default_w = scale

    for e in st.getEdgeList().toVector():
        u, v = e.getSource().getId(), e.getTarget().getId()
        key = frozenset((u % n, v % n))
        w = bulk.weight_of_edge.get(key)
        if w is None:
            vals = [wv for ek, wv in bulk.weight_of_edge.items()
                    if (u % n) in ek]
            w = float(np.median(vals)) if vals else default_w
        e.setSquaredLength(w / scale)
    return TesseraBulk(st, reg, n, tube_cells, bulk.weight_of_edge)


