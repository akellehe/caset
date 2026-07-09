"""Bulk cobordism build for the economic register — tessera#602.

Proton/ProtonIngredients semantics with the economic action:

* **Initial boundary — full primal pin.** Layer 0 of the bulk is the
  entire year-t complex: every edge with its transaction value as the
  weight (metric knob unchanged from `econ_register`), every filled
  triad. Nothing about the initial economy is free.
* **Target — dual.** The year-t+1 state enters only through its
  register: period targets in the year-t harmonic chart, imposed on the
  far boundary layer. Never a primal weight assignment.
* **Action.** S_econ[F] = Σ_e R_e F_e² over bulk edges — the integrated
  gauge (electric) energy of the interpolating flow history F, with the
  magnetic content readable per filled face. The flow is exactly
  divergence-free at every bulk vertex (vertical edges carry inter-period
  storage), so the Gauss law holds through the bulk by construction.
* **Verdict.** The optimal action of the observed transition, in excess
  of the IPF-null transition's optimal action, on the same pinned bulk.
  Surgery (removing interior cells) is the discrete move that lowers the
  action when the fixed topology prices the transition high.

Bulk geometry (the cylinder cobordism): two copies of the year-t
complex joined by prisms — one vertical edge per vertex, two tube
triangles per boundary edge, three tetrahedra per filled boundary triad.
Economic time is the third dimension; this is where L₂ and L₃ act
(`bulk_betti`), since the boundary complex itself admits no 3-cells
(tournament obstruction — see gauge_dictionary.md §9).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from econ_register import Register


@dataclasses.dataclass
class Bulk:
    """Cylinder cobordism over a year-t register (layer 0 fully pinned)."""
    reg: Register
    n_base: int                      # vertices per layer
    edges: list[tuple[int, int]]     # bulk edges; layer-1 vertex v -> v+n_base
    kind: np.ndarray                 # 0 = layer0 (pinned), 1 = layer1, 2 = vertical
    weights: np.ndarray              # gross-volume weight per bulk edge (year-t)
    metric: np.ndarray               # R per bulk edge
    triangles: list[tuple[int, int, int]]
    tetrahedra: list[tuple[int, int, int, int]]
    d1: np.ndarray                   # V_bulk x E_bulk
    d2: np.ndarray                   # E_bulk x T_bulk
    layer0_idx: np.ndarray           # bulk-edge index of each base edge, layer 0
    layer1_idx: np.ndarray           # same, layer 1


def build_cylinder(reg: Register) -> Bulk:
    """Prism the year-t complex through one unit of economic time.

    Layer-0 cells inherit the pinned year-t weights; layer-1 and tube
    cells inherit the same weights (the geometry is the pinned initial
    economy carried forward — only the FLOW is free; changing the
    geometry itself is surgery, not relaxation).
    """
    n = len(reg.vertices)
    up = lambda v: v + n
    edges: list[tuple[int, int]] = []
    kind: list[int] = []
    weights: list[float] = []

    layer0_idx = np.zeros(len(reg.edges), dtype=int)
    layer1_idx = np.zeros(len(reg.edges), dtype=int)
    for k, (i, j) in enumerate(reg.edges):
        layer0_idx[k] = len(edges)
        edges.append((i, j)); kind.append(0); weights.append(reg.gross[k])
    for k, (i, j) in enumerate(reg.edges):
        layer1_idx[k] = len(edges)
        edges.append((up(i), up(j))); kind.append(1); weights.append(reg.gross[k])
    vert_idx = {}
    total_gross = float(reg.gross.sum())
    for v in range(n):
        vert_idx[v] = len(edges)
        # storage capacity: a vertex can carry forward what it turns over
        turnover = sum(reg.gross[k] for k, e in enumerate(reg.edges) if v in e)
        edges.append((v, up(v))); kind.append(2)
        weights.append(max(turnover, 1.0))
    diag_idx = {}
    for k, (i, j) in enumerate(reg.edges):
        diag_idx[k] = len(edges)
        edges.append((i, up(j))); kind.append(2)
        weights.append(reg.gross[k])

    triangles: list[tuple[int, int, int]] = []
    for t in reg.triangles:                       # layer-0 fills (pinned)
        triangles.append(t)
    for (i, j, kk) in reg.triangles:              # layer-1 fills
        triangles.append((up(i), up(j), up(kk)))
    for k, (i, j) in enumerate(reg.edges):        # tube: 2 triangles per edge
        triangles.append((i, j, up(j)))
        triangles.append((i, up(j), up(i)))

    # prisms over filled triads -> 3 tetrahedra each (standard split,
    # consistent with the (i, up(j)) diagonals). The two internal prism
    # faces per triad are genuine 2-cells of the complex and must be in
    # the triangle list for the boundary operator d3 to be well-posed.
    tetrahedra: list[tuple[int, int, int, int]] = []
    for (a, b, c) in reg.triangles:
        tetrahedra.append((a, b, c, up(c)))
        tetrahedra.append((a, b, up(b), up(c)))
        tetrahedra.append((a, up(a), up(b), up(c)))
        triangles.append((a, b, up(c)))       # internal: T1 ∩ T2
        triangles.append((a, up(b), up(c)))   # internal: T2 ∩ T3

    E = len(edges)
    eidx = {}
    for k, (i, j) in enumerate(edges):
        eidx[(i, j)] = (k, +1.0)
        eidx[(j, i)] = (k, -1.0)

    d1 = np.zeros((2 * n, E))
    for k, (i, j) in enumerate(edges):
        d1[i, k] = -1.0
        d1[j, k] = +1.0
    d2 = np.zeros((E, len(triangles)))
    for tcol, (a, b, c) in enumerate(triangles):
        for u, v in ((a, b), (b, c), (c, a)):
            if (u, v) not in eidx:      # tube face needing the diagonal
                raise KeyError(f"missing bulk edge {(u, v)}")
            kk, sign = eidx[(u, v)]
            d2[kk, tcol] = sign

    weights_arr = np.asarray(weights)
    R = (1.0 / weights_arr)
    R = R / np.median(R)
    return Bulk(reg, n, edges, np.asarray(kind), weights_arr, R,
                triangles, tetrahedra, d1, d2, layer0_idx, layer1_idx)


def transition_action(bulk: Bulk, p_target: np.ndarray) -> dict:
    """Minimal economic action carrying the pinned year-t economy to a
    far boundary with the target periods.

    Solves min_F Σ R_e F_e² subject to: ∂F = 0 at every bulk vertex
    (exact Gauss law; vertical edges are inter-period storage),
    F = f_t on every layer-0 edge (the full primal pin), and
    Hᵀ R₁ F|layer1 = p_target (the dual target). Equality-constrained
    quadratic programme solved by its KKT system; the value function is
    the action of the cheapest flow history, and its excess over the
    IPF-null target is the certificate analogue.
    """
    reg = bulk.reg
    E = len(bulk.edges)
    free = np.where(bulk.kind != 0)[0]           # layer-1 + vertical + diagonal
    nf = len(free)

    # start from F = f_t on layer 0, zero elsewhere; solve for free part
    F0 = np.zeros(E)
    F0[bulk.layer0_idx] = reg.net

    # constraints on the free variables: A x = b
    rows = []
    rhs = []
    d1f = bulk.d1[:, free]
    rows.append(d1f)
    rhs.append(-bulk.d1 @ F0)                    # conservation everywhere
    Hproj = np.zeros((reg.b1, E))
    Hproj[:, bulk.layer1_idx] = (reg.harmonics * reg.metric[:, None]).T
    rows.append(Hproj[:, free])
    rhs.append(p_target - Hproj @ F0)            # dual period target
    A = np.vstack(rows)
    b = np.concatenate(rhs)

    Rf = bulk.metric[free]
    # KKT: [2 diag(Rf)  Aᵀ; A  0] [x; λ] = [−2 Rf·x0_free…; b]; x0 free part is 0
    m = A.shape[0]
    KKT = np.zeros((nf + m, nf + m))
    KKT[:nf, :nf] = 2.0 * np.diag(Rf)
    KKT[:nf, nf:] = A.T
    KKT[nf:, :nf] = A
    sol, *_ = np.linalg.lstsq(KKT, np.concatenate([np.zeros(nf), b]),
                              rcond=None)
    x = sol[:nf]
    F = F0.copy()
    F[free] = x

    # feasibility of the constraint system (an exact infeasibility here is
    # the hard floor; lstsq returns the least-squares surrogate)
    constraint_residual = float(np.linalg.norm(A @ x - b)
                                / max(np.linalg.norm(b), 1e-300))
    action = float(np.sum(bulk.metric * F * F))
    base_action = float(np.sum(reg.metric * reg.net * reg.net))
    curl = bulk.d2.T @ (bulk.metric * F)         # magnetic content per face
    storage = F[bulk.kind == 2]
    return {
        "action": action,
        "action_rel": action / max(base_action, 1e-300),
        "constraint_residual": constraint_residual,
        "magnetic_energy": float(np.sum(curl * curl)),
        "storage_mass": float(np.abs(storage).sum()),
        "flow": F,
    }


def bulk_betti(bulk: Bulk) -> dict:
    """b1, b2, b3 of the built bulk — L2/L3 live here (the boundary
    complex admits no 3-cells; the bulk's prisms do)."""
    T2, T3 = len(bulk.triangles), len(bulk.tetrahedra)
    tri_index = {tuple(sorted(t)): i for i, t in enumerate(bulk.triangles)}
    d3 = np.zeros((T2, T3))
    for col, tet in enumerate(bulk.tetrahedra):
        s = sorted(tet)
        for omit in range(4):
            face = tuple(s[:omit] + s[omit + 1:])
            d3[tri_index[face], col] = (-1.0) ** omit
    r1 = np.linalg.matrix_rank(bulk.d1)
    r2 = np.linalg.matrix_rank(bulk.d2)
    r3 = np.linalg.matrix_rank(d3) if T3 else 0
    E = len(bulk.edges)
    V = 2 * bulk.n_base
    return {
        "b0": V - r1,
        "b1": E - r1 - r2,
        "b2": T2 - r2 - r3,
        "b3": T3 - r3,
        "euler": V - E + T2 - T3,
    }
