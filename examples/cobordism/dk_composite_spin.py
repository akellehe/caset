# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Composite proton spin — total ½ (proton) vs 3/2 (Δ) from the three register holes (#485).

The *constituent* fermion is spin-½ by construction (#483, #477). This module reads the
**composite** total spin of the emergent bound state — the quantum number that actually
distinguishes a proton (J=½) from a Δ (J=3/2): three spin-½'s combine as
`2⊗2⊗2 = 2 ⊕ 2 ⊕ 4`, i.e. total `J² ∈ {¾ (spin-½), 15/4 (spin-3/2)}`.

A simplicial complex has no global frame, so per-hole spin polarizations aren't directly
comparable. The frame-free fix (as in #477) is **holonomy**: relate the three holes' spinor
frames by the spin-connection **Wilson line** (open-path holonomy), then combine.

## Construction (this file builds (1)–(2); (3)–(5) are the remaining geometry build)
1. **Spinor holonomy element** — `spinor_holonomy(ε, Σ)`: the `Spin(d)` group element
   `exp(ε·Σ)` of a rotation by `ε` in the plane whose generator is `Σ = Σ_ij` (from
   `dk_spin_readout.spin_generators`). Its eigenvalues are `e^{±iε/2}` — the `ε/2`
   half-angle of the Spin→SO double cover (spin-½). **DONE + tested.**
2. **Half-angle / double-cover certificate** — `is_double_cover(...)`: the eigenvalue
   phases are `±ε/2`, not `±ε`. **DONE + tested.**
3. **Per-hinge geometric generator** — from `Simplex.gramMatrix()`: embed the top cell
   (Cholesky of the Gram matrix), get the hinge's 2-plane and its orthogonal-complement
   normal 2-plane, map that bivector onto the `Σ_ij` basis. `deficitAngle()` gives `ε`.
   **TODO** (the heavy piece — the discrete spin connection).
4. **Inter-hole Wilson line** — order-compose the per-hinge holonomies along a dual path
   between two register holes (reuse `WilsonLoop.dualLatticeLoop`/`geodesicLoop` for the
   path). Transports one hole's spinor frame into another's. **TODO.**
5. **Combine → J²** — transport the three holes' carried-rep spinors to a common frame via
   (4), form the (anti)symmetrized 3-spinor bound state, and read its total-spin Casimir
   `J²` → `¾` (proton) or `15/4` (Δ). **TODO.** Honest negative allowed (indefinite mix).
"""
import numpy as np
import scipy.linalg


def spinor_holonomy(eps, sigma):
    """The `Spin(d)` holonomy of a rotation by angle `eps` in the plane with spin generator
    `sigma` (a `Σ_ij = ¼[γ_i,γ_j]` from `dk_spin_readout.spin_generators`, eigenvalues
    `±i/2`): `exp(eps · sigma)`. Eigenvalues are `e^{±i eps/2}` — the `eps/2` half-angle of
    the double cover. `eps` may be complex (Lorentzian deficit)."""
    return scipy.linalg.expm(complex(eps) * np.asarray(sigma, dtype=complex))


def holonomy_phases(eps, sigma):
    """The distinct eigenvalue phases of `spinor_holonomy(eps, sigma)`, sorted."""
    ev = np.linalg.eigvals(spinor_holonomy(eps, sigma))
    return sorted({round(float(np.angle(e)), 8) for e in ev})


def is_double_cover(eps, sigma, tol=1e-7):
    """True iff the holonomy phases are `±eps/2` (the spin-½ double cover) — not `±eps`
    (vector). Certifies the spinor lift is genuinely spin-½."""
    phases = set(np.round(holonomy_phases(eps, sigma), 7))
    want = {round(float(np.angle(np.exp(1j * eps / 2))), 7),
            round(float(np.angle(np.exp(-1j * eps / 2))), 7)}
    return want <= phases


# --- (3)-(5): the remaining geometry build (per-hinge generator from gramMatrix, the
# --- inter-hole Wilson line, and the J² combination) live here; see the module docstring.


if __name__ == "__main__":
    import importlib.util
    import os
    import math

    _here = os.path.dirname(__file__)
    _spec = importlib.util.spec_from_file_location(
        "sr", os.path.join(_here, "dk_spin_readout.py"))
    sr = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(sr)
    eo_spec = importlib.util.spec_from_file_location(
        "eo", os.path.join(_here, "emergent_optimizer.py"))
    eo = importlib.util.module_from_spec(eo_spec)
    eo_spec.loader.exec_module(eo)

    host = eo.build_closed_s4(n_refine=12, seed=0)
    dk = eo.cob.DiracKahler(host)
    sigma = sr.spin_generators(dk)[(1, 2)]            # a Σ_ij spin generator
    for eps in (0.3, 1.0, 2.0, math.pi / 2):
        ph = holonomy_phases(eps, sigma)
        print(f"ε={eps:.3f}: holonomy phases {ph}  (±ε/2 = ±{eps/2:.3f})  "
              f"double-cover={is_double_cover(eps, sigma)}")


# ===== The discrete spin connection + composite J² (built & validated #485; the per-hole
# ===== SPIN STATE is still a placeholder — see the caveat on `composite_j2`). =====
import scipy.linalg as _sla
import collections as _collections


def _dirac_gammas():
    """Standard Euclidean 4×4 Dirac gammas (signature (4,0)); `{γ_a,γ_b}=2δ_ab`."""
    s1 = np.array([[0, 1], [1, 0]], complex)
    s2 = np.array([[0, -1j], [1j, 0]])
    s3 = np.array([[1, 0], [0, -1]], complex)
    return [np.kron(s1, s1), np.kron(s1, s2), np.kron(s1, s3), np.kron(s2, np.eye(2))]


_G = _dirac_gammas()


def rotation_to_spin(R):
    """Lift an SO(4) rotation `R` to its `Spin(4)` (4×4) element via `exp(½ Σ A_ab γ_aγ_b)`,
    `A=log R`. Validated: a rotation by θ gives eigenphases ±θ/2 (the spin-½ double cover)."""
    A = _sla.logm(np.asarray(R, dtype=float)).real
    bivec = sum(0.5 * A[a, b] * (_G[a] @ _G[b])
                for a in range(4) for b in range(a + 1, 4))
    return _sla.expm(bivec)


def embed_cell(cell):
    """Flat ℝ⁴ coordinates of a top cell's vertices from `cell.gramMatrix()` (vertex 0 at
    the origin). Recovers the Gram matrix to ~1e-6."""
    g = np.array(cell.gramMatrix(False)).reshape(4, 4)
    wv, V = np.linalg.eigh(g)
    edges = V @ np.diag(np.sqrt(np.abs(wv)))
    vs = [v.getId() for v in cell.getVertices()]
    coords = {vs[0]: np.zeros(4)}
    for i in range(1, 5):
        coords[vs[i]] = edges[i - 1]
    return coords


def facet_transport(cell_a, cell_b):
    """The `Spin(4)` parallel transport across the shared `(d-1)`-facet of two adjacent top
    cells — orthogonal Procrustes on the 4 shared vertices, lifted to the spinor rep. `None`
    if they don't share a facet."""
    ca, cb = embed_cell(cell_a), embed_cell(cell_b)
    shared = sorted(set(ca) & set(cb))
    if len(shared) < 4:
        return None
    pa = np.array([ca[v] for v in shared]); pa -= pa.mean(0)
    pb = np.array([cb[v] for v in shared]); pb -= pb.mean(0)
    u, _, vt = np.linalg.svd(pb.T @ pa); r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1; r = u @ vt
    return rotation_to_spin(r)


def wilson_line(cells, adj, cell_i, cell_j):
    """The `Spin(4)` holonomy relating two holes' frames: the product of `facet_transport`
    along a BFS dual path between `cell_i` and `cell_j`. `adj` is the dual adjacency."""
    prev = {id(cell_i): None}
    q = _collections.deque([cell_i])
    while q:
        c = q.popleft()
        if c is cell_j:
            break
        for c2 in adj[id(c)]:
            if id(c2) not in prev:
                prev[id(c2)] = c
                q.append(c2)
    if id(cell_j) not in prev:
        return None
    path = []
    c = cell_j
    while c is not None:
        path.append(c); c = prev[id(c)]
    holo = np.eye(4, dtype=complex)
    for a, b in zip(path[::-1], path[::-1][1:]):
        ft = facet_transport(a, b)
        if ft is None:
            return None
        holo = ft @ holo
    return holo


def _dual_adjacency(cells, top_tuple):
    adj = _collections.defaultdict(list)
    tt = {id(c): set(top_tuple(c)) for c in cells}
    for c in cells:
        for c2 in cells:
            if c is not c2 and len(tt[id(c)] & tt[id(c2)]) == 4:
                adj[id(c)].append(c2)
    return adj


def composite_j2(st, holes, top_tuple, per_hole_spinors=None):
    """Total-spin Casimir `J²` of the three-hole bound state, combining the per-hole spin-½'s
    related by the inter-hole `wilson_line`s. `J² → ¾` = spin-½ (proton); `→ 15/4` = 3/2 (Δ).

    CAVEAT (#485, unresolved): `per_hole_spinors` are the per-hole spin STATES. They MUST be
    the emergent register's actual spinor content (the carried reps, via the fiber↔total-space
    map of #477) for a physical verdict. The default below is a **placeholder** ("spin-up in
    each local frame") — it exercises the geometry/Wilson-line machinery but is NOT the
    emergent state, so its `J²` is model-dependent (it came out ~2.16, between ¾ and 15/4)."""
    cells = list(st.getTopSimplices())
    adj = _dual_adjacency(cells, top_tuple)

    def cell_of(h):
        hv = set(h)
        return max(cells, key=lambda c: len(hv & set(top_tuple(c))))

    hc = [cell_of(h) for h in holes[:3]]
    lines = [wilson_line(cells, adj, hc[0], hc[j]) for j in range(3)]
    if any(u is None for u in lines):
        return None
    if per_hole_spinors is None:                         # PLACEHOLDER — not the emergent state
        up = np.zeros(4, complex); up[0] = 1.0
        per_hole_spinors = [up, up, up]
    psi = [lines[j] @ per_hole_spinors[j] for j in range(3)]
    state = np.kron(np.kron(psi[0], psi[1]), psi[2])
    state = state / np.linalg.norm(state)
    sg = [(-1j * 0.25) * (_G[(k + 1) % 3 + 1] @ _G[(k + 2) % 3 + 1]
                          - _G[(k + 2) % 3 + 1] @ _G[(k + 1) % 3 + 1]) for k in range(3)]

    def s_total(a):
        out = np.zeros((64, 64), complex)
        for i in range(3):
            ops = [np.eye(4)] * 3; ops[i] = sg[a]
            out += np.kron(np.kron(ops[0], ops[1]), ops[2])
        return out

    j2 = sum(s_total(a) @ s_total(a) for a in range(3))
    return float((state.conj() @ j2 @ state).real)
