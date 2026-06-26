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

## The build — one unified per-cell frame for BOTH extraction and transport

The attempt-1 failure was inconsistent frames: the spinor extraction's per-cell frame and
the Wilson line's frame were *independent* conventions, so the readout was not covariant.
The fix is a single frame `F(c)` used by both, built from the cell's **vertex SET** so it is
relabel-invariant and gauge-covariant:

1. `embed_cell(c)` — flat ℝ⁴ coords from `c.gramMatrix()` (canonically right-handed).
2. `cell_frame(c)` — `F(c)` = the centered principal axes (inertia-tensor eigenvectors,
   axis signs fixed by the order-independent third moment). Depending only on the point set,
   the frame-local coords are **relabel-invariant** and **gauge-covariant** (`F → R F`),
   unlike an origin-vertex polar/QR frame which rotates by a generic SO(4) when relabeling
   moves the origin.
3. `facet_transport(a, b)` — the `Spin(4)` transport `b → a` across the shared facet, the
   rotation aligning the shared facet's coords in `F(b)` to those in `F(a)`, lifted by
   `rotation_to_spin`. Shares `F(c)` with the extraction by construction (the attempt-1 fix).
4. `wilson_line(i, j)` — composes `facet_transport` along a BFS dual path `j → … → i`.
5. `emergent_spinor` — the per-hole spinor from the carried representative `psi`
   (`EigenstateSynthesis(st, 3).carriedRepresentative`): least-squares the cell's tetrahedral
   faces' trivector minors (in `F(c)`) against `psi`, map the resulting 3-form to the Clifford
   algebra (`Phi = Σ ω_t γ_iγ_jγ_k`), and read `s = Phi · [1,0,0,0]`.
6. `composite_j2` — transport the three holes' spinors to a common frame via the Wilson lines,
   form the product state, and read the total-spin Casimir `J²`.

`spinor_holonomy` / `holonomy_phases` / `is_double_cover` are the spin-½ double-cover
building blocks (a rotation by ε gives eigenphases ±ε/2). Validation gates (GAUGE, RELABEL)
live in the test module and in `__main__`. **Post-hoc only, never a loop condition.**

Outcome (see `docs/design/composite_proton_spin_findings.md`): the readout robustly identifies
a **three–spin-½ (baryon)** bound state — `|⟨S⟩|=½` per hole and the product `J²` floors at the
n=3 value `3/2`. The per-cell frame is the **dual-edge** frame (`cell_frame`): its axes come
from the cell's circumcenter-to-neighbor-circumcenter vectors, whose *lengths* carry the
neighbors' sizes, so the frame is non-degenerate even when the cell's own vertices are
near-symmetric — this makes GAUGE and RELABEL pass on every non-uniform structure (b₃=3..7 and
real converged geometry), where a primal-vertex frame failed sporadically. (Only a perfectly
uniform metric leaves the dual edges equal too — a genuine, unphysical symmetry.) The one
remaining limitation is physical, not numerical: the **composite** total spin (proton ½ vs
Δ 3/2) is an *entanglement* property, and a product of per-hole spinors floors at `3/2` and
cannot reach the proton ¾. The `joint_*` functions are the exploratory step toward the
entangled joint-state read.
"""
import cmath
import collections
import math

import numpy as np
import scipy.linalg

import tessera as T

cob = T.cobordism

# The 4 trivector index triples over the cell frame's axes (a 3-blade in ℝ⁴ is dual to a
# vector; these are its 4 components). The SAME triples index the det-minors and the gammas.
_TRIPLES = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]


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
    A = scipy.linalg.logm(np.asarray(R, dtype=float)).real
    bivec = sum(0.5 * A[a, b] * (_G[a] @ _G[b])
                for a in range(4) for b in range(a + 1, 4))
    return scipy.linalg.expm(bivec)


def embed_cell(cell):
    """Flat ℝ⁴ coordinates of a top cell's vertices from `cell.gramMatrix()` (vertex 0 at
    the origin), canonically right-handed. Recovers the Gram matrix to ~1e-6."""
    g = np.array(cell.gramMatrix(False)).reshape(4, 4)
    wv, V = np.linalg.eigh(g)
    edges = V @ np.diag(np.sqrt(np.abs(wv)))
    if np.linalg.det(edges) < 0:                       # canonical right-handed embedding
        edges[:, -1] *= -1
    vs = [v.getId() for v in cell.getVertices()]
    coords = {vs[0]: np.zeros(4)}
    for i in range(1, 5):
        coords[vs[i]] = edges[i - 1]
    return coords


def _facet_neighbors(cells, top_tuple):
    """`{facet frozenset: [top cells sharing it]}` — the dual adjacency, by vertex SET."""
    fmap = collections.defaultdict(list)
    for c in cells:
        vs = top_tuple(c)
        for j in range(len(vs)):
            fmap[frozenset(vs[:j] + vs[j + 1:])].append(c)
    return fmap


def _circumcenter(cell, coords, vs):
    bary = np.asarray(cell.circumcenterBarycentric(), dtype=float)
    return sum(bary[i] * coords[vs[i]] for i in range(len(vs)))


def _facet_outward_normal(coords, facet, apex):
    """Unit normal to the facet's 3-plane (in these coords), pointing away from `apex`."""
    pts = np.array([coords[v] for v in facet])
    cen = pts.mean(0)
    _u, _s, vt = np.linalg.svd(pts - cen)
    n = vt[-1] / np.linalg.norm(vt[-1])
    return cen, (-n if np.dot(coords[apex] - cen, n) > 0 else n)


def _dual_height(cell):
    """`{facet frozenset: |distance from this cell's circumcenter to that facet|}` — the
    intrinsic dual half-edge each facet contributes, in the cell's own embedding."""
    coords = embed_cell(cell)
    vs = [v.getId() for v in cell.getVertices()]
    cc = _circumcenter(cell, coords, vs)
    out = {}
    for j in range(5):
        facet = frozenset(vs[:j] + vs[j + 1:])
        cen, n = _facet_outward_normal(coords, sorted(facet), vs[j])
        out[facet] = abs(float((cc - cen) @ n))
    return out


def cell_frame(cell, neighbors=None):
    """The unified per-cell frame `F(c)` and its (raw) coords, used by BOTH the spinor
    extraction and the Wilson-line transport, built from a VERTEX-SET-derived point cloud so
    the frame-local coords are **relabel-invariant** (the same physical frame whatever the
    labeling) and **gauge-covariant** (`F → R F` under a global rotation `R`) — unlike an
    origin-vertex polar/QR frame, which rotates by a generic SO(4) when relabeling moves the
    origin and breaks covariance.

    `neighbors` is the `_facet_neighbors` map. When given (the default in the readout), the
    point cloud is the cell's five **dual-edge vectors** — circumcenter → facet-neighbor
    circumcenter, `(h_a + h_b)·n_facet`. The edge LENGTH carries the neighbor's size, so these
    are generically anisotropic even when the cell itself is regular: this is what rescues the
    readout on near-symmetric cells, where the cell's own vertices (the `neighbors=None`
    fallback) are isotropic and have no canonical frame. (A perfectly uniform metric leaves
    even the dual edges equal — a genuine symmetry, unreachable by a non-degenerate metric.)

    The frame is the inertia (principal-axis) eigenbasis of the centered point cloud; a
    coincident-eigenvalue block is canonically resolved by the order-independent third-moment
    tensor, and each axis' sign by its third moment. Returns `(coords, F, vs)`."""
    coords = embed_cell(cell)
    vs = [v.getId() for v in cell.getVertices()]
    if neighbors is None:                              # fallback: the cell's own vertices
        cloud = np.array([coords[v] for v in vs])
    else:                                              # default: dual-edge vectors
        cc = _circumcenter(cell, coords, vs)
        vset = tuple(sorted(vs))
        h_self = _dual_height(cell)
        rows = []
        for j in range(5):
            facet = frozenset(vs[:j] + vs[j + 1:])
            cen, n = _facet_outward_normal(coords, sorted(facet), vs[j])
            h_b = 0.0
            for nb in neighbors.get(facet, ()):
                if tuple(sorted(x.getId() for x in nb.getVertices())) != vset:
                    h_b = _dual_height(nb)[facet]
                    break
            rows.append((h_self[facet] + h_b) * n)
        cloud = np.array(rows)
    pc = cloud - cloud.mean(0)
    w, V = np.linalg.eigh(pc.T @ pc)                   # principal axes as columns, ascending
    proj = pc @ V
    wsq = (proj ** 2).sum(1)
    i = 0                                              # lift coincident moments with the
    while i < 4:                                       # third-moment tensor on each block
        j = i + 1
        while j < 4 and abs(w[j] - w[i]) < 1e-6 * max(1.0, abs(w[i])):
            j += 1
        if j - i > 1:
            sub = proj[:, i:j]
            _ww, vv = np.linalg.eigh((sub * wsq[:, None]).T @ sub)
            V[:, i:j] = V[:, i:j] @ vv
        i = j
    proj = pc @ V
    for k in range(4):                                # canonical, order-independent signs
        if np.sum(proj[:, k] ** 3) < 0:
            V[:, k] *= -1
    return coords, V, vs


def _frames(cells, top_tuple):
    """Precompute the dual-edge `cell_frame` of every top cell, keyed by `id(cell)`."""
    fmap = _facet_neighbors(cells, top_tuple)
    return {id(c): cell_frame(c, fmap) for c in cells}


def facet_transport(cell_a, cell_b, frames=None):
    """The `Spin(4)` transport mapping `cell_b`'s spinor frame to `cell_a`'s, from the SAME
    `cell_frame`s the extraction uses: align `b`'s shared-facet coords (in `F(b)`) to `a`'s
    (in `F(a)`) by orthogonal Procrustes, det-corrected to a proper rotation, then
    `rotation_to_spin`. `frames` is the precomputed `_frames` map. `None` if no shared facet."""
    ca, Fa, _ = frames[id(cell_a)] if frames else cell_frame(cell_a)
    cb, Fb, _ = frames[id(cell_b)] if frames else cell_frame(cell_b)
    shared = sorted(set(ca) & set(cb))
    if len(shared) < 4:
        return None
    xa = np.array([Fa.T @ ca[v] for v in shared]); xa -= xa.mean(0)
    xb = np.array([Fb.T @ cb[v] for v in shared]); xb -= xb.mean(0)
    u, _s, vt = np.linalg.svd(xb.T @ xa)               # rot with rot @ xb = xa is V Uᵀ
    d = np.eye(4)
    d[3, 3] = np.sign(np.linalg.det(vt.T @ u.T))
    return rotation_to_spin(vt.T @ d @ u.T)


def _dual_adjacency(cells, top_tuple):
    """Dual adjacency keyed by `id(cell)`: two top cells are adjacent iff they share a
    `(d-1)`-facet (4 common vertices)."""
    adj = collections.defaultdict(list)
    tt = {id(c): set(top_tuple(c)) for c in cells}
    for c in cells:
        for c2 in cells:
            if c is not c2 and len(tt[id(c)] & tt[id(c2)]) == 4:
                adj[id(c)].append(c2)
    return adj


def wilson_line(cells, adj, cell_i, cell_j, frames=None):
    """The `Spin(4)` holonomy mapping `cell_j`'s frame to `cell_i`'s: the product of
    `facet_transport` along a BFS dual path `j → … → i`. `None` if disconnected."""
    prev = {id(cell_i): None}
    q = collections.deque([cell_i])
    while q:
        c = q.popleft()
        for c2 in adj[id(c)]:
            if id(c2) not in prev:
                prev[id(c2)] = c
                q.append(c2)
    if id(cell_j) not in prev:
        return None
    holo = np.eye(4, dtype=complex)
    c = cell_j
    while prev[id(c)] is not None:
        p = prev[id(c)]
        ft = facet_transport(p, c, frames)             # maps c-frame → p-frame
        if ft is None:
            return None
        holo = ft @ holo
        c = p
    return holo


def _cells3_by_set(st):
    """`{frozenset(3-cell): (index, stored-tuple)}` over the canonical 3-cell ordering that
    indexes the carried representative, plus the count `|C_3|`."""
    cells3 = cob.HodgeLaplacian(st).harmonics(3)[0].simplices()
    return {frozenset(t): (i, list(t)) for i, t in enumerate(cells3)}, len(cells3)


def emergent_spinor(by_set, psi, frame):
    """The per-hole spinor extracted from the carried representative `psi` (a 3-cochain) over
    the hole cell's five tetrahedral faces, in the cell's own frame. `frame` is the cell's
    `cell_frame` triple `(coords, F, vs)`. Order-agnostic: faces are matched by vertex SET,
    oriented from the cochain's stored order (never sorted by id).

    For each face: build its trivector (the 4 det-minors of the 3 edge vectors in `F` over
    `_TRIPLES`) as an `M`-row and take `psi` on that 3-cell as the RHS. `omega = lstsq(M, b)`
    is the carried 3-form's components; `Phi = Σ omega_t γ_iγ_jγ_k` maps it into the Clifford
    algebra, and `s = Phi · [1,0,0,0]` (normalized) is the spinor."""
    coords, F, vs = frame
    x = {v: F.T @ coords[v] for v in coords}
    rows, rhs = [], []
    for drop in range(5):
        face = tuple(vs[i] for i in range(5) if i != drop)
        key = frozenset(face)
        if key not in by_set:
            continue
        idx, stored = by_set[key]
        edge = np.array([x[stored[t]] - x[stored[0]] for t in (1, 2, 3)])   # 3×4
        rows.append([np.linalg.det(edge[:, list(tr)]) for tr in _TRIPLES])
        rhs.append(psi[idx])
    omega, *_ = np.linalg.lstsq(np.array(rows, dtype=complex),
                                np.array(rhs, dtype=complex), rcond=None)
    phi = sum(omega[t] * (_G[i] @ _G[j] @ _G[k]) for t, (i, j, k) in enumerate(_TRIPLES))
    s = phi @ np.array([1, 0, 0, 0], complex)
    n = np.linalg.norm(s)
    return s / n if n > 1e-30 else s


def emergent_spinors(st, holes, top_tuple):
    """The three emergent per-hole spinors, each in its own hole cell's (dual-edge) frame."""
    cells = list(st.getTopSimplices())
    frames = _frames(cells, top_tuple)

    def cell_of(h):
        hv = set(h)
        return max(cells, key=lambda c: len(hv & set(top_tuple(c))))

    by_set, n3 = _cells3_by_set(st)
    es = cob.EigenstateSynthesis(st, 3)
    out = []
    for h in holes[:3]:
        psi = np.asarray(es.carriedRepresentative([list(h)], [1.0]), dtype=complex)
        out.append(emergent_spinor(by_set, psi, frames[id(cell_of(h))]))
    return out


# The single-Dirac spatial spin operators S_a (eigenvalues ±½ each; the single-particle
# Casimir Σ_a S_a² = ¾ I, i.e. spin-½), in the (1+3) convention (axes 1,2,3 spatial).
_SG = [(-1j * 0.25) * (_G[(k + 1) % 3 + 1] @ _G[(k + 2) % 3 + 1]
                       - _G[(k + 2) % 3 + 1] @ _G[(k + 1) % 3 + 1]) for k in range(3)]


def composite_j2(st, holes, top_tuple, per_hole_spinors):
    """Total-spin Casimir `J²` of the three-hole bound state: transport each hole's spinor to
    a common frame via the inter-hole `wilson_line`s, form the product state, and read
    `J² = Σ_a (Σ_i S_a^{(i)})²`. `→ ¾` = spin-½ (proton); `→ 15/4` = 3/2 (Δ); an intermediate
    value is an indefinite mixture (an honest negative). `None` if a Wilson line is missing.

    `per_hole_spinors` are the three emergent per-hole spinors (see `emergent_spinors`), each
    in its own hole cell's frame."""
    cells = list(st.getTopSimplices())
    adj = _dual_adjacency(cells, top_tuple)
    frames = _frames(cells, top_tuple)

    def cell_of(h):
        hv = set(h)
        return max(cells, key=lambda c: len(hv & set(top_tuple(c))))

    hc = [cell_of(h) for h in holes[:3]]
    lines = [wilson_line(cells, adj, hc[0], hc[j], frames) for j in range(3)]
    if any(u is None for u in lines):
        return None
    psi = [lines[j] @ per_hole_spinors[j] for j in range(3)]
    state = np.kron(np.kron(psi[0], psi[1]), psi[2])
    state = state / np.linalg.norm(state)

    def s_total(a):
        out = np.zeros((64, 64), complex)
        for i in range(3):
            ops = [np.eye(4)] * 3
            ops[i] = _SG[a]
            out += np.kron(np.kron(ops[0], ops[1]), ops[2])
        return out

    j2 = sum(s_total(a) @ s_total(a) for a in range(3))
    return float((state.conj() @ j2 @ state).real)


def emergent_j2(st, holes, top_tuple):
    """`composite_j2` on the emergent per-hole spinors — the full composite-spin readout."""
    return composite_j2(st, holes, top_tuple, emergent_spinors(st, holes, top_tuple))


# ===== Joint-state read (EXPLORATORY, #485 §5) — attempt the ½-vs-3/2 channel =====
# `composite_j2` multiplies three INDEPENDENTLY-extracted per-hole spinors, so it is a product
# state: its `J²` floors at 3/2 and can never reach the proton's ¾ (an entangled, mixed-
# symmetry combination). The joint read below instead extracts the three spins from ONE
# correlated `carriedRepresentative([h0,h1,h2], [1,ω,ω²])` carrying the color singlet across all
# three holes, so the color phases imprint correlations. It leans toward the proton channel and
# can produce definite channels (a clean Δ on some structures), but it inherits relabel-
# sensitivity from carriedRepresentative's multi-hole/complex-target handling, so it is NOT yet
# a clean (gate-passing) observable — a prototype toward the entangled joint-state read.
_W3 = cmath.exp(2j * math.pi / 3)
_PAULI = [np.array([[0, 1], [1, 0]], complex),
          np.array([[0, -1j], [1j, 0]]),
          np.array([[1, 0], [0, -1]], complex)]


def joint_spinors(st, holes, top_tuple, target=(1.0, _W3, _W3 * _W3)):
    """The three per-hole spinors read from the SINGLE joint carried representative of the color
    singlet `[1,ω,ω²]` over the three holes (vs `emergent_spinors`' independent per-hole reads)."""
    cells = list(st.getTopSimplices())
    frames = _frames(cells, top_tuple)

    def cell_of(h):
        hv = set(h)
        return max(cells, key=lambda c: len(hv & set(top_tuple(c))))

    by_set, _n3 = _cells3_by_set(st)
    psi = np.asarray(cob.EigenstateSynthesis(st, 3).carriedRepresentative(
        [list(h) for h in holes[:3]], list(target)), dtype=complex)
    return [emergent_spinor(by_set, psi, frames[id(cell_of(h))]) for h in holes[:3]]


def joint_j2(st, holes, top_tuple):
    """`composite_j2` on the joint (color-correlated) per-hole spinors — the exploratory
    ½-vs-3/2 attempt."""
    return composite_j2(st, holes, top_tuple, joint_spinors(st, holes, top_tuple))


def spin_channel_weights(st, holes, top_tuple, per_hole_spinors):
    """Decompose the transported three-spin state into the total-spin channels: returns
    `(w_delta, w_proton)` = weights in `J=3/2` (Δ) and `J=½` (proton). Reduces each transported
    Dirac spinor to its Bloch vector, builds the 3-qubit state, and projects with
    `P_{3/2} = (J² − ¾)/3`. For a product state `w_delta ≥ ¼` (so `w_proton ≤ ¾`) — the
    composite ¾ proton is unreachable without genuine entanglement (#485 §5)."""
    cells = list(st.getTopSimplices())
    adj = _dual_adjacency(cells, top_tuple)
    frames = _frames(cells, top_tuple)

    def cell_of(h):
        hv = set(h)
        return max(cells, key=lambda c: len(hv & set(top_tuple(c))))

    hc = [cell_of(h) for h in holes[:3]]
    lines = [wilson_line(cells, adj, hc[0], hc[j], frames) for j in range(3)]
    if any(u is None for u in lines):
        return None
    qubits = []
    for j in range(3):
        s = lines[j] @ per_hole_spinors[j]
        s = s / np.linalg.norm(s)
        n = np.array([float(np.real(s.conj() @ _SG[a] @ s)) for a in range(3)])
        nn = np.linalg.norm(n)
        rho = 0.5 * (np.eye(2) + (sum(n[a] / nn * _PAULI[a] for a in range(3)) if nn > 1e-12
                                  else np.zeros((2, 2))))
        _w, vec = np.linalg.eigh(rho)
        qubits.append(vec[:, -1])
    state = np.kron(np.kron(qubits[0], qubits[1]), qubits[2])
    state = state / np.linalg.norm(state)

    def j_total(a):
        out = np.zeros((8, 8), complex)
        for i in range(3):
            ops = [np.eye(2)] * 3
            ops[i] = 0.5 * _PAULI[a]
            out += np.kron(np.kron(ops[0], ops[1]), ops[2])
        return out

    j2 = sum(j_total(a) @ j_total(a) for a in range(3))
    w_delta = float((state.conj() @ ((j2 - 0.75 * np.eye(8)) / 3.0) @ state).real)
    return w_delta, 1.0 - w_delta


if __name__ == "__main__":
    import importlib.util
    import math
    import os

    _here = os.path.dirname(__file__)

    def _load(name):
        spec = importlib.util.spec_from_file_location(name, os.path.join(_here, name + ".py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    sr = _load("dk_spin_readout")
    eo = _load("emergent_optimizer")

    # (1) the spin-½ double-cover building block
    host = eo.build_closed_s4(n_refine=12, seed=0)
    sigma = sr.spin_generators(eo.cob.DiracKahler(host))[(1, 2)]
    for eps in (0.3, 1.0, 2.0, math.pi / 2):
        ph = holonomy_phases(eps, sigma)
        print(f"ε={eps:.3f}: holonomy phases {ph}  (±ε/2 = ±{eps/2:.3f})  "
              f"double-cover={is_double_cover(eps, sigma)}")

    # (2) the composite J² readout on a converged b₃ structure, with the validation gates
    import cmath
    w = cmath.exp(2j * math.pi / 3)
    for S in range(3, 40):
        host = eo.build_closed_s4(n_refine=20, seed=S)
        opt = eo.EmergentOptimizer(host, [[1, w, w * w], [1, w * w, w]], [1, w, w * w],
                                   degrees=(3,), gamma=1.0, seed=S)
        sv = [v.getId() for v in host.getVertexList().toVector()][:2]
        opt.construct_inputs(sv, rounds=12)
        opt.run_stage1(max_steps=30, n_candidates=8, patience=8)
        holes = eo.emergent_holes(opt.st, 3)
        if len(holes) >= 3:
            break
    st = opt.st
    T.ReggeSolver(st, T.MatterConfiguration())            # materialize the skeleton
    j2 = emergent_j2(st, holes, eo._top_tuple)
    print(f"\nseed {S}: emergent composite J² = {j2:.4f}  "
          f"(¾ = {0.75} proton / 15/4 = {3.75} Δ)")

    # (3) the mandatory validation gates: J² invariant under a per-cell SO(4) gauge of the
    # embedding (GAUGE) and under a vertex-id relabeling (RELABEL).
    import random
    _orig_embed = embed_cell
    _rng = np.random.default_rng(7)
    _rmap = {}

    def _gauged_embed(cell):
        c = _orig_embed(cell)
        key = tuple(sorted(c))
        if key not in _rmap:
            a = _rng.standard_normal((4, 4))
            _rmap[key] = scipy.linalg.expm(a - a.T)
        r = _rmap[key]
        return {v: r @ x for v, x in c.items()}

    embed_cell = _gauged_embed
    j2_gauge = emergent_j2(st, holes, eo._top_tuple)
    embed_cell = _orig_embed

    allv = sorted({v for s in st.getTopSimplices() for v in eo._top_tuple(s)})
    shuf = allv[:]
    random.Random(3).shuffle(shuf)
    perm = dict(zip(allv, shuf))
    rcells = [[perm[v] for v in eo._top_tuple(s)] for s in st.getTopSimplices()]
    redges = {}
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        redges[tuple(sorted((perm[a], perm[b])))] = complex(e.getSquaredLength())
    st2 = T.Spacetime.fromCells(4, rcells, 1.0, 0.0)
    for e in st2.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        e.setSquaredLength(redges[tuple(sorted((a, b)))])
    T.ReggeSolver(st2, T.MatterConfiguration())
    holes2 = [tuple(sorted(perm[v] for v in h)) for h in holes[:3]]
    j2_relabel = emergent_j2(st2, holes2, eo._top_tuple)
    print(f"  GAUGE   |ΔJ²| = {abs(j2_gauge - j2):.2e}")
    print(f"  RELABEL |ΔJ²| = {abs(j2_relabel - j2):.2e}")

    # (4) constituent content + the composite-channel decomposition. The constituents are
    # spin-½ (|⟨S⟩|=½); the product J² lives in [3/2, 15/4] (the three-spin-½ baryon range),
    # so the proton ¾ is unreachable without entanglement. The joint (color-correlated) read
    # leans toward the proton channel (§5).
    spn = emergent_spinors(st, holes, eo._top_tuple)
    w_delta, w_proton = spin_channel_weights(st, holes, eo._top_tuple, spn)
    print(f"  constituents: product J² floor 3/2 ≤ {j2:.4f} ≤ 15/4 ceiling (three spin-½)")
    print(f"  channels (independent): Δ(3/2)={w_delta:.3f}  proton(½)={w_proton:.3f}")
    jw = spin_channel_weights(st, holes, eo._top_tuple,
                              joint_spinors(st, holes, eo._top_tuple))
    print(f"  channels (joint [1,ω,ω²]): Δ(3/2)={jw[0]:.3f}  proton(½)={jw[1]:.3f}  "
          f"J²={joint_j2(st, holes, eo._top_tuple):.4f}")
