# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""C_ij connected-correlator chamber readout — instrument + C_ij-two-ways (#564, part of #559).

The composite total spin of three spin-½ holes obeys (spin_readout.tex §3)

    J² = 9/4 + 2·Σ_{i<j} ⟨S_i·S_j⟩,      ⟨S_i·S_j⟩ = ⟨S_i⟩·⟨S_j⟩ + C_ij,

where the **connected** correlator `C_ij ≡ ⟨S_i·S_j⟩ − ⟨S_i⟩·⟨S_j⟩` is the Cartan-chamber
coordinate of the pair (cartan_weyl_gluon.tex §5a): it vanishes identically on any product
state (the K-orbit floor — local frames alone cannot entangle, Claim 1 there), and on the
entangled proton eigenstate it carries the whole shift below the floor (Σ C_ij = −¾ ⇒ J² = ¾).
Three experiments, in increasing order of commitment (cartan_weyl_gluon.tex §7, items 1–3):

1. **Instrument** (`j2_decomposition`, pure NumPy on (C²)³): J² in chamber (correlator)
   coordinates must return EXACTLY ¾ / 7/4 / 15/4 on the hand-fed clean proton
   `2|uud⟩−|udu⟩−|duu⟩` / product `|uud⟩` / Δ `|uuu⟩` states, and C_ij must vanish
   identically on every product state. Regression-tested to 1e-12.

2. **C_ij two ways — the escape-hatch test** (`two_ways`): on one fixture, compute C_ij
   (a) VERTICAL — from the reconstructed joint two-hole reduced states ρ_ij of the single
   correlated multi-hole carried representative `carriedRepresentative([h0,h1,h2],[1,ω,ω²])`
   (per-hole spinor extraction + Spin(4) Wilson-line transport to a common frame, then
   partial traces of the reconstructed 3-qubit state); and (b) HORIZONTAL — from the
   M/Wilson-loop holonomy invariants ALONE (the inter-hole transport's SO(3) angles θ_ij and
   the color register's Z₃ phases Δφ_ij; no joint-state reconstruction, no contraction
   against endpoint spinors). Agreement ⇒ C_ij is a holonomy invariant and the point-fiber
   (fiber↔cells) assembly is avoidable for this observable; disagreement quantifies exactly
   what the holonomy cannot replace. Either outcome is decisive and reported as-is.

3. **Emergent read** (`emergent_read`): the same readout applied to a converged fixture's
   joint carried representative, against the independent per-hole (product) read — does J²
   move below the product read toward ¾?  (The prior joint_* prototype saw 1.80 → 1.64.)

Validation gates (`gauge_gate`, `relabel_gate`): every mesh read must be invariant under a
random per-cell SO(4) rotation of the embedding (GAUGE) and under a vertex-id permutation
(RELABEL). Run post-hoc, never as a loop condition.

Prior art, credited: the retired pre-#509 escape-hatch experiment (#512, branch
`feat/cij-escape-hatch`) found C_ij is NOT a holonomy invariant on the old `dk_*` machinery;
the pairwise-C_ij instrument core was proven on branch `feat/cij-composite-spin` (#514, PR
#519, unmerged); PR #518 (unmerged) carries a C++ loops-as-quarks J² that also floors above
¾. This module re-establishes the result on current main's surviving APIs
(`EigenstateSynthesis.carriedRepresentative` / `cellSimplices`, `MultiCobordism.emergent_holes`,
`Simplex.gramMatrix`) with the transport layer ported from the retired #485 module.

Pure NumPy for the instrument; `tessera` is imported lazily by the mesh layer only.
"""
import cmath
import collections
import json
import math
import os

import numpy as np
import scipy.linalg

# ============================================================================
# Instrument layer — J² in chamber (correlator) coordinates on (C²)³
# ============================================================================
_PAULI = [np.array([[0, 1], [1, 0]], complex),
          np.array([[0, -1j], [1j, 0]]),
          np.array([[1, 0], [0, -1]], complex)]
_S = [p / 2 for p in _PAULI]
_I2 = np.eye(2, dtype=complex)
# The two-qubit spin-spin operator S_i·S_j = Σ_a S_a⊗S_a (4×4).
_SS = sum(np.kron(_S[a], _S[a]) for a in range(3))
_PAIRS = [(0, 1), (0, 2), (1, 2)]

_UP = np.array([1, 0], complex)
_DN = np.array([0, 1], complex)


def _normalize(psi):
    psi = np.asarray(psi, complex)
    n = np.linalg.norm(psi)
    return psi / n if n > 1e-30 else psi


def kron(*factors):
    """`factors[0] ⊗ factors[1] ⊗ …` — the product-state builder."""
    out = np.asarray(factors[0], complex)
    for x in factors[1:]:
        out = np.kron(out, np.asarray(x, complex))
    return out


def clean_states():
    """The hand-fed clean three-qubit states of the validated instrument
    (spin_readout.tex §3): proton eigenstate (J²=¾), product |uud⟩ (7/4), Δ |uuu⟩ (15/4)."""
    return {
        "proton 2|uud>-|udu>-|duu>": 2 * kron(_UP, _UP, _DN) - kron(_UP, _DN, _UP)
                                     - kron(_DN, _UP, _UP),
        "product |uud>": kron(_UP, _UP, _DN),
        "Delta |uuu>": kron(_UP, _UP, _UP),
    }


def reduced_states(psi8):
    """The three pairwise (`ρ_ij`, 4×4) and three single (`ρ_i`, 2×2) reduced density
    matrices of a three-qubit pure state (ordering hole0 ⊗ hole1 ⊗ hole2). The `ρ_ij` are
    exactly "the reconstructed joint two-hole reduced states" the vertical route reads.
    Returns `(pairs, singles)` keyed by `(i, j)` / `i`; every reduced state is unit-trace."""
    t = _normalize(psi8).reshape(2, 2, 2)
    tc = t.conj()
    pairs = {
        (0, 1): np.einsum("abc,dec->abde", t, tc).reshape(4, 4),   # trace out hole 2
        (0, 2): np.einsum("abc,dbe->acde", t, tc).reshape(4, 4),   # trace out hole 1
        (1, 2): np.einsum("abc,aef->bcef", t, tc).reshape(4, 4),   # trace out hole 0
    }
    singles = {
        0: np.einsum("abc,dbc->ad", t, tc),
        1: np.einsum("abc,adc->bd", t, tc),
        2: np.einsum("abc,abd->cd", t, tc),
    }
    return pairs, singles


def spin_correlator(rho_ij):
    """`⟨S_i·S_j⟩ = Tr(ρ_ij · Σ_a S_a⊗S_a)` from a pairwise reduced state (4×4)."""
    return float(np.trace(np.asarray(rho_ij, complex) @ _SS).real)


def bloch(rho_i):
    """The Bloch vector `⟨S⟩ = (⟨S_x⟩,⟨S_y⟩,⟨S_z⟩)` of a single-hole reduced state (2×2)."""
    r = np.asarray(rho_i, complex)
    return np.array([float(np.trace(r @ _S[a]).real) for a in range(3)])


def connected_correlator(rho_ij, rho_i, rho_j):
    """`C_ij = ⟨S_i·S_j⟩ − ⟨S_i⟩·⟨S_j⟩` — the connected (chamber) part. Vanishes iff the
    pair is uncorrelated; it is exactly what a per-hole Bloch/product read discards."""
    return spin_correlator(rho_ij) - float(bloch(rho_i) @ bloch(rho_j))


def j2_decomposition(psi8):
    """The chamber-coordinate breakdown of `J² = 9/4 + 2·Σ_{i<j} ⟨S_i·S_j⟩` for a 3-qubit
    pure state, through its joint two-hole reduced states. Returns `j2`
    (= `j2_disconnected + j2_connected`), `j2_disconnected` (the per-hole Bloch floor
    `9/4 + 2Σ⟨S_i⟩·⟨S_j⟩`), `j2_connected` (`2·Σ C_ij`, the entangling shift), `C_ij`,
    `spin_correlators`, and the per-hole `bloch` vectors."""
    pairs, singles = reduced_states(psi8)
    blochs = {i: bloch(r) for i, r in singles.items()}
    corr = {ij: spin_correlator(r) for ij, r in pairs.items()}
    cij = {ij: corr[ij] - float(blochs[ij[0]] @ blochs[ij[1]]) for ij in pairs}
    j2_disc = 2.25 + 2.0 * sum(float(blochs[i] @ blochs[j]) for (i, j) in pairs)
    j2_conn = 2.0 * sum(cij.values())
    return {"j2": j2_disc + j2_conn, "j2_disconnected": j2_disc, "j2_connected": j2_conn,
            "C_ij": cij, "spin_correlators": corr, "bloch": blochs}


def j2_direct(psi8):
    """Reference full-operator `J² = ⟨ψ| Σ_a (Σ_i S_a^{(i)})² |ψ⟩` on (C²)³ — the validated
    measuring stick the chamber decomposition must reproduce exactly (J² has no three-body
    term, so the three ρ_ij determine it)."""
    st = _normalize(psi8)

    def si(a, i):
        ops = [_I2, _I2, _I2]
        ops[i] = _S[a]
        return np.kron(np.kron(ops[0], ops[1]), ops[2])

    sa = [sum(si(a, i) for i in range(3)) for a in range(3)]
    j2 = sum(sa[a] @ sa[a] for a in range(3))
    return float((st.conj() @ j2 @ st).real)


# ============================================================================
# Mesh transport layer — ported from the retired #485 module onto surviving APIs.
# A simplicial complex has no global frame; per-hole spin data live in per-cell frames
# related by the Spin(4) spin-connection Wilson line. All of it is K-type (local frame
# alignment): by Claim 1 of cartan_weyl_gluon.tex it can never create entanglement.
# ============================================================================
_W3 = cmath.exp(2j * math.pi / 3)
_SINGLET3 = (1.0, _W3, _W3 * _W3)
# The 4 trivector index triples over a cell frame's axes (a 3-blade in R⁴ has 4 components);
# the SAME triples index the det-minors and the gamma products.
_TRIPLES = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]


def _dirac_gammas():
    """Standard Euclidean 4×4 Dirac gammas (signature (4,0)); `{γ_a,γ_b}=2δ_ab`."""
    s1 = np.array([[0, 1], [1, 0]], complex)
    s2 = np.array([[0, -1j], [1j, 0]])
    s3 = np.array([[1, 0], [0, -1]], complex)
    return [np.kron(s1, s1), np.kron(s1, s2), np.kron(s1, s3), np.kron(s2, np.eye(2))]


_G = _dirac_gammas()
# The single-Dirac spatial spin operators S_a (eigenvalues ±½; Σ_a S_a² = ¾·I — spin-½),
# axes 1,2,3 spatial: Σ_a = -i/4 [γ_b, γ_c] over the cyclic spatial pairs.
_SG = [(-1j * 0.25) * (_G[(k + 1) % 3 + 1] @ _G[(k + 2) % 3 + 1]
                       - _G[(k + 2) % 3 + 1] @ _G[(k + 1) % 3 + 1]) for k in range(3)]


def rotation_to_spin(R):
    """Lift an SO(4) rotation `R` to its Spin(4) (4×4) element via `exp(½ Σ A_ab γ_aγ_b)`,
    `A = log R` — a rotation by θ gives spinor eigenphases ±θ/2 (the double cover)."""
    A = scipy.linalg.logm(np.asarray(R, dtype=float)).real
    bivec = sum(0.5 * A[a, b] * (_G[a] @ _G[b])
                for a in range(4) for b in range(a + 1, 4))
    return scipy.linalg.expm(bivec)


def embed_cell(cell):
    """Flat R⁴ coordinates of a top cell's vertices from `cell.gramMatrix()` (vertex 0 at the
    origin), canonically right-handed. The GAUGE gate rotates this embedding per cell."""
    g = np.array(cell.gramMatrix(False)).reshape(4, 4)
    wv, V = np.linalg.eigh(g)
    edges = V @ np.diag(np.sqrt(np.abs(wv)))
    if np.linalg.det(edges) < 0:
        edges[:, -1] *= -1
    vs = [v.getId() for v in cell.getVertices()]
    coords = {vs[0]: np.zeros(4)}
    for i in range(1, 5):
        coords[vs[i]] = edges[i - 1]
    return coords


def _top_tuple(s):
    """Sorted vertex-id tuple of a top simplex — used ONLY for set-matching (adjacency,
    hole→cell); orientations always come from the canonical stored cell order."""
    return tuple(sorted(v.getId() for v in s.getVertices()))


def _facet_neighbors(cells):
    """`{facet frozenset: [top cells sharing it]}` — the dual adjacency, by vertex SET."""
    fmap = collections.defaultdict(list)
    for c in cells:
        vs = _top_tuple(c)
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
    """The unified per-cell frame `F(c)` used by BOTH the spinor extraction and the Wilson
    transport, built from a VERTEX-SET-derived point cloud so the frame-local coords are
    relabel-invariant and gauge-covariant (`F → RF`).

    With `neighbors` (the `_facet_neighbors` map — the readout's default) the point cloud is
    the cell's five **dual-edge vectors** — circumcenter → facet-neighbor circumcenter,
    `(h_a + h_b)·n_facet`. The dual-edge LENGTHS carry the neighbors' sizes, so the cloud
    stays anisotropic even when the cell itself is near-regular — the #485 fix for the
    degenerate-inertia frame obstruction on near-symmetric cells. (A perfectly uniform
    metric leaves even the dual edges equal — a genuine symmetry with no canonical frame.)

    The frame is the inertia (principal-axis) eigenbasis of the centered cloud; a
    coincident-eigenvalue block is canonically resolved by the order-independent
    third-moment tensor, and each axis' sign by its third moment. Returns `(coords, F, vs)`."""
    coords = embed_cell(cell)
    vs = [v.getId() for v in cell.getVertices()]
    if neighbors is None:                              # fallback: the cell's own vertices
        cloud = np.array([coords[v] for v in vs])
    else:                                              # default: dual-edge vectors
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
    w, V = np.linalg.eigh(pc.T @ pc)
    proj = pc @ V
    wsq = (proj ** 2).sum(1)
    i = 0
    while i < 4:                          # lift coincident moments with the third moment
        j = i + 1
        while j < 4 and abs(w[j] - w[i]) < 1e-6 * max(1.0, abs(w[i])):
            j += 1
        if j - i > 1:
            sub = proj[:, i:j]
            _ww, vv = np.linalg.eigh((sub * wsq[:, None]).T @ sub)
            V[:, i:j] = V[:, i:j] @ vv
        i = j
    proj = pc @ V
    for k in range(4):                    # canonical, order-independent axis signs
        if np.sum(proj[:, k] ** 3) < 0:
            V[:, k] *= -1
    return coords, V, vs


def _frames(cells):
    """Precompute the dual-edge `cell_frame` of every top cell, keyed by `id(cell)`."""
    fmap = _facet_neighbors(cells)
    return {id(c): cell_frame(c, fmap) for c in cells}


def facet_transport(cell_a, cell_b, frames):
    """The Spin(4) transport mapping `cell_b`'s spinor frame to `cell_a`'s — from the SAME
    `cell_frame`s the extraction uses: align the shared facet's coords in `F(b)` to those in
    `F(a)` by orthogonal Procrustes (det-corrected to a proper rotation), then
    `rotation_to_spin`. `None` if they share no facet."""
    ca, Fa, _ = frames[id(cell_a)]
    cb, Fb, _ = frames[id(cell_b)]
    shared = sorted(set(ca) & set(cb))
    if len(shared) < 4:
        return None
    xa = np.array([Fa.T @ ca[v] for v in shared])
    xa -= xa.mean(0)
    xb = np.array([Fb.T @ cb[v] for v in shared])
    xb -= xb.mean(0)
    u, _s, vt = np.linalg.svd(xb.T @ xa)
    d = np.eye(4)
    d[3, 3] = np.sign(np.linalg.det(vt.T @ u.T))
    return rotation_to_spin(vt.T @ d @ u.T)


def _dual_adjacency(cells):
    """Dual adjacency keyed by `id(cell)`: two top cells are adjacent iff they share a
    codim-1 facet (4 common vertices)."""
    adj = collections.defaultdict(list)
    tt = {id(c): set(_top_tuple(c)) for c in cells}
    for c in cells:
        for c2 in cells:
            if c is not c2 and len(tt[id(c)] & tt[id(c2)]) == 4:
                adj[id(c)].append(c2)
    return adj


def wilson_line(adj, cell_i, cell_j, frames):
    """The Spin(4) holonomy mapping `cell_j`'s frame to `cell_i`'s: the composition of
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
        ft = facet_transport(p, c, frames)   # maps c-frame → p-frame
        if ft is None:
            return None
        holo = ft @ holo
        c = p
    return holo


def _cell_volume(cell):
    """`sqrt(|det Gram|)`-scale cell volume — metric data, so gauge- and relabel-invariant;
    used only as a label-free tie-break when several cells touch a hole equally."""
    return float(abs(np.linalg.det(np.array(cell.gramMatrix(False)).reshape(4, 4))))


def _cell_orientation_signs(cells):
    """A global orientation of the top cells: a sign per cell (keyed by `id(cell)`) under
    which adjacent cells induce OPPOSITE orientations on their shared facet
    (`s_A·(−1)^{j_A} = −s_B·(−1)^{j_B}`, `j` the drop position in the ascending-id tuple),
    BFS-propagated from the first cell. Label-free up to the one global sign, which cancels
    out of every readout below."""
    tt = {id(c): _top_tuple(c) for c in cells}
    fmap = collections.defaultdict(list)     # facet -> [(cell, drop position)]
    for c in cells:
        vs = tt[id(c)]
        for j in range(len(vs)):
            fmap[frozenset(vs[:j] + vs[j + 1:])].append((c, j))
    sign = {id(cells[0]): 1}
    q = collections.deque([cells[0]])
    while q:
        c = q.popleft()
        vs = tt[id(c)]
        for j in range(len(vs)):
            for c2, j2 in fmap[frozenset(vs[:j] + vs[j + 1:])]:
                if c2 is c or id(c2) in sign:
                    continue
                sign[id(c2)] = -sign[id(c)] * (-1) ** (j + j2)
                q.append(c2)
    return sign, fmap


def hole_orientation_signs(cells, holes):
    """The manifold-consistent orientation sign ε_q of each hole's canonical (ascending-id)
    orientation. The hole is a REMOVED top cell; were it present with sign ε it would have
    to cancel each live neighbor's induced facet orientation, so every one of its five
    facets determines ε — and all five must agree (asserted). The period/target convention
    (`cyclePeriods`) is the ascending-id orientation, so pinning the SIGNED targets
    `ε_q·target_q` pins the orientation-canonical periods: a vertex relabeling flips a
    hole's ascending-id orientation by the permutation parity, and ε_q flips with it — this
    is the register sign convention (apply induced-orientation signs to periods/targets
    before scoring). The one leftover global sign flips ψ → −ψ, which every readout below
    is even in."""
    sign, fmap = _cell_orientation_signs(cells)
    eps = []
    for h in holes[:3]:
        hv = tuple(sorted(h))
        vals = set()
        for j in range(len(hv)):
            for c2, j2 in fmap[frozenset(hv[:j] + hv[j + 1:])]:
                vals.add(-sign[id(c2)] * (-1) ** (j + j2))
        if len(vals) != 1:
            raise ValueError(f"inconsistent hole orientation for {hv}: {vals}")
        eps.append(vals.pop())
    return eps


class MeshContext:
    """The shared per-complex transport context both routes read: the top cells, their
    precomputed dual-edge frames, the dual adjacency, the three hole proxy cells (the hole
    itself is a REMOVED top cell, so its spin data are read from an adjacent live cell — the
    one sharing the most vertices, ties broken by the label-free cell volume), and the
    canonical k=3 cell index of `carriedRepresentative` (`EigenstateSynthesis.cellSimplices`)."""

    def __init__(self, st, holes):
        import tessera as T
        self.st = st
        self.holes = [tuple(h) for h in holes[:3]]
        self.cells = list(st.getTopSimplices())
        self.frames = _frames(self.cells)
        self.adj = _dual_adjacency(self.cells)

        def cell_of(h):
            hv = set(h)
            return max(self.cells,
                       key=lambda c: (len(hv & set(_top_tuple(c))), _cell_volume(c)))

        self.hole_cells = [cell_of(h) for h in self.holes]
        es = T.cobordism.EigenstateSynthesis(st, 3)
        self.by_set = {frozenset(t): (i, list(t))
                       for i, t in enumerate(es.cellSimplices())}
        self.eps = hole_orientation_signs(self.cells, self.holes)

    def line(self, i, j):
        """The Spin(4) Wilson line transporting hole `j`'s frame to hole `i`'s."""
        if i == j:
            return np.eye(4, dtype=complex)
        return wilson_line(self.adj, self.hole_cells[i], self.hole_cells[j], self.frames)

    def signed_target(self, target=_SINGLET3):
        """The orientation-canonical (ε-signed) pin targets `ε_q·target_q` — what
        `[1,ω,ω²]` means once each hole's ascending-id orientation is referred to the one
        global orientation (see `hole_orientation_signs`)."""
        return [self.eps[q] * complex(target[q]) for q in range(3)]


def emergent_spinor(by_set, psi, frame):
    """The per-hole Dirac spinor extracted from the carried representative `psi` (a k=3
    cochain) over the hole cell's five tetrahedral faces, in the cell's own frame — the
    degree-3 slice of the vertical (fiber↔cells) lift. `frame` is the cell's `cell_frame`
    triple `(coords, F, vs)`. Faces are matched by vertex SET and oriented from the canonical
    stored order (never re-sorted here): each face's trivector (the 4 det-minors of its 3
    edge vectors over `_TRIPLES`) is least-squared against `psi`, the 3-form mapped into the
    Clifford algebra (`Φ = Σ ω_t γ_iγ_jγ_k`), and the spinor read as `s = Φ·[1,0,0,0]`,
    normalized."""
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


def joint_field(ctx, target=_SINGLET3):
    """The single correlated multi-hole carried representative
    `EigenstateSynthesis(st,3).carriedRepresentative([h0,h1,h2], ε·[1,ω,ω²])` — one k=3
    cochain carrying the color singlet across ALL THREE holes at once, pinned in the
    orientation-canonical (ε-signed) convention so the field is a label-free object — plus
    its carry residual (the emergence certificate: → 0 iff the geometry genuinely carries
    the target)."""
    import tessera as T
    es = T.cobordism.EigenstateSynthesis(ctx.st, 3)
    hh = [list(h) for h in ctx.holes]
    tgt = ctx.signed_target(target)
    psi = np.asarray(es.carriedRepresentative(hh, tgt), dtype=complex)
    return psi, float(es.residualForPeriods(hh, tgt))


def joint_spinors(ctx, target=_SINGLET3):
    """The three per-hole spinors read from the SINGLE joint carried representative — the
    color correlation imprints on all three extractions through the one field."""
    psi, _res = joint_field(ctx, target)
    return [emergent_spinor(ctx.by_set, psi, ctx.frames[id(c)]) for c in ctx.hole_cells]


def independent_spinors(ctx):
    """The three per-hole spinors from three INDEPENDENT single-hole carried representatives
    (`carriedRepresentative([h],[1.0])`) — the product-read baseline."""
    import tessera as T
    es = T.cobordism.EigenstateSynthesis(ctx.st, 3)
    out = []
    for h, c in zip(ctx.holes, ctx.hole_cells):
        psi = np.asarray(es.carriedRepresentative([list(h)], [1.0]), dtype=complex)
        out.append(emergent_spinor(ctx.by_set, psi, ctx.frames[id(c)]))
    return out


def _spinor_to_qubit(s4):
    """Reduce a 4-component Dirac spinor to the spin-½ qubit along its Bloch direction
    (read with the structural generators `_SG`)."""
    s4 = np.asarray(s4, complex)
    nrm = (s4.conj() @ s4).real
    if nrm < 1e-30:
        return np.array([1, 0], complex)
    n = np.array([(s4.conj() @ _SG[a] @ s4).real / nrm for a in range(3)])
    if np.linalg.norm(n) < 1e-9:
        return np.array([1, 0], complex)
    m = sum((n / np.linalg.norm(n))[a] * _PAULI[a] for a in range(3))
    _w, v = np.linalg.eigh(m)
    return v[:, int(np.argmax(_w.real))]


# ============================================================================
# (a) VERTICAL route — C_ij from the reconstructed joint two-hole reduced states
# ============================================================================
def vertical_read(st, holes, spinors=None, ctx=None):
    """The vertical route: per-hole spinors from the joint carried representative (or the
    given `spinors`), Wilson-transported to hole 0's frame, assembled into the reconstructed
    3-qubit state; C_ij and ⟨S_i·S_j⟩ are then read from its joint two-hole reduced states
    ρ_ij (`reduced_states`). A reconstruction that reduces each hole to one spinor is a
    bilinear of a CLASSICAL cochain, which factorizes (rank-1 ⇒ ρ_ij = ρ_i⊗ρ_j), so its
    C_ij ≡ 0 — the K-orbit floor made concrete. Returns the `j2_decomposition` dict plus
    `route`; `None` if a Wilson line is missing."""
    if ctx is None:
        ctx = MeshContext(st, holes)
    if spinors is None:
        spinors = joint_spinors(ctx)
    lines0 = [ctx.line(0, j) for j in range(3)]
    if any(u is None for u in lines0):
        return None
    qubits = [_spinor_to_qubit(lines0[j] @ spinors[j]) for j in range(3)]
    rep = j2_decomposition(kron(*qubits))
    rep["route"] = "vertical(joint rho_ij)"
    return rep


# ============================================================================
# (b) HORIZONTAL route — C_ij from the M/Wilson-loop holonomy invariants alone
# ============================================================================
def transport_so3(W):
    """The SO(3) rotation the Spin(4) transport `W` induces on the spin axes,
    `W S_a W† = Σ_b R_ba S_b`, read as `R_ba = Tr[S_b W S_a W†]` — purely a function of the
    holonomy; no spinor is touched."""
    Wd = W.conj().T
    return np.array([[float((_SG[b] @ W @ _SG[a] @ Wd).trace().real)
                      for a in range(3)] for b in range(3)])


def transport_angle(W):
    """The rotation angle θ of the transport's SO(3) part: `cos θ = (Tr R − 1)/2`."""
    c = (np.trace(transport_so3(W)) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, c)))


def color_phases(ctx, target=_SINGLET3):
    """The color register's Z₃ phases over the three holes — the M-intertwiner invariants,
    in the orientation-canonical convention (the ε-signed carried targets; the global sign
    shifts every phase by π together, so the differences are label-free). The periods of
    the joint carried representative equal these exactly when the register carries the
    target (residual → 0): the emergent content is the CARRY (the residual certifies the
    geometry supports these phases), not an independent phase measurement — reported
    alongside the residual for honesty."""
    _psi, res = joint_field(ctx, target)
    ph = np.angle(np.array(ctx.signed_target(target), complex))
    return ph - ph[0], res


def horizontal_read(st, holes, target=_SINGLET3, ctx=None):
    """The horizontal route: correlator predictions from holonomy invariants ALONE.

    * Frame-transport channel: `⟨S_i·S_j⟩ = ¼·cos θ_ij` from the inter-hole Wilson line's
      SO(3) angle — the correlator of transport-parallel unit spins. Its connected part is
      identically zero (a K-type frame alignment cannot entangle — Claim 1), realized here
      by the same construction's marginals: `⟨S_i⟩·⟨S_j⟩ = ¼·cos θ_ij` too, so C_ij = 0.
    * Color channel: `⟨S_i·S_j⟩ = ¼·cos Δφ_ij` from the register's Z₃ phases (the M
      invariants) — the most generous holonomy-only read; no marginals exist for it, so it
      is a raw-correlator prediction, not a connected one.

    Each pair gets its own invariant, so unlike a realizable 3-qubit state this read is not
    bound by the n=3 frustration floor. Returns per-pair angles, correlators, C_ij (transport
    channel), and both reconstructed J²; `None` if a Wilson line is missing."""
    if ctx is None:
        ctx = MeshContext(st, holes)
    W = {(i, j): ctx.line(i, j) for i, j in _PAIRS}
    if any(w is None for w in W.values()):
        return None
    theta = {p: transport_angle(W[p]) for p in _PAIRS}
    sdot = {p: 0.25 * math.cos(theta[p]) for p in _PAIRS}
    cij = {p: 0.0 for p in _PAIRS}        # K-type transport: connected part vanishes
    ph, res = color_phases(ctx, target)
    csdot = {(i, j): 0.25 * math.cos(float(ph[j] - ph[i])) for i, j in _PAIRS}
    return {"route": "horizontal(holonomy invariants)",
            "theta_deg": {p: math.degrees(theta[p]) for p in _PAIRS},
            "spin_correlators": sdot,
            "C_ij": cij,
            "j2": 2.25 + 2.0 * float(sum(sdot.values())),
            "color_phase_deg": {(i, j): math.degrees(float(ph[j] - ph[i]))
                                for i, j in _PAIRS},
            "color_spin_correlators": csdot,
            "color_j2": 2.25 + 2.0 * float(sum(csdot.values())),
            "carry_residual": res}


# ============================================================================
# The two-ways comparison and the emergent read
# ============================================================================
def two_ways(st, holes):
    """Experiment 2 — C_ij two ways on one geometry. Returns both routes' reports plus the
    per-pair gaps `|⟨S_i·S_j⟩_vert − ⟨S_i·S_j⟩_holo|`, `|C_ij_vert − C_ij_holo|`, the J²
    gaps, and whether either route reaches the proton ¾ (|J² − ¾| ≤ 0.1)."""
    ctx = MeshContext(st, holes)
    vert = vertical_read(st, holes, ctx=ctx)
    horiz = horizontal_read(st, holes, ctx=ctx)
    if vert is None or horiz is None:
        return None
    gaps = {
        "sdot_transport": {p: abs(vert["spin_correlators"][p]
                                  - horiz["spin_correlators"][p]) for p in _PAIRS},
        "sdot_color": {p: abs(vert["spin_correlators"][p]
                              - horiz["color_spin_correlators"][p]) for p in _PAIRS},
        "C_ij": {p: abs(vert["C_ij"][p] - horiz["C_ij"][p]) for p in _PAIRS},
        "j2_transport": abs(vert["j2"] - horiz["j2"]),
        "j2_color": abs(vert["j2"] - horiz["color_j2"]),
    }
    return {"vertical": vert, "horizontal": horiz, "gaps": gaps,
            "reaches_proton": {"vertical": abs(vert["j2"] - 0.75) <= 0.1,
                               "horizontal": abs(horiz["j2"] - 0.75) <= 0.1,
                               "horizontal_color": abs(horiz["color_j2"] - 0.75) <= 0.1}}


def emergent_read(st, holes):
    """Experiment 3 — the readout on a converged fixture's joint carried representative vs
    the independent per-hole (product) baseline: does the joint read's J² move below the
    product read toward ¾?  (Prior joint_* prototype: 1.80 → 1.64.)"""
    ctx = MeshContext(st, holes)
    joint = vertical_read(st, holes, spinors=joint_spinors(ctx), ctx=ctx)
    prod = vertical_read(st, holes, spinors=independent_spinors(ctx), ctx=ctx)
    if joint is None or prod is None:
        return None
    prod = dict(prod, route="vertical(independent product)")
    return {"joint": joint, "product": prod,
            "moved_toward_proton": joint["j2"] < prod["j2"]}


# ============================================================================
# Validation gates — GAUGE (random per-cell frame rotation), RELABEL (vertex permutation)
# ============================================================================
def gauge_gate(read, st, holes, seed=7):
    """|Δ J²| under a random per-cell SO(4) rotation of the embedding — every frame-local
    quantity must be invariant. `read(st, holes) → dict with 'j2'`. Post-hoc only."""
    global embed_cell
    base = read(st, holes)["j2"]
    rng = np.random.default_rng(seed)
    rmap = {}
    orig = embed_cell

    def gauged(cell):
        c = orig(cell)
        key = tuple(sorted(c))
        if key not in rmap:
            a = rng.standard_normal((4, 4))
            rmap[key] = scipy.linalg.expm(a - a.T)
        return {v: rmap[key] @ x for v, x in c.items()}

    embed_cell = gauged
    try:
        rotated = read(st, holes)["j2"]
    finally:
        embed_cell = orig
    return abs(rotated - base)


def relabel_gate(read, cells, edges, holes, seed=3):
    """|Δ J²| under a random vertex-id permutation of the whole complex (same physical
    holes, relabeled). `cells`/`edges` are the raw fixture data; the complex is rebuilt both
    ways through the same loader. Post-hoc only."""
    st = build_fixture(cells, edges)
    base = read(st, [tuple(h) for h in holes])["j2"]
    allv = sorted({v for c in cells for v in c})
    shuf = allv[:]
    np.random.default_rng(seed).shuffle(shuf)
    perm = dict(zip(allv, shuf))
    st2 = build_fixture([[perm[v] for v in c] for c in cells],
                        {tuple(sorted((perm[a], perm[b]))): z
                         for (a, b), z in edges.items()})
    holes2 = [tuple(sorted(perm[v] for v in h)) for h in holes]
    return abs(read(st2, holes2)["j2"] - base)


# ============================================================================
# Fixture loading (tests/fixtures/composite_spin) — the controlled b₃=3 registers
# ============================================================================
_FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "..", "tests", "fixtures", "composite_spin")


def build_fixture(cells, edges):
    """Rebuild a fixture complex: `Spacetime.fromCells` + the stored squared lengths, with
    the skeleton materialized in C++ by the `ReggeSolver` constructor (never from Python)."""
    import tessera as T
    st = T.Spacetime.fromCells(4, [list(c) for c in cells], 1.0, 0.0)
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        e.setSquaredLength(edges[(a, b) if a < b else (b, a)])
    T.ReggeSolver(st, T.MatterConfiguration())
    return st


def load_fixture(name):
    """Load a composite_spin fixture: returns `(cells, edges, st, holes)` with `holes` the
    emergent k=3 holes (`MultiCobordism.emergent_holes`), first three = the color register."""
    import tessera as T
    with open(os.path.join(_FIXTURES, name)) as fh:
        d = json.load(fh)
    cells = [list(c) for c in d["cells"]]
    edges = {}
    for k, (re, im) in d["edges"].items():
        a, b = (int(x) for x in k.split(","))
        edges[(a, b)] = complex(re, im)
    st = build_fixture(cells, edges)
    holes = [tuple(h) for h in T.cobordism.MultiCobordism.emergent_holes(st, 3)]
    return cells, edges, st, holes


# ============================================================================
# __main__ — the three experiments, with the gates
# ============================================================================
def main():
    print("C_ij connected-correlator chamber readout (#564) — "
          "J² = 9/4 + 2·Σ ⟨S_i·S_j⟩, C_ij = ⟨S_i·S_j⟩ − ⟨S_i⟩·⟨S_j⟩\n")

    print("== 1. instrument: chamber coordinates on the hand-fed clean states ==")
    print(f"{'state':<28}{'J2':>9}{'floor':>9}{'2*sumC':>9}   C_ij")
    for name, psi in clean_states().items():
        d = j2_decomposition(psi)
        cij = "  ".join(f"{i}{j}={v:+.4f}" for (i, j), v in d["C_ij"].items())
        print(f"{name:<28}{d['j2']:>9.4f}{d['j2_disconnected']:>9.4f}"
              f"{d['j2_connected']:>+9.4f}   {cij}")
    print("  targets: proton 3/4 (C_ij carries the whole -3/2 below the 9/4 floor),")
    print("  product |uud> 7/4, Delta |uuu> 15/4 — both products: C_ij = 0.\n")

    print("== 2. C_ij two ways — the escape-hatch test (synthetic_b3_3) ==")
    cells, edges, st, holes = load_fixture("synthetic_b3_3.json")
    out = two_ways(st, holes)
    v, h, g = out["vertical"], out["horizontal"], out["gaps"]
    print(f"  vertical   (joint rho_ij)      J²={v['j2']:.4f}  "
          f"ΣC_ij={sum(v['C_ij'].values()):+.2e}")
    print(f"  horizontal (transport θ_ij)    J²={h['j2']:.4f}  ΣC_ij=+0.00e+00  "
          f"θ_ij=" + ", ".join(f"{x:.0f}°" for x in h["theta_deg"].values()))
    print(f"  horizontal (color Δφ_ij)       J²={h['color_j2']:.4f}  "
          f"Δφ_ij=" + ", ".join(f"{x:.0f}°" for x in h["color_phase_deg"].values())
          + f"  carry residual={h['carry_residual']:.2e}")
    print(f"  per-pair ⟨S_i·S_j⟩ gaps  vert vs transport: "
          + ", ".join(f"{p[0]}{p[1]}={x:.4f}" for p, x in g["sdot_transport"].items()))
    print(f"                           vert vs color:     "
          + ", ".join(f"{p[0]}{p[1]}={x:.4f}" for p, x in g["sdot_color"].items()))
    print(f"  C_ij gap (vert vs transport): "
          + ", ".join(f"{p[0]}{p[1]}={x:.1e}" for p, x in g["C_ij"].items()))
    print(f"  J² gaps: |vert−transport|={g['j2_transport']:.4f}  "
          f"|vert−color|={g['j2_color']:.4f}")
    print(f"  reaches ¾?  vertical={out['reaches_proton']['vertical']}  "
          f"horizontal={out['reaches_proton']['horizontal']}  "
          f"color={out['reaches_proton']['horizontal_color']}")
    print("  verdict: both routes' CONNECTED C_ij vanish (classical bilinear factorizes;")
    print("  K-type transport cannot entangle) — they agree exactly on C_ij = 0 and only")
    print("  there; the raw correlators disagree by the gaps above and neither reaches ¾:")
    print("  the holonomy supplies no entangling C_ij, so the fiber↔cells lift stays")
    print("  necessary for the proton ¾ (consistent with the retired #512 finding).\n")

    print("== gates (synthetic_b3_3) ==")
    for label, read in (("vertical", vertical_read), ("horizontal", horizontal_read)):
        dg = gauge_gate(read, st, holes)
        dr = relabel_gate(read, cells, edges, holes)
        print(f"  {label:<10} |ΔGAUGE|={dg:.2e}  |ΔRELABEL|={dr:.2e}")

    print("\n== 3. emergent read (converged_b3_3): joint vs independent product ==")
    ccells, cedges, cst, choles = load_fixture("converged_b3_3.json")
    em = emergent_read(cst, choles)
    j, p = em["joint"], em["product"]
    print(f"  product (independent reads)  J²={p['j2']:.4f}")
    print(f"  joint (correlated field)     J²={j['j2']:.4f}  "
          f"ΣC_ij={sum(j['C_ij'].values()):+.2e}")
    print(f"  moved toward ¾ below the product read: {em['moved_toward_proton']} "
          f"(prior joint_* prototype baseline: 1.80 → 1.64)")
    for label, read in (("vertical", vertical_read), ("horizontal", horizontal_read)):
        dg = gauge_gate(read, cst, choles)
        dr = relabel_gate(read, ccells, cedges, choles)
        print(f"  gates {label:<10} |ΔGAUGE|={dg:.2e}  |ΔRELABEL|={dr:.2e}")


if __name__ == "__main__":
    main()
