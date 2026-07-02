# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""C_ij connected-correlator chamber readout — instrument + the two-route comparison (#564).

The composite total spin of three spin-½ holes obeys (spin_readout.tex §3)

    J² = 9/4 + 2·Σ_{i<j} ⟨S_i·S_j⟩,      ⟨S_i·S_j⟩ = ⟨S_i⟩·⟨S_j⟩ + C_ij,

where the **connected** correlator `C_ij ≡ ⟨S_i·S_j⟩ − ⟨S_i⟩·⟨S_j⟩` is the Cartan-chamber
coordinate of the pair (cartan_weyl_gluon.tex §5a): it vanishes identically on any product
state (the K-orbit floor — local frames alone cannot entangle, Claim 1 there), and on the
entangled proton eigenstate it carries the whole shift below the floor (Σ C_ij = −¾ ⇒ J² = ¾).
Three experiments (cartan_weyl_gluon.tex §7, items 1–3):

1. **Instrument** (`j2_decomposition`, pure NumPy on (C²)³): J² in chamber (correlator)
   coordinates must return EXACTLY ¾ / 7/4 / 15/4 on the hand-fed clean proton
   `2|uud⟩−|udu⟩−|duu⟩` / product `|uud⟩` / Δ `|uuu⟩` states, and C_ij must vanish
   identically on every product state. Regression-tested to 1e-12.

2. **Two routes on one fixture** (`two_ways`) — what each route MEASURES, honestly scoped:

   (a) VERTICAL (`vertical_read`) — the per-hole spinors are extracted from the SINGLE
   correlated multi-hole carried representative and Wilson-transported to a common frame,
   but the only 3-qubit state reconstructable from a classical (single-particle) cochain is
   their PRODUCT: any bilinear pair read of ψ factorizes (rank-1 ⇒ ρ_ij = ρ_i⊗ρ_j), so the
   reconstruction is **separable by construction and its C_ij ≡ 0 is a structural
   consequence, not a field measurement**. The route measures the separable correlators
   `⟨S_i·S_j⟩ = ⟨S_i⟩·⟨S_j⟩` of the field's extracted spin directions, and their J².

   (b) HORIZONTAL (`horizontal_read`) — the inter-hole Wilson lines' SO(3) angles θ_ij
   (genuine, frame-free holonomy measurements) mapped to transport-parallel correlators
   `¼·cos θ_ij` and their J². A K-type transport cannot entangle and no correlated
   transport construction exists in the current machinery, so **this route yields no C_ij
   measurement either**. The register's color phases are NOT measured holonomy on these
   fixtures — `carriedRepresentative` pins ψ's periods to the caller's targets by
   construction — so they appear only as the pinned-input reference (`color_reference`),
   never in a measurement comparison.

   **What the comparison can and cannot say:** it CANNOT decide the escape-hatch question
   (whether C_ij is a holonomy invariant) — neither route measures C_ij with current
   machinery. It DOES measure the ¾-exclusion (both routes' J² stay far from the entangled
   proton value on the fixture) and the per-pair gap between the two separable correlator
   sets (the field's extracted spin directions vs the transport-parallel directions). The
   entangling C_ij remains readable only through a genuine two-hole joint lift — the
   fiber↔cells kernel (#495); the retired #512 run reached the same conclusion on the
   pre-#509 machinery.

3. **Emergent read** (`emergent_read`): the joint-field read vs the independent per-hole
   (product) read on a converged fixture — does J² move below the product read toward ¾?
   (The prior joint_* prototype saw 1.80 → 1.64; both reads are separable reconstructions,
   so any movement is color imprinting on the extracted spin directions, not entanglement.)

Validation gates (`gauge_gate`, `relabel_gate`): every reported numeric channel — per-pair
quantities included, not just J² — must be invariant under a random per-cell SO(4) rotation
of the embedding (GAUGE, threaded explicitly through `MeshContext(gauge=...)`, never
monkey-patched) and under a vertex relabeling + cell-order shuffle with the register
re-derived on the relabeled complex (RELABEL). Run post-hoc, never as a loop condition.
Passing RELABEL requires pinning the joint field in the orientation-canonical convention:
each hole's ascending-vertex-id orientation is referred to the complex's own coherent
orientation via `ChainComplex.endSignCovector` (`induced_orientation_signs`), and the
ε-signed targets are pinned — the register sign convention. The leftover global sign flips
ψ → −ψ, which every readout is even in.

Register selection (`register_holes`): the color register is the first three emergent holes
in `MultiCobordism.emergent_holes` order — the #485 fixture convention — with a deficit an
error and any surplus an explicit, warned choice (never a silent slice).

Prior art, credited: the retired pre-#509 escape-hatch experiment (#512, branch
`feat/cij-escape-hatch`); the pairwise-C_ij instrument core proven on branch
`feat/cij-composite-spin` (#514, PR #519, unmerged — its field-sourced Werner extraction is
the constructive sibling this module deliberately does not duplicate); PR #518 (unmerged)
carries a C++ loops-as-quarks J² that also floors above ¾; the `register_holes` /
`endSignCovector` patterns mirror the sibling pair-loop flavor readout (#561).

Pure NumPy for the instrument; `tessera` is imported lazily by the mesh layer only.
"""
import cmath
import collections
import json
import math
import os
import warnings

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
    matrices of a three-qubit pure state (ordering hole0 ⊗ hole1 ⊗ hole2). Returns
    `(pairs, singles)` keyed by `(i, j)` / `i`; every reduced state is unit-trace."""
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
    pair is uncorrelated (a product); it is exactly what a per-hole Bloch/product read
    discards."""
    return spin_correlator(rho_ij) - float(bloch(rho_i) @ bloch(rho_j))


def j2_decomposition(psi8):
    """The chamber-coordinate breakdown of `J² = 9/4 + 2·Σ_{i<j} ⟨S_i·S_j⟩` for a 3-qubit
    pure state, through its joint two-hole reduced states. Returns `j2`
    (= `j2_disconnected + j2_connected`), `j2_disconnected` (the per-hole Bloch floor
    `9/4 + 2Σ⟨S_i⟩·⟨S_j⟩`), `j2_connected` (`2·Σ C_ij`, the entangling shift), `C_ij`
    (via `connected_correlator`), `spin_correlators`, and the per-hole `bloch` vectors."""
    pairs, singles = reduced_states(psi8)
    blochs = {i: bloch(r) for i, r in singles.items()}
    corr = {ij: spin_correlator(r) for ij, r in pairs.items()}
    cij = {(i, j): connected_correlator(pairs[(i, j)], singles[i], singles[j])
           for (i, j) in pairs}
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
    """Flat R⁴ coordinates of a top cell's vertices from `cell.gramMatrix()` (vertex 0 at
    the origin), canonically right-handed. `{vertex id: R⁴ point}`."""
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


def _dual_height(cell, coords):
    """`{facet frozenset: |distance from this cell's circumcenter to that facet|}` — the
    intrinsic dual half-edge each facet contributes, in the given embedding `coords`."""
    vs = [v.getId() for v in cell.getVertices()]
    cc = _circumcenter(cell, coords, vs)
    out = {}
    for j in range(5):
        facet = frozenset(vs[:j] + vs[j + 1:])
        cen, n = _facet_outward_normal(coords, sorted(facet), vs[j])
        out[facet] = abs(float((cc - cen) @ n))
    return out


def cell_frame(cell, coords, neighbors, heights):
    """The unified per-cell frame `F(c)` used by BOTH the spinor extraction and the Wilson
    transport, built from a VERTEX-SET-derived point cloud so the frame-local coords are
    relabel-invariant and gauge-covariant (`F → RF`).

    The point cloud is the cell's five **dual-edge vectors** — circumcenter → facet-neighbor
    circumcenter, `(h_a + h_b)·n_facet`, with `heights` the precomputed `_dual_height` maps
    keyed by `id(cell)`. The dual-edge LENGTHS carry the neighbors' sizes, so the cloud
    stays anisotropic even when the cell itself is near-regular — the #485 fix for the
    degenerate-inertia frame obstruction on near-symmetric cells. (A perfectly uniform
    metric leaves even the dual edges equal — a genuine symmetry with no canonical frame.)

    The frame is the inertia (principal-axis) eigenbasis of the centered cloud; a
    coincident-eigenvalue block is canonically resolved by the order-independent
    third-moment tensor, and each axis' sign by its third moment. Returns `(coords, F, vs)`."""
    vs = [v.getId() for v in cell.getVertices()]
    vset = tuple(sorted(vs))
    h_self = heights[id(cell)]
    rows = []
    for j in range(5):
        facet = frozenset(vs[:j] + vs[j + 1:])
        cen, n = _facet_outward_normal(coords, sorted(facet), vs[j])
        h_b = 0.0
        for nb in neighbors.get(facet, ()):
            if tuple(sorted(x.getId() for x in nb.getVertices())) != vset:
                h_b = heights[id(nb)][facet]
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


def _frames(cells, gauge=None):
    """Precompute the dual-edge `cell_frame` of every top cell, keyed by `id(cell)`. Each
    cell is embedded ONCE (the coords are shared by `_dual_height` and `cell_frame`).
    `gauge`, when given, is a callable `vertex-set tuple → SO(4) matrix` applied to each
    cell's embedding — the explicit hook the GAUGE gate threads through `MeshContext`
    instead of patching module state."""
    fmap = _facet_neighbors(cells)
    coords = {}
    for c in cells:
        emb = embed_cell(c)
        if gauge is not None:
            rot = gauge(tuple(sorted(emb)))
            emb = {v: rot @ x for v, x in emb.items()}
        coords[id(c)] = emb
    heights = {id(c): _dual_height(c, coords[id(c)]) for c in cells}
    return {id(c): cell_frame(c, coords[id(c)], fmap, heights) for c in cells}


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


def wilson_line(adj, cell_i, cell_j, frames):
    """The Spin(4) holonomy mapping `cell_j`'s frame to `cell_i`'s: the composition of
    `facet_transport` along a BFS dual path `j → … → i`. `None` if disconnected.

    NB the BFS tie-breaks between equal-length dual paths follow the adjacency enumeration
    order, and on a curved complex distinct paths differ by the enclosed holonomy — a
    path-dependence intrinsic to any open-line transport. On these small fixtures the
    gates verify the reads at ~1e-14 (the RELABEL gate shuffles the cell enumeration too);
    a canonical path rule (or loop-averaged transport) is needed before this reads larger
    emergent specimens."""
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
    """`|det Gram|`-scale cell volume — metric data, so gauge- and relabel-invariant;
    used only as a label-free tie-break when several cells touch a hole equally."""
    return float(abs(np.linalg.det(np.array(cell.gramMatrix(False)).reshape(4, 4))))


# ============================================================================
# Register selection + orientation — the single validated entry point
# ============================================================================
def register_holes(st, count=3):
    """The structure's `count` register holes (removed top 4-cells) in
    `MultiCobordism.emergent_holes` order — the #485 fixture convention (mirrors the
    sibling pair-loop readout's selection). Raises on a deficit; a surplus is an explicit,
    warned truncation, never a silent slice. Returns `(holes, dropped)`."""
    import tessera as T
    holes = [tuple(h) for h in T.cobordism.MultiCobordism.emergent_holes(st, 3)]
    if len(holes) < count:
        raise ValueError(f"need >= {count} register holes, found {len(holes)}")
    if len(holes) > count:
        warnings.warn(
            f"register selection: {len(holes)} emergent holes; using the first {count} "
            f"(emergent_holes order), dropping {holes[count:]} — the confinement "
            f"constraint ranges over ALL holes, so the read covers a sub-register",
            stacklevel=2)
    return holes[:count], holes[count:]


def induced_orientation_signs(st, holes):
    """The induced-orientation signs `ε_h = ±1` of the holes' boundary cycles relative to
    the complex's own coherent orientation (`ChainComplex.endSignCovector` — lex-rooted,
    component-aware, and a hard error on any facet with >2 cofaces or a non-orientable
    surface): the label-free orientation under which every closed form's signed periods
    obey `Σ_h ε_h p_h = 0`. The period/target convention (`cyclePeriods`) is the
    ascending-vertex-id orientation, which flips with relabeling parity; pinning the
    SIGNED targets `ε_h·target_h` therefore pins the orientation-canonical periods — the
    register sign convention. Determined up to one global sign (the propagation root),
    which flips ψ → −ψ; every readout here is even in it."""
    import tessera as T
    tops = [[v.getId() for v in s.getVertices()] for s in st.getTopSimplices()]
    return list(T.cobordism.ChainComplex.endSignCovector(tops, [list(h) for h in holes]))


class MeshContext:
    """The shared per-complex transport context both routes read: the validated register
    (`register_holes` when `holes` is not given; a supplied list is validated the same
    way), the top cells, their precomputed dual-edge frames (`gauge` threads the GAUGE
    gate's per-cell rotations through explicitly), the C++ dual adjacency
    (`Spacetime.getDualAdjacency`), the three hole proxy cells (the hole itself is a
    REMOVED top cell, so its spin data are read from an adjacent live cell — the one
    sharing the most vertices, ties broken by the label-free cell volume), one cached
    `EigenstateSynthesis` (its `cellSimplices` is the canonical k=3 index of
    `carriedRepresentative`), the holes' induced-orientation signs ε, and a memoized
    joint field."""

    def __init__(self, st, holes=None, gauge=None):
        import tessera as T
        self.st = st
        if holes is None:
            self.holes, self.dropped = register_holes(st)
        else:
            holes = [tuple(h) for h in holes]
            if len(holes) < 3:
                raise ValueError(f"need >= 3 register holes, got {len(holes)}")
            if len(holes) > 3:
                warnings.warn(
                    f"register selection: given {len(holes)} holes; using the first 3, "
                    f"dropping {holes[3:]}", stacklevel=2)
            self.holes, self.dropped = holes[:3], holes[3:]
        self.cells = list(st.getTopSimplices())
        self.frames = _frames(self.cells, gauge=gauge)
        rows, cols, n = st.getDualAdjacency()
        if n != len(self.cells):
            raise RuntimeError(f"dual adjacency count {n} != top cells {len(self.cells)}")
        self.adj = collections.defaultdict(list)
        for r, c in zip(rows, cols):
            self.adj[id(self.cells[r])].append(self.cells[c])

        def cell_of(h):
            hv = set(h)
            return max(self.cells,
                       key=lambda c: (len(hv & set(_top_tuple(c))), _cell_volume(c)))

        self.hole_cells = [cell_of(h) for h in self.holes]
        self.es = T.cobordism.EigenstateSynthesis(st, 3)
        self.by_set = {frozenset(t): (i, list(t))
                       for i, t in enumerate(self.es.cellSimplices())}
        self.eps = induced_orientation_signs(st, self.holes)
        self._joint = {}

    def line(self, i, j):
        """The Spin(4) Wilson line transporting hole `j`'s frame to hole `i`'s."""
        if i == j:
            return np.eye(4, dtype=complex)
        return wilson_line(self.adj, self.hole_cells[i], self.hole_cells[j], self.frames)

    def signed_target(self, target=_SINGLET3):
        """The orientation-canonical (ε-signed) pin targets `ε_q·target_q` — what
        `[1,ω,ω²]` means once each hole's ascending-id orientation is referred to the one
        global orientation (see `induced_orientation_signs`)."""
        return [self.eps[q] * complex(target[q]) for q in range(3)]

    def joint_field(self, target=_SINGLET3):
        """The single correlated multi-hole carried representative
        `carriedRepresentative([h0,h1,h2], ε·target)` — one k=3 cochain carrying the color
        target across ALL THREE holes at once, pinned in the orientation-canonical
        convention so the field is a label-free object — plus its carry residual (the
        emergence certificate: → 0 iff the geometry genuinely carries the target).
        Memoized per target."""
        key = tuple(complex(t) for t in target[:3])
        if key not in self._joint:
            hh = [list(h) for h in self.holes]
            tgt = self.signed_target(target)
            psi = np.asarray(self.es.carriedRepresentative(hh, tgt), dtype=complex)
            self._joint[key] = (psi, float(self.es.residualForPeriods(hh, tgt)))
        return self._joint[key]


# ============================================================================
# Per-hole spinor extraction (the degree-3 slice of the vertical lift)
# ============================================================================
def emergent_spinor(by_set, psi, frame):
    """The per-hole Dirac spinor extracted from the carried representative `psi` (a k=3
    cochain) over the hole cell's five tetrahedral faces, in the cell's own frame. `frame`
    is the cell's `cell_frame` triple `(coords, F, vs)`. Faces are matched by vertex SET
    and oriented from the canonical stored order (never re-sorted here): each face's
    trivector (the 4 det-minors of its 3 edge vectors over `_TRIPLES`) is least-squared
    against `psi`, the 3-form mapped into the Clifford algebra (`Φ = Σ ω_t γ_iγ_jγ_k`),
    and the spinor read as `s = Φ·[1,0,0,0]`, normalized."""
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


def joint_spinors(ctx, target=_SINGLET3):
    """The three per-hole spinors read from the SINGLE joint carried representative — the
    color correlation imprints on all three extractions through the one field. (The
    spinors are still one per hole: the classical cochain supports no joint two-hole
    amplitude, which is exactly the vertical route's structural limitation.)"""
    psi, _res = ctx.joint_field(target)
    return [emergent_spinor(ctx.by_set, psi, ctx.frames[id(c)]) for c in ctx.hole_cells]


def independent_spinors(ctx):
    """The three per-hole spinors from three INDEPENDENT single-hole carried
    representatives (`carriedRepresentative([h],[1.0])`) — the product-read baseline."""
    out = []
    for h, c in zip(ctx.holes, ctx.hole_cells):
        psi = np.asarray(ctx.es.carriedRepresentative([list(h)], [1.0]), dtype=complex)
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
# (a) VERTICAL route — the separable reconstruction from the joint carried field
# ============================================================================
def vertical_read(st, holes=None, spinors=None, ctx=None):
    """The vertical route: per-hole spinors from the joint carried representative (or the
    given `spinors`), Wilson-transported to hole 0's frame, assembled into the
    reconstructed 3-qubit state, and decomposed through its two-hole reduced states ρ_ij.

    HONEST SCOPE: the only 3-qubit state reconstructable from a classical
    (single-particle) cochain is the PRODUCT of the per-hole extractions — any bilinear
    pair read of ψ factorizes (rank-1 ⇒ ρ_ij = ρ_i⊗ρ_j) — so the reported `C_ij` is a
    STRUCTURAL zero (separable by construction), not a measurement of the field's
    connected correlator. What the route measures: the separable correlators
    `⟨S_i·S_j⟩ = ⟨S_i⟩·⟨S_j⟩` of the field's extracted spin directions, and their J².
    A nonzero C_ij would require a genuine two-hole joint lift (the fiber↔cells kernel,
    #495). Returns the `j2_decomposition` dict plus `route`; `None` if a Wilson line is
    missing."""
    if ctx is None:
        ctx = MeshContext(st, holes)
    if spinors is None:
        spinors = joint_spinors(ctx)
    lines0 = [ctx.line(0, j) for j in range(3)]
    if any(u is None for u in lines0):
        return None
    qubits = [_spinor_to_qubit(lines0[j] @ spinors[j]) for j in range(3)]
    rep = j2_decomposition(kron(*qubits))
    rep["route"] = "vertical(separable reconstruction of the joint field)"
    return rep


# ============================================================================
# (b) HORIZONTAL route — the Wilson-line invariants (the measured holonomy)
# ============================================================================
def transport_so4(W):
    """The SO(4) vector rotation the Spin(4) transport `W` covers:
    `R_ab = ¼·Tr[γ_a W γ_b W†]` — genuinely orthogonal with det +1 for any valid Spin(4)
    element (the structural check a broken `facet_transport`/`rotation_to_spin` fails)."""
    Wd = W.conj().T
    return np.array([[float((_G[a] @ W @ _G[b] @ Wd).trace().real) / 4.0
                      for b in range(4)] for a in range(4)])


def transport_spin_projection(W):
    """The transport's action PROJECTED onto the diagonal (physical spatial) spin sector:
    `M_ba = Tr[S_b W S_a W†]` with the structural generators `_SG` — purely a function of
    the holonomy; no spinor is touched. NB `Spin(4) = SU(2)_L × SU(2)_R` and the `S_a` are
    the diagonal su(2); a generic transport also mixes them into the chiral (axial, L−R)
    sector, so `M` is an SO(3) rotation only when that mixing vanishes —
    `axial_mixing(W) = max|MᵀM − I|` quantifies the leakage and is reported per pair."""
    Wd = W.conj().T
    return np.array([[float((_SG[b] @ W @ _SG[a] @ Wd).trace().real)
                      for a in range(3)] for b in range(3)])


def axial_mixing(W):
    """`max|MᵀM − I|` of the diagonal-spin projection `M` — 0 iff the transport keeps the
    diagonal spin sector closed (no chiral/axial leakage); a genuine, gate-checked
    measurement of how far the frame transport is from a pure spatial rotation."""
    M = transport_spin_projection(W)
    return float(np.max(np.abs(M.T @ M - np.eye(3))))


def transport_angle(W):
    """The angle θ read from the diagonal-spin projection's trace,
    `cos θ = (Tr M − 1)/2` (clamped) — exactly the SO(3) rotation angle when the axial
    mixing vanishes, and the same trace invariant the retired #512 experiment used (kept
    for comparability); `axial_mixing` reports how far `M` is from a rotation."""
    c = (np.trace(transport_spin_projection(W)) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, c)))


def horizontal_read(st, holes=None, target=_SINGLET3, ctx=None):
    """The horizontal route: the inter-hole Wilson lines' trace angles θ_ij — genuine,
    frame-free holonomy measurements (from the diagonal-spin projection, with the
    per-pair `axial_mixing` reporting how far each projection is from a pure SO(3)
    rotation) — mapped to the transport-parallel correlators `⟨S_i·S_j⟩ = ¼·cos θ_ij` and
    their J². A K-type frame transport cannot entangle (Claim 1) and no correlated
    transport construction exists in the current machinery, so this route yields NO C_ij
    measurement; nothing is reported as one.

    `color_reference` is NOT a measurement: on a pinned register the carried
    representative's periods equal the caller's targets by construction, so the
    register's color phases are the pinned INPUT — reported label-free via the unsigned
    target (the orientation-canonical physical periods; the ε-signed values are only the
    ascending-id API representation) — with the carry residual as the only emergent
    content (→ 0 iff the geometry genuinely carries the pin).

    Each pair gets its own transport angle, so unlike a realizable 3-qubit state this
    read is not bound by the n=3 frustration floor. Returns per-pair angles, correlators,
    `axial_mixing`, `j2`, and `color_reference`; `None` if a Wilson line is missing."""
    if ctx is None:
        ctx = MeshContext(st, holes)
    W = {(i, j): ctx.line(i, j) for i, j in _PAIRS}
    if any(w is None for w in W.values()):
        return None
    theta = {p: transport_angle(W[p]) for p in _PAIRS}
    sdot = {p: 0.25 * math.cos(theta[p]) for p in _PAIRS}
    _psi, res = ctx.joint_field(target)
    ph = np.angle(np.array(target[:3], complex))    # unsigned: the label-free physical pin
    ph = ph - ph[0]
    return {"route": "horizontal(Wilson-line invariants)",
            "theta_deg": {p: math.degrees(theta[p]) for p in _PAIRS},
            "spin_correlators": sdot,
            "axial_mixing": {p: axial_mixing(W[p]) for p in _PAIRS},
            "j2": 2.25 + 2.0 * float(sum(sdot.values())),
            "color_reference": {
                "phase_deg": {(i, j): math.degrees(float(ph[j] - ph[i]))
                              for i, j in _PAIRS},
                "carry_residual": res}}


# ============================================================================
# The two-route comparison and the emergent read
# ============================================================================
def two_ways(st, holes=None):
    """Experiment 2 — the two routes on one geometry, compared on what they MEASURE: the
    per-pair separable correlators (vertical: the field's extracted spin directions;
    horizontal: the transport-parallel prediction `¼·cos θ_ij`) and their J². Neither
    route measures C_ij (vertical: structural zero — separable by construction;
    horizontal: K-type transport, no correlated construction), so the comparison cannot
    decide the escape-hatch question; `verdict` states exactly that. `reaches_proton`
    reports the ¾-exclusion (|J² − ¾| ≤ 0.1)."""
    ctx = MeshContext(st, holes)
    vert = vertical_read(st, ctx=ctx)
    horiz = horizontal_read(st, ctx=ctx)
    if vert is None or horiz is None:
        return None
    gaps = {
        "sdot_transport": {p: abs(vert["spin_correlators"][p]
                                  - horiz["spin_correlators"][p]) for p in _PAIRS},
        "j2_transport": abs(vert["j2"] - horiz["j2"]),
    }
    return {"vertical": vert, "horizontal": horiz, "gaps": gaps,
            "reaches_proton": {"vertical": abs(vert["j2"] - 0.75) <= 0.1,
                               "horizontal": abs(horiz["j2"] - 0.75) <= 0.1},
            "verdict": ("neither route measures C_ij with current machinery (vertical: "
                        "separable by construction; horizontal: K-type transport) — the "
                        "escape-hatch question is undecided by this experiment; measured: "
                        "the 3/4-exclusion and the separable-correlator gaps")}


def emergent_read(st, holes=None):
    """Experiment 3 — the readout on a converged fixture's joint carried representative vs
    the independent per-hole (product) baseline: does the joint read's J² move below the
    product read toward ¾?  (Prior joint_* prototype: 1.80 → 1.64.) Both reads are
    separable reconstructions — any movement is the color correlation imprinting on the
    extracted spin directions, not entanglement."""
    ctx = MeshContext(st, holes)
    joint = vertical_read(st, spinors=joint_spinors(ctx), ctx=ctx)
    prod = vertical_read(st, spinors=independent_spinors(ctx), ctx=ctx)
    if joint is None or prod is None:
        return None
    prod = dict(prod, route="vertical(independent product)")
    return {"joint": joint, "product": prod,
            "moved_toward_proton": joint["j2"] < prod["j2"]}


# ============================================================================
# Validation gates — GAUGE (random per-cell frame rotation), RELABEL (vertex permutation)
# Both compare EVERY reported numeric channel, not just the scalar J².
# ============================================================================
def report_delta(a, b):
    """The max absolute difference over every numeric leaf of two nested report dicts
    (floats, complex, arrays; dict keys must match — a missing key is an error; strings
    are skipped). This is what the gates compare, so per-pair channels are gated too."""
    if isinstance(a, dict):
        if set(a) != set(b):
            raise KeyError(f"report keys differ: {sorted(a)} vs {sorted(b)}")
        deltas = [report_delta(a[k], b[k]) for k in a]
        return max(deltas) if deltas else 0.0
    if isinstance(a, str):
        return 0.0
    return float(np.max(np.abs(np.asarray(a, dtype=complex)
                               - np.asarray(b, dtype=complex))))


def gauge_gate(read, st, holes, seed=7):
    """Max |Δ| over every reported channel under a random per-cell SO(4) rotation of the
    embedding, threaded explicitly through `MeshContext(gauge=...)` (no module state is
    patched — the base and gauged reads use two independent contexts). Post-hoc only."""
    base = read(st, holes)
    rng = np.random.default_rng(seed)
    rmap = {}

    def gauge(key):
        if key not in rmap:
            a = rng.standard_normal((4, 4))
            rmap[key] = scipy.linalg.expm(a - a.T)
        return rmap[key]

    rotated = read(st, holes, ctx=MeshContext(st, holes, gauge=gauge))
    return report_delta(base, rotated)


def relabel_gate(read, cells, edges, holes, seed=3):
    """Max |Δ| over every reported channel under a random vertex-id permutation of the
    whole complex — with the cell LIST order shuffled too, so the gate also catches
    dependence on enumeration order. The holes are RE-DERIVED on the relabeled complex
    (`MultiCobordism.emergent_holes` — verifying detection finds the same physical holes),
    and the original register's images are matched among them by permuted vertex set so
    the target assignment compares like with like (a missing image is an error, not a
    gate value; the "first N" selection convention itself is labeling-relative, which is
    why the gate aligns rather than re-selects). Post-hoc only."""
    st = build_fixture(cells, edges)
    base = read(st, [tuple(h) for h in holes])
    rng = np.random.default_rng(seed)
    allv = sorted({v for c in cells for v in c})
    shuf = allv[:]
    rng.shuffle(shuf)
    perm = dict(zip(allv, shuf))
    relabeled = [[perm[v] for v in c] for c in cells]
    st2 = build_fixture([relabeled[i] for i in rng.permutation(len(relabeled))],
                        {tuple(sorted((perm[a], perm[b]))): z
                         for (a, b), z in edges.items()})
    import tessera as T
    rederived = [tuple(h) for h in T.cobordism.MultiCobordism.emergent_holes(st2, 3)]
    found = {frozenset(h): h for h in rederived}
    holes2 = []
    for h in [tuple(x) for x in holes[:3]]:
        image = frozenset(perm[v] for v in h)
        if image not in found:
            raise ValueError(f"emergent_holes on the relabeled complex is missing the "
                             f"image of hole {h}")
        holes2.append(found[image])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # the surplus-hole warning already fired once
        relabeled_report = read(st2, holes2)
    return report_delta(base, relabeled_report)


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
    validated 3-hole register (`register_holes`; any surplus hole is warned and dropped
    there, never silently)."""
    with open(os.path.join(_FIXTURES, name)) as fh:
        d = json.load(fh)
    cells = [list(c) for c in d["cells"]]
    edges = {}
    for k, (re, im) in d["edges"].items():
        a, b = (int(x) for x in k.split(","))
        edges[(a, b)] = complex(re, im)
    st = build_fixture(cells, edges)
    holes, _dropped = register_holes(st)
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

    print("== 2. the two routes (synthetic_b3_3) ==")
    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter("always")
        cells, edges, st, holes = load_fixture("synthetic_b3_3.json")
    for w in wlog:
        print(f"  [register] {w.message}")
    out = two_ways(st, holes)
    v, h, g = out["vertical"], out["horizontal"], out["gaps"]
    print(f"  vertical   (separable reconstruction)  J²={v['j2']:.4f}  "
          f"ΣC_ij={sum(v['C_ij'].values()):+.2e}  [structural zero — not a measurement]")
    print(f"  horizontal (transport θ_ij)            J²={h['j2']:.4f}  "
          f"θ_ij=" + ", ".join(f"{x:.0f}°" for x in h["theta_deg"].values())
          + "  [no C_ij measurement — K-type]")
    print(f"  axial mixing of the spin projection (0 = pure SO(3)): "
          + ", ".join(f"{p[0]}{p[1]}={x:.3f}" for p, x in h["axial_mixing"].items()))
    cr = h["color_reference"]
    print(f"  color reference (pinned input, NOT measured): Δφ_ij="
          + ", ".join(f"{x:.0f}°" for x in cr["phase_deg"].values())
          + f"  carry residual={cr['carry_residual']:.2e}")
    print(f"  per-pair ⟨S_i·S_j⟩ gaps (vert vs transport): "
          + ", ".join(f"{p[0]}{p[1]}={x:.4f}" for p, x in g["sdot_transport"].items()))
    print(f"  J² gap: |vert−transport|={g['j2_transport']:.4f}")
    print(f"  reaches ¾?  vertical={out['reaches_proton']['vertical']}  "
          f"horizontal={out['reaches_proton']['horizontal']}")
    print(f"  verdict: {out['verdict']}\n")

    print("== gates (synthetic_b3_3): max |Δ| over EVERY reported channel ==")
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
          f"ΣC_ij={sum(j['C_ij'].values()):+.2e}  [both reads separable]")
    print(f"  moved toward ¾ below the product read: {em['moved_toward_proton']} "
          f"(prior joint_* prototype baseline: 1.80 → 1.64)")
    for label, read in (("vertical", vertical_read), ("horizontal", horizontal_read)):
        dg = gauge_gate(read, cst, choles)
        dr = relabel_gate(read, ccells, cedges, choles)
        print(f"  gates {label:<10} |ΔGAUGE|={dg:.2e}  |ΔRELABEL|={dr:.2e}")


if __name__ == "__main__":
    main()
