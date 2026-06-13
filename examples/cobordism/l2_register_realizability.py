# MIT License
# Copyright (c) 2025 Andrew Kelleher
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Spectral gate realizability on the L_2 register: the same staged synthesis as
``spectral_gate_realizability.py``, one degree up.

The carried register here is **ker L_2 of a surgery-opened triangulated S^3** --
the 12-vertex join of two hexagons, with three pairwise vertex-disjoint
tetrahedra removed by the boundary-fixed surgery. The register theorems are
dimension-blind (sphere minus three balls with L_{n-1} in any n): opening the
holes grows b_2 0 -> 2, ker L_2 emerges as the two-dimensional carried register,
the harmonic periods over the three boundary 2-spheres conserve total charge,
and the realizable gate set is exactly the charge-conservation criterion -- the
same 13 named gates as the 2d register, decided by the same Hodge spectrum.

What is genuinely new at this degree is the substrate, not the gate set: on
3-complex bulks the Regge action has stationary points (vanishing interior
deficit angles), so the mediated objective F_beta = r_U + beta*|S_Regge| can
select among the bulks realizing a gate -- in two dimensions the Regge term is
Gauss--Bonnet-topological and exerts no selection pressure.

Everything battery-shaped is imported from ``spectral_gate_realizability.py``
(the gates, thresholds, criterion, Choi reading, parallel pool); this module
owns only the geometry layer: (p,q)-gon join seeds, tet holes, signed
tet-facet periods, and the 3d composed stellar growth (cone a tetrahedron's
four faces onto a fresh interior vertex, then remove the parent tet).

A note on growth and extra cuts: a cell is "interior" to the synthesis only if
it touches no boundary vertex, and the canonical hexagon join's three holes
consume all 12 vertices -- so on the canonical bulk neither extra cuts nor
additive growth can start. The topology search therefore draws larger
(p,q)-join seeds (16+ vertices), where interior room exists; this is the same
seed-laddering the 2d search does with geodesic subdivisions.

Run:
    python examples/cobordism/l2_register_realizability.py
    python examples/cobordism/l2_register_realizability.py --h3
    python examples/cobordism/l2_register_realizability.py --retries 300 --jobs 10
    python examples/cobordism/l2_register_realizability.py --gate sqrt-SWAP
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_base():
    """Load the 2d example as a module and register it in sys.modules so the
    spawn-based parallel pool can re-resolve its functions in worker processes."""
    name = "spectral_gate_realizability"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_HERE, "spectral_gate_realizability.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base()                       # sets the 10-CPU thread caps on import
np = BASE.np
tessera = BASE.tessera
cob = tessera.cobordism

REALIZE = BASE.REALIZE
CERT_FLOOR = BASE.CERT_FLOOR
CANONICAL_SET = BASE.CANONICAL_SET
_CP_IN = BASE._CP_IN


# --------------------------------------------------------------------------- #
# The (p,q)-gon join family of triangulated S^3 seeds. S^3 = S^1 * S^1: every
# tetrahedron is (an edge of the p-gon) x (an edge of the q-gon), so a p-gon
# joined with a q-gon gives p+q vertices and p*q tets. Vertex-disjoint hole
# triples need three pairwise disjoint edges in EACH factor (p, q >= 6).
# --------------------------------------------------------------------------- #
def _join_cells(p, q):
    """The tetrahedra of the p-gon * q-gon join (vertices 0..p-1, then p..p+q-1)."""
    ae = [(i, (i + 1) % p) for i in range(p)]
    be = [(p + j, p + (j + 1) % q) for j in range(q)]
    return sorted(tuple(sorted((*ea, *eb))) for ea in ae for eb in be)


def _diag_holes(p, q, k=3):
    """The canonical C_3-orbit hole triple: the first k even edges of each factor,
    diagonally matched -- cyclically permuted by the shift (i,j) -> (i+2, j+2)
    (for p = q = 6 this is the verified hexagon-join triple; for p = q = 9 use
    stride 3 via _stride_holes)."""
    return [tuple(sorted((2 * t, (2 * t + 1) % p,
                          p + 2 * t, p + (2 * t + 1) % q))) for t in range(k)]


def _stride_holes(p, q, stride, k=3):
    """k diagonally-matched holes at the given edge stride in each factor."""
    return [tuple(sorted((stride * t, (stride * t + 1) % p,
                          p + stride * t, p + (stride * t + 1) % q)))
            for t in range(k)]


_HEXJOIN = _join_cells(6, 6)
_HEXJOIN_HOLES = _diag_holes(6, 6)        # the verified canonical triple


def _tet_facets(cell):
    """The four oriented boundary triangles of a sorted tetrahedron (a,b,c,d),
    with the (-1)^j induced-orientation signs of the boundary operator."""
    a, b, c, d = sorted(cell)
    return [((b, c, d), +1), ((a, c, d), -1), ((a, b, d), +1), ((a, b, c), -1)]


def _bulk(cells, weight=1.0, phase=0.0):
    """A pre-geometric 3-complex (top cells = tetrahedra) from a tet list, all
    edges pinned to a uniform Hermitian weight -- the 3d twin of the 2d builder."""
    return tessera.Spacetime.fromCells(3, [list(c) for c in cells], weight, phase)


# --------------------------------------------------------------------------- #
# Dual-complex validity. The mediated objective scores the DUAL complex
# (ReggeSolver::dualReggeAction on W*), and the Poincare/Lefschetz dual block
# decomposition is a valid cell complex iff the primal is a combinatorial
# manifold(-with-boundary). The check itself lives in C++
# (ChainComplex::dualComplexIsValid, EigenstateSynthesis::dualComplexValid)
# so every layer -- the registers here, and the C++ growth/surgery paths --
# can gate moves on it; these are thin delegating wrappers. (Metric
# dual-validity -- non-degenerate circumcentric volumes -- is the mediation
# layer's separate concern; on unit-metric register bulks circumcenters are
# barycentric and fine automatically.)
# --------------------------------------------------------------------------- #
def dual_complex_is_valid(top_cells, n, facet_cells=None):
    """(ok, reason): is the dual block complex of this pure n-complex a valid
    cell decomposition -- equivalently, is the primal a combinatorial manifold
    with boundary? *facet_cells*, when given (the complex's full (n-1)-cell
    list), also catches dangling facets: (n-1)-cells with zero cofaces that
    the Hodge Laplacian still sees even though no top cell carries them.
    Delegates to ``ChainComplex.dualComplexIsValid``."""
    ok, why = cob.ChainComplex.dualComplexIsValid(
        [[int(v) for v in c] for c in top_cells], int(n),
        [[int(v) for v in f] for f in (facet_cells or [])])
    return bool(ok), str(why)


def register_dual_valid(es, n=3):
    """The dual-validity verdict for a synthesis object's CURRENT complex: top
    cells from the surgery state, facet cells from the (n-1)-cell list the
    Hodge Laplacian is built over. Delegates to
    ``EigenstateSynthesis.dualComplexValid``."""
    ok, why = es.dualComplexValid()
    return bool(ok), str(why)


def _betti2(st):
    return [int(b) for b in cob.ChainComplex.fromSpacetime(st).bettiNumbers()][2]


def _ker_l2_dim(st):
    return len(cob.HodgeLaplacian(st).harmonics(2))


# --------------------------------------------------------------------------- #
# The L_2 register: ker L_2 of the surgery-grown S^3, read by eigendecomposition.
# The structural mirror of the 2d Register, with circle periods over hole edges
# replaced by signed 2-sphere periods over hole facets.
# --------------------------------------------------------------------------- #
class RegisterL2:
    """The carried register V = ker L_2 of the surgery-grown S^3. Holds the grown
    bulk, its degree-2 ``EigenstateSynthesis`` (the genuine L_2 residual core),
    the harmonic 2-forms in the bulk's cell order, their hole-facet restriction
    and period rows, and the orientation signs that symmetrize the
    boundary-period constraint to Sigma = 0. All OUTPUTS read off the grown bulk.

    The default ``RegisterL2()`` is the verified hexagon-join / 3-canonical-hole
    register. ``cells`` / ``class_holes`` / ``extra_holes`` / ``grow_vertices``
    drive the surgery-topology search exactly as in the 2d Register: a different
    (p,q)-join seed, a different vertex-disjoint tet triple, extra
    ``removeInteriorCell`` surgeries, and seeded additive stellar growth."""

    def __init__(self, cells=_HEXJOIN, class_holes=_HEXJOIN_HOLES, extra_holes=(),
                 grow_vertices=0, grow_seed=0):
        self.seed_cells = list(cells)
        self.class_holes = [tuple(sorted(h)) for h in class_holes]
        self.reg_facets = [f for hole in self.class_holes
                           for f, _s in _tet_facets(hole)]
        self.fidx = {f: i for i, f in enumerate(self.reg_facets)}

        self.st = _bulk(self.seed_cells)
        self.es = cob.EigenstateSynthesis(self.st, 2)
        for hole in self.class_holes:                       # the holonomy holes
            self.es.removeInteriorCell(list(hole))
        self.grown = self._stellar_grow(grow_vertices, grow_seed)
        self.extra_opened = []                              # extra surgery (b_2 growth)
        for cell in extra_holes:
            cs = tuple(sorted(cell))
            avail = {tuple(sorted(int(v) for v in c))
                     for c in self.es.interiorTopCells()}
            if cs in avail and self.es.removeInteriorCell(list(cs)):
                ok, _why = register_dual_valid(self.es)
                if not ok:                  # accept moves only if the DUAL stays
                    self.es.restoreLastRemoval()            # a valid complex
                    continue
                self.extra_opened.append(cs)
        self.dual_valid, self.dual_reason = register_dual_valid(self.es)

        # The register core reads off the C++ layer (#286), exactly as the 2d
        # Register: harmonic amplitude matrix, sphere-period rows with the
        # boundary operator's signs, and the deterministic end sign covector
        # from the surface's fundamental chain.
        self.cells = [tuple(int(v) for v in c) for c in self.es.cellSimplices()]
        self.H_full = np.asarray(
            cob.HodgeLaplacian(self.st).harmonicMatrix(2),
            dtype=complex).reshape(-1, len(self.cells))
        self.dim = int(self.H_full.shape[0])
        self._reg_col = [self.cells.index(f) for f in self.reg_facets]
        self.P = np.asarray(
            self.es.cyclePeriods([list(h) for h in self.class_holes]),
            dtype=complex).reshape(self.dim, len(self.class_holes))
        self.n = np.asarray(cob.ChainComplex.endSignCovector(
            [[int(v) for v in c] for c in self.es.topCells()],
            [list(h) for h in self.class_holes]), dtype=float)
        self.sign = self.n.copy()

    def _stellar_grow(self, n, seed):
        """ADD up to *n* interior vertices by boundary-fixed stellar subdivision,
        composed from the two surgery primitives exactly as in 2d: cone a fresh
        vertex onto an interior tetrahedron's four faces (``attachInteriorVertex``
        with the facet fan -- dW untouched), then remove the subdivided tet
        (``removeInteriorCell`` -- its faces keep two cofaces, so dW stays
        bit-exact). Each application adds ONE vertex and preserves ker L_2 (the
        fan is homotopic to the tet it replaces); sites are drawn by the seeded
        RNG from the current interior top cells. On seeds whose holes consume
        every vertex (the canonical hexagon join) there is no interior site and
        the budget is simply unused."""
        rng = random.Random(int(seed))
        grown = 0
        for _ in range(int(n)):
            sites = sorted(tuple(sorted(int(v) for v in c))
                           for c in self.es.interiorTopCells())
            if not sites:
                break
            cell = rng.choice(sites)
            fan = [list(f) for f, _s in _tet_facets(cell)]
            if not self.es.attachInteriorVertex(fan):
                continue
            if not self.es.removeInteriorCell(list(cell)):
                self.es.detachLastInteriorVertex()
                continue
            ok, _why = register_dual_valid(self.es)
            if not ok:                      # accept moves only if the DUAL stays
                self.es.restoreLastRemoval()                # a valid complex
                self.es.detachLastInteriorVertex()
                continue
            grown += 1
        if grown:
            # attachInteriorVertex wires the new cells through the endpoint
            # TIME rule rather than a causal cone placement; re-pin the bulk
            # uniform so the documented unit cochain metric holds by
            # construction (the 2d register does the same).
            for e in self.st.getEdgeList().toVector():
                e.setSquaredLength(1.0)
                e.setPhase(0.0)
        return grown

    @property
    def rank(self):
        """The rank of the carried period space over the holonomy holes -- same
        genuine (rank < #holes) vs saturated (rank == #holes) semantics as 2d."""
        return int(np.linalg.matrix_rank(self.P, tol=1e-9)) if self.dim else 0

    def harmonic_form(self, raw_periods):
        """The carried harmonic 2-form whose three sphere-periods are the
        projection of *raw_periods* onto the carried period space, plus a minimal
        leak 2-form on one facet per hole so the cochain's periods are EXACTLY
        *raw_periods* -- the L_2 twin of the 2d leak construction."""
        coeffs, *_ = np.linalg.lstsq(self.P.T, raw_periods, rcond=None)
        full = (coeffs @ self.H_full).astype(complex)
        leak = raw_periods - coeffs @ self.P
        for k, hole in enumerate(self.class_holes):
            first_facet = _tet_facets(hole)[0][0]           # sign +1 by convention
            full[self._reg_col[self.fidx[first_facet]]] += leak[k]
        return full

    def spectral_residual(self, raw_periods):
        """The genuine Hodge residual ||(I-psi psi^dag) L_2 psi||^2 of the 2-form
        with the given raw periods -- the continuous spectral realizability score.
        -> 0 iff the periods lie in the carried register V.
        One C++ call (the lstsq-project-leak-residual verdict primitive)."""
        return float(self.es.residualForPeriods(
            [list(h) for h in self.class_holes],
            [complex(z) for z in np.asarray(raw_periods, dtype=complex)]))


# --------------------------------------------------------------------------- #
# STAGE 3 emergence and the identity anchor, on the hexagon join.
# --------------------------------------------------------------------------- #
def register_emergence():
    """Surgery opens the three tet holes one at a time; b_2 and ker L_2 emerge
    0 -> 2 from the spectrum (the closed S^3 carries nothing)."""
    st = _bulk(_HEXJOIN)
    es = cob.EigenstateSynthesis(st, 2)
    interior = {tuple(sorted(int(v) for v in c)) for c in es.interiorTopCells()}
    trace = [{"step": "closed S^3 seed", "b2": _betti2(st), "kerL2": _ker_l2_dim(st)}]
    for hole in _HEXJOIN_HOLES:
        assert hole in interior, "holonomy hole must be a genuine interior cell"
        es.removeInteriorCell(list(hole))
        trace.append({"step": f"open {hole}", "b2": _betti2(st),
                      "kerL2": _ker_l2_dim(st)})
    return trace


def synthesize_state(reg, raw_periods):
    """STAGE 1 for one register state at degree 2: confirm the state is carried as
    a harmonic of the grown bulk by the genuine metric Hodge residual."""
    res = reg.spectral_residual(raw_periods)
    return res, int(reg.st.getVertexList().size()), int(reg.es.order())


def identity_anchor(reg):
    """The falsifiable core at degree 2: the identity post-interaction state is
    carried ONLY once surgery has grown the full register (b_2 = 2, ker L_2 = 2);
    on the closed S^3, the one-hole ball, and the two-hole shell it floors."""
    raw = reg.sign * _CP_IN
    psi_full = reg.harmonic_form(raw)
    by_cell = {reg.cells[i]: psi_full[i] for i in range(len(reg.cells))}
    rows = []
    for k in range(len(_HEXJOIN_HOLES) + 1):
        st = _bulk(_HEXJOIN)
        es = cob.EigenstateSynthesis(st, 2)
        for hole in _HEXJOIN_HOLES[:k]:
            es.removeInteriorCell(list(hole))
        cells = [tuple(int(v) for v in c) for c in es.cellSimplices()]
        psi = np.array([by_cell.get(c, 0.0) for c in cells], dtype=complex)
        res = float(es.residual([complex(z) for z in psi]))
        rows.append({"holes_open": k, "b2": _betti2(st), "kerL2": _ker_l2_dim(st),
                     "residual": res, "realizable": bool(res < REALIZE)})
    return rows


# --------------------------------------------------------------------------- #
# STAGE 3 gate scoring -- identical decision, one degree up.
# --------------------------------------------------------------------------- #
def post_interaction(reg, U):
    """The post-interaction state U|psi_B> on the carried register: the spectral
    residual of the 2-form with raw periods sign * (U_reg cp_in) -> 0 iff carried
    by ker L_2. Returns (residual, b_2, leakage |Sigma|)."""
    u_reg = np.asarray(U, dtype=complex)[1:4, 1:4]
    cp_out = u_reg @ _CP_IN.astype(complex)
    res = reg.spectral_residual(reg.sign * cp_out)
    return res, _betti2(reg.st), float(abs(cp_out.sum()))


def gate_sweep(reg, on_progress=None):
    """The full battery on the L_2 register (serial; each gate is one residual)."""
    rows = []
    for name, U, fam in BASE._gates():
        res, b2, leak = post_interaction(reg, U)
        rows.append({"gate": name, "family": fam, "residual": res, "b2": b2,
                     "leak": leak, "realizable": bool(res < REALIZE)})
        if on_progress is not None:
            on_progress()
    return rows


# --------------------------------------------------------------------------- #
# --h3: the value level at degree 2 -- anchor, Gram, Z_spec vs the amplitude,
# bulk independence across join seeds. Mirrors the 2d implementation exactly.
# --------------------------------------------------------------------------- #
def _carried_form(reg, raw_periods):
    """The pure ker-L_2 representative (least squares, NO leak correction) and the
    norm of the un-carried period remainder."""
    raw = np.asarray(raw_periods, dtype=complex)
    coeffs, *_ = np.linalg.lstsq(reg.P.T, raw, rcond=None)
    full = (coeffs @ reg.H_full).astype(complex)
    leak = raw - coeffs @ reg.P
    return full, float(np.linalg.norm(leak))


def register_gram(reg, scale):
    """The register Gram in period coordinates on a flat-orthonormal basis of the
    Sigma = 0 subspace. G = I is the period-map isometry; the hexagon join's hole
    triple is a full S_3 orbit of its automorphism group, so the isometric-chart
    proposition applies (a fortiori: a cyclic symmetry plus the real metric
    already suffices)."""
    e1 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    e2 = np.array([1.0, 1.0, -2.0]) / np.sqrt(6.0)
    forms = [_carried_form(reg, reg.sign * e.astype(complex))[0] for e in (e1, e2)]
    return scale * np.array([[complex(np.vdot(a, b)) for b in forms] for a in forms])


def h3_value_sweep(reg, n_states=8, seed=2026, on_progress=None):
    """H3 on the L_2 spectral data: Z_spec = scale * <h(psi_A), h(U psi_B)> against
    the flat register amplitude and the independent Choi reading, for every gate
    the construction realizes. One scale from the T1 anchor; then predictions."""
    hodge = cob.HodgeLaplacian(reg.st)
    w2 = np.asarray(hodge.weights(2), dtype=float)
    # At degree 2 the cochain weights are simplex AREAS, so the unit-edge pin
    # gives the uniform equilateral metric w_2 = sqrt(3)/4 -- a single scalar,
    # absorbed by the T1 anchor. Uniformity (not unit value) is what the
    # plain-contraction pairing and the isometric-chart proposition require.
    info = {"metric_mean": float(w2.mean()),
            "metric_uniform_dev": float(np.max(np.abs(w2 - w2.mean())))}

    cp_b = (_CP_IN / np.linalg.norm(_CP_IN)).astype(complex)
    h_b, leak_b = _carried_form(reg, reg.sign * cp_b)
    info["psi_b_leak"] = leak_b
    scale = 1.0 / float(np.vdot(h_b, h_b).real)           # the T1 anchor
    info["scale"] = scale
    info["gram"] = register_gram(reg, scale)
    info["gram_dev"] = float(np.max(np.abs(info["gram"] - np.eye(2))))

    states = [cp_b] + BASE.random_carried_states(n_states, seed)
    forms_a = [_carried_form(reg, reg.sign * cp)[0] for cp in states]

    rows = []
    for name, U, fam in BASE._gates():
        u_reg = np.asarray(U, dtype=complex)[1:4, 1:4]
        cp_out = u_reg @ cp_b
        res, _b2, leak = post_interaction(reg, U)
        realized = bool(res < REALIZE)
        if not realized:
            rows.append({"gate": name, "family": fam, "realizable": False,
                         "residual": res, "leak": leak,
                         "max_dev": None, "choi_dev": None, "n_pairs": 0})
            if on_progress is not None:
                on_progress()
            continue
        h_out, _ = _carried_form(reg, reg.sign * cp_out)
        devs, choi_devs = [], []
        for cp_a, h_a in zip(states, forms_a):
            amp = complex(np.vdot(cp_a, cp_out))
            z = scale * complex(np.vdot(h_a, h_out))
            devs.append(abs(z - amp))
            choi_devs.append(abs(BASE._amp_choi(U, cp_a, cp_out, cp_b) - amp))
        rows.append({"gate": name, "family": fam, "realizable": True,
                     "residual": res, "leak": leak,
                     "max_dev": float(max(devs)), "choi_dev": float(max(choi_devs)),
                     "n_pairs": len(states)})
        if on_progress is not None:
            on_progress()
    return rows, info


def _equivariant_variant():
    """The symmetry-preserving bulk variant: the (9,9)-gon join with the hole
    triple at edge stride 3 -- cyclically permuted by the shift (i,j) -> (i+3,
    j+3), so the isometric-chart proposition applies and the value must carry
    over exactly. The holes are still tetrahedra, so the boundary pair is the
    same three triangulated 2-spheres as the canonical register's."""
    return RegisterL2(cells=_join_cells(9, 9), class_holes=_stride_holes(9, 9, 3))


def _vertex_disjoint_tets(cells, k, rng, tries=600):
    """k pairwise vertex-disjoint tetrahedra drawn at random from a seed."""
    cl = [tuple(sorted(c)) for c in cells]
    for _ in range(tries):
        pick = rng.sample(cl, k)
        verts = [v for c in pick for v in c]
        if len(set(verts)) == 4 * k:
            return pick
    return None


def _anisotropic_variant_registers(n_variants, seed):
    """Re-grown GENUINE registers with generic (seeded, vertex-disjoint) tet
    triples on the (8,8) join -- carried, but with no symmetry to enforce an
    isometric chart. Their Gram defect is the control the equivariant witness is
    read against."""
    rng = random.Random(seed)
    out, tries = [], 0
    cells = _join_cells(8, 8)
    while len(out) < n_variants and tries < 60:
        tries += 1
        holes = _vertex_disjoint_tets(cells, 3, rng)
        if holes is None:
            continue
        reg = RegisterL2(cells=cells, class_holes=holes)
        if reg.dim != 2 or reg.rank >= len(reg.class_holes):
            continue                                      # saturated / degenerate
        if post_interaction(reg, BASE._gates()[0][1])[0] >= REALIZE:
            continue                                      # identity anchor must hold
        out.append(reg)
    return out


def h3_invariance(n_variants=2, n_states=4, seed=2026, on_progress=None):
    """Bulk independence of the value at degree 2, and what it turns on: the H3
    table re-run on re-grown registers with the SAME state battery. On the
    symmetric (9,9)-join variant the value carries over exactly; on generic
    (8,8) hole draws the chart is anisotropic and the deviation from the
    amplitude equals the Gram-defect prediction a^dag (G - I) b exactly."""
    cp_b = (_CP_IN / np.linalg.norm(_CP_IN)).astype(complex)
    states = [cp_b] + BASE.random_carried_states(n_states, seed)
    realized = [(name, np.asarray(U, dtype=complex)[1:4, 1:4])
                for name, U, _f in BASE._gates() if BASE.conserves_charge(U)]
    e_basis = np.array([[1.0, -1.0, 0.0] / np.sqrt(2.0),
                        [1.0, 1.0, -2.0] / np.sqrt(6.0)])

    def survey(reg):
        h_b, _ = _carried_form(reg, reg.sign * cp_b)
        scale = 1.0 / float(np.vdot(h_b, h_b).real)
        gram = register_gram(reg, scale)
        forms_a = [_carried_form(reg, reg.sign * cp)[0] for cp in states]
        vals, defect = {}, 0.0
        for name, u_reg in realized:
            cp_out = u_reg @ cp_b
            h_out, _ = _carried_form(reg, reg.sign * cp_out)
            b = e_basis @ cp_out
            zs = []
            for cp_a, h_a in zip(states, forms_a):
                z = scale * complex(np.vdot(h_a, h_out))
                amp = complex(np.vdot(cp_a, cp_out))
                a = e_basis @ cp_a
                predicted = complex(np.conj(a) @ (gram - np.eye(2)) @ b)
                defect = max(defect, abs((z - amp) - predicted))
                zs.append(z)
            vals[name] = zs
        return vals, float(np.max(np.abs(gram - np.eye(2)))), defect

    base_vals, _gd, _dd = survey(RegisterL2())
    if on_progress is not None:
        on_progress()

    def drift_vs_base(vals):
        return float(max(abs(vals[name][k] - base_vals[name][k])
                         for name in vals for k in range(len(states))))

    eq = _equivariant_variant()
    eq_vals, eq_gram_dev, eq_defect = survey(eq)
    equivariant = {"nV": int(eq.st.getVertexList().size()), "rank": eq.rank,
                   "gram_dev": eq_gram_dev, "drift": drift_vs_base(eq_vals),
                   "defect_residual": eq_defect}
    if on_progress is not None:
        on_progress()

    anisotropic = []
    for reg in _anisotropic_variant_registers(n_variants, seed):
        vals, gram_dev, defect = survey(reg)
        anisotropic.append({"nV": int(reg.st.getVertexList().size()),
                            "rank": reg.rank, "gram_dev": gram_dev,
                            "drift": drift_vs_base(vals),
                            "defect_residual": defect})
        if on_progress is not None:
            on_progress()
    return {"equivariant": equivariant, "anisotropic": anisotropic,
            "n_gates": len(realized), "n_pairs": len(states)}


# --------------------------------------------------------------------------- #
# The surgery-topology search at degree 2: randomized (p,q)-join seeds, random
# vertex-disjoint tet triples, extra cuts, additive stellar growth -- then the
# full battery re-decided by the same L_2 spectrum.
# --------------------------------------------------------------------------- #
_SEEDS = [(6, 6), (8, 8), (6, 8), (8, 6), (10, 10)]
_SEED_WEIGHTS = [3, 3, 2, 2, 1]


def _extra_tets(cells, holes, n, rng):
    """Up to *n* extra tets, pairwise vertex-disjoint and disjoint from the
    holonomy holes (the Register opens the genuinely removable subset)."""
    if n <= 0:
        return []
    used = {v for h in holes for v in h}
    cand = [tuple(sorted(c)) for c in cells
            if not (set(c) & used) and tuple(sorted(c)) not in holes]
    rng.shuffle(cand)
    out, chosen = [], set()
    for c in cand:
        if len(out) >= n:
            break
        if set(c) & chosen:
            continue
        out.append(c)
        chosen |= set(c)
    return out


def score_variant(cells, holes, extra, grow=0, grow_seed=0):
    """Build the L_2 register on one surgery-grown 3-topology and re-decide the
    full battery. Same compact summary and genuine/saturated semantics as 2d."""
    reg = RegisterL2(cells=cells, class_holes=holes, extra_holes=extra,
                     grow_vertices=grow, grow_seed=grow_seed)
    realized, by_name = [], {}
    for name, U, _fam in BASE._gates():
        res, _b2, _leak = post_interaction(reg, U)
        ok = bool(res < REALIZE)
        by_name[name] = ok
        if ok:
            realized.append(name)
    rank = reg.rank
    n_holes = len(reg.class_holes)
    identity_ok = by_name.get("Identity", False)
    s3_all = all(by_name.get(n, False) for n in CANONICAL_SET[:6])
    genuine = bool(identity_ok and s3_all and rank < n_holes
                   and reg.dual_valid
                   and len(realized) < len(BASE._gates()))
    extends = sorted(g for g in realized if g not in CANONICAL_SET) if genuine else []
    return {
        "seed": None, "nV": int(reg.st.getVertexList().size()),
        "b2": _betti2(reg.st), "dim": reg.dim, "rank": rank,
        "n_holes": n_holes, "n_extra": len(reg.extra_opened),
        "n_grown": reg.grown,
        "dual_valid": reg.dual_valid,
        "realized": realized, "n_realized": len(realized),
        "identity": identity_ok, "s3_all": s3_all,
        "saturated": bool(rank >= n_holes), "genuine": genuine, "extends": extends,
    }


def _retry_worker(task):
    """One surgery-topology retry at degree 2: pick a (p,q)-join seed, a
    vertex-disjoint tet triple, extra cuts, and an additive-growth count by the
    retry's RNG, then score the variant."""
    idx, base_seed, kmax, max_add = task
    rng = random.Random(base_seed * 1_000_003 + idx)
    p, q = rng.choices(_SEEDS, weights=_SEED_WEIGHTS)[0]
    cells = _join_cells(p, q)
    holes = _vertex_disjoint_tets(cells, 3, rng)
    if holes is None:
        return None
    n_extra = rng.randint(0, kmax)
    extra = _extra_tets(cells, holes, n_extra, rng)
    n_grow = rng.randint(0, max(int(max_add), 0))
    grow_seed = rng.randrange(1, 2**31)
    try:
        out = score_variant(cells, holes, extra, grow=n_grow, grow_seed=grow_seed)
    except Exception:                                       # a degenerate draw
        return None
    out["seed"] = f"{p}x{q}"
    return out


def surgery_search(retries, jobs, base_seed=12345, kmax=3, max_add=20,
                   on_progress=None):
    """Score *retries* randomized surgery-grown 3-topologies in parallel and
    aggregate -- the degree-2 twin of the 2d search."""
    tasks = [(i, base_seed, kmax, max_add) for i in range(retries)]
    results = [r for r in BASE._parallel_map(_retry_worker, tasks, jobs,
                                             on_progress=on_progress) if r]
    genuine = [r for r in results if r["genuine"]]
    saturated = [r for r in results if r["saturated"]]
    invalid = [r for r in results if not r["genuine"] and not r["saturated"]]
    extensions = [r for r in genuine if r["extends"]]
    genuine_sizes = Counter(r["n_realized"] for r in genuine)
    new_gates = sorted({g for r in extensions for g in r["extends"]})
    return {
        "retries": retries, "scored": len(results),
        "n_genuine": len(genuine), "n_saturated": len(saturated),
        "n_invalid": len(invalid),
        "n_dual_invalid": sum(1 for r in results if not r["dual_valid"]),
        "seeds": sorted(Counter(r["seed"] for r in results).items()),
        "max_b2": max((r["b2"] for r in results), default=0),
        "max_nV": max((r["nV"] for r in results), default=0),
        "max_grown": max((r.get("n_grown", 0) for r in results), default=0),
        "max_add": max_add,
        "genuine_sizes": sorted(genuine_sizes.items()),
        "extensions": extensions, "new_gates": new_gates,
        "grows": bool(new_gates),
    }


# --------------------------------------------------------------------------- #
# Report rendering + main.
# --------------------------------------------------------------------------- #
def _print_header():
    print("Spectral gate realizability on the L_2 register (S^3 hexagon-join, "
          "surgery)\n  (the staged synthesis one degree up: tet holes open b_2 "
          "0 -> 2, ker L_2 carries the register, the same Hodge spectrum "
          "decides; the substrate the Regge-mediated objective selects on)\n")


def _emergence_and_anchor(reg, check):
    trace = register_emergence()
    print("  STAGE 3 register emergence (removeInteriorCell opens the three tet "
          "holes; ker L_2 emerges from the spectrum, boundary bit-exact):")
    print("      " + "  ->  ".join(
        f"{t['step']}: b_2={t['b2']}, ker L_2={t['kerL2']}" for t in trace))
    print(f"        => surgery grows b_2 0 -> {trace[-1]['b2']} and ker L_2 0 -> "
          f"{trace[-1]['kerL2']}; boundary-period constraint n ~ "
          f"{np.round(reg.n, 2)} (orientation signs {reg.sign}; symmetrized to "
          f"Sigma=0).")
    check("surgery grows b_2 0->2 on its own", trace[-1]["b2"] == 2)
    check("ker L_2 (the register) emerges 0->2 under surgery",
          [t["kerL2"] for t in trace] == [0, 0, 1, 2])
    check("carried register V is 2-dimensional", reg.dim == 2)
    check("the dual complex is a valid cell complex (the primal is a "
          "combinatorial manifold with boundary)", reg.dual_valid)

    cp_b = _CP_IN
    cp_a = np.asarray(BASE._gates()[1][1])[1:4, 1:4] @ _CP_IN  # psi_A = SWAP|psi_B>
    res_b, nv_b, nc_b = synthesize_state(reg, reg.sign * cp_b)
    res_a, nv_a, nc_a = synthesize_state(reg, reg.sign * cp_a)
    print("\n  STAGE 1 boundary synthesis (geo(psi) carried as a harmonic of "
          "ker L_2 on the grown S^3):")
    print(f"      geo(psi_B): |V|={nv_b} |C_2|={nc_b}  ||(I-PP)L_2 psi_B||^2 = "
          f"{res_b:.2e}  (carried)")
    print(f"      geo(psi_A): |V|={nv_a} |C_2|={nc_a}  ||(I-PP)L_2 psi_A||^2 = "
          f"{res_a:.2e}  (carried)")
    check("stage-1 geo(psi_B) carries psi_B as a harmonic", res_b < REALIZE)
    check("stage-1 geo(psi_A) carries psi_A as a harmonic", res_a < REALIZE)

    anchor = identity_anchor(reg)
    print("\n  Identity sanity check (the falsifiable core, decided spectrally):")
    for r in anchor:
        print(f"      {r['holes_open']} holes open: b_2={r['b2']} "
              f"ker L_2={r['kerL2']}  r={r['residual']:.2e}  "
              f"{'REALIZES' if r['realizable'] else 'floors'}")
    print("        => the identity FLOORS on every seed with ker L_2 < 2 and "
          "REALIZES only once surgery opens b_2 0 -> 2. Surgery is load-bearing "
          "at this degree exactly as at degree 1.")
    check("identity floors on every under-grown seed (ker L_2 < 2)",
          all((not r["realizable"]) and r["residual"] > CERT_FLOOR
              for r in anchor[:-1]))
    check("identity realizes once surgery grows the full register (b_2=2)",
          anchor[-1]["realizable"] and anchor[-1]["b2"] == 2)
    return trace, anchor, {"geo_psi_B": [res_b, nv_b, nc_b],
                           "geo_psi_A": [res_a, nv_a, nc_a]}


def _print_sweep(rows, check):
    realized = [r for r in rows if r["realizable"]]
    floored = [r for r in rows if not r["realizable"]]
    print(f"\n  STAGE 3 battery ({len(rows)} gates, the same battery as the 2d "
          f"register, scored by the L_2 spectrum):")
    print(f"      realized ({len(realized)}): "
          + ", ".join(r["gate"] for r in realized))
    lo = min(r["residual"] for r in floored)
    hi = max(r["residual"] for r in floored)
    lo_r = max(r["residual"] for r in realized)
    print(f"      floored ({len(floored)}): residuals in {lo:.2f}..{hi:.2f}; "
          f"worst realized residual {lo_r:.1e}")
    agree = all(r["realizable"] == BASE.conserves_charge(U)
                for r, (_n, U, _f) in zip(rows, BASE._gates()))
    check("the realizable set is EXACTLY the charge-conservation criterion "
          "(spectral == closed form on every gate)", agree)
    check("the realizable set matches the canonical 13 named gates",
          sorted(r["gate"] for r in realized) == sorted(CANONICAL_SET))
    check("every floored gate is certified by a nonzero leak",
          all(r["leak"] > 1e-6 for r in floored))
    return realized, floored


def _print_h3(rows, info, inv, check):
    realized = [r for r in rows if r["realizable"]]
    floored = [r for r in rows if not r["realizable"]]
    print("\n  H3 at the VALUE level on the L_2 register (one scale from the T1 "
          "anchor, then every number is a prediction):")
    print(f"      uniform cochain metric: w_2 = {info['metric_mean']:.6f} "
          f"(= sqrt(3)/4, the unit-edge triangle area; the anchor absorbs the "
          f"scale), max deviation {info['metric_uniform_dev']:.1e}")
    g = info["gram"]
    print(f"      register Gram on V: [[{g[0, 0]:.6f}, {g[0, 1]:.6f}], "
          f"[{g[1, 0]:.6f}, {g[1, 1]:.6f}]]  max|G - I| = {info['gram_dev']:.2e}")
    print("        (the hexagon join's hole triple is a full S_3 orbit of its "
          "automorphism group -- the isometric-chart proposition applies.)")
    header = (f"      {'gate':16} {'family':16} {'r(U)':>10} "
              f"{'max|Z_spec - amp|':>18} {'choi dev':>10} {'pairs':>6}")
    print(header)
    print("      " + "-" * (len(header) - 6))
    for r in realized:
        print(f"      {r['gate']:16} {r['family']:16} {r['residual']:>10.1e} "
              f"{r['max_dev']:>18.2e} {r['choi_dev']:>10.1e} {r['n_pairs']:>6}")
    lo = min(r["leak"] for r in floored)
    hi = max(r["leak"] for r in floored)
    print(f"      ({len(floored)} floored gates: no carried post-state -- leak "
          f"|Sigma| in {lo:.2f}..{hi:.2f} -- so no spectral value exists.)")
    eq = inv["equivariant"]
    print(f"      bulk independence ({inv['n_gates']} gates x {inv['n_pairs']} "
          f"pairs): the symmetric (9,9)-join variant (|V|={eq['nV']}) carries "
          f"the value EXACTLY -- Gram dev {eq['gram_dev']:.1e}, drift "
          f"{eq['drift']:.2e}.")
    for v in inv["anisotropic"]:
        print(f"        generic (8,8) hole draw (|V|={v['nV']}): Gram dev "
              f"{v['gram_dev']:.2e}, value deviation {v['drift']:.2e} -- equal "
              f"to the Gram-defect prediction a*(G-I)b to "
              f"{v['defect_residual']:.1e}.")
    worst = max(r["max_dev"] for r in realized)
    worst_choi = max(r["choi_dev"] for r in realized)
    print(f"        => Z_spec = <psi_A|U|psi_B> on every realized gate (worst "
          f"{worst:.2e}); the Choi/operator reading agrees (worst "
          f"{worst_choi:.2e}).")
    check("the register bulk has a uniform cochain metric (anchor absorbs the "
          "scale)", info["metric_uniform_dev"] < 1e-12)
    check("psi_B is carried (its periods lie in V)", info["psi_b_leak"] < 1e-9)
    check("the register Gram is the identity (period map is a scaled isometry)",
          info["gram_dev"] < 1e-9)
    check("T1 anchor: the identity's spectral value is <psi_A|psi_B> on every "
          "pair", realized[0]["gate"] == "Identity"
          and realized[0]["max_dev"] < 1e-9)
    check("H3: Z_spec = <psi_A|U|psi_B> for EVERY realized gate, on every pair",
          worst < 1e-9)
    check("the Choi/operator reading agrees with the flat amplitude",
          worst_choi < 1e-9)
    check("every floored gate has no carried post-state (leak != 0)",
          all(r["leak"] > 1e-6 for r in floored))
    check("the value carries over exactly to the symmetric (9,9)-join variant "
          "(bulk independence)", eq["gram_dev"] < 1e-9 and eq["drift"] < 1e-9)
    check("a generic hole draw's value deviation equals its register Gram "
          "defect", len(inv["anisotropic"]) >= 1
          and all(v["defect_residual"] < 1e-9 for v in inv["anisotropic"]))
    return worst


def _print_search(search, check):
    print(f"\n  Surgery-topology search at degree 2 ({search['scored']}/"
          f"{search['retries']} randomized surgery-grown S^3 topologies; seeds = "
          f"(p,q)-gon joins {dict(search['seeds'])}, max |V|={search['max_nV']}, "
          f"max b_2 grown = {search['max_b2']}; cuts AND additions -- up to "
          f"{search['max_add']} added vertices allowed, {search['max_grown']} "
          f"actually added in one draw):")
    print(f"      genuine registers (rank < #holes, S_3 anchor intact): "
          f"{search['n_genuine']}")
    print(f"      saturated registers (rank == #holes -- the register dissolves): "
          f"{search['n_saturated']}")
    if search["n_invalid"]:
        print(f"      invalid draws (S_3 anchor not met): {search['n_invalid']}")
    sizes = ", ".join(f"{n} gates x{c}" for n, c in search["genuine_sizes"])
    print(f"      genuine realizable-set sizes: {sizes or '(none)'}")
    if search["grows"]:
        print(f"        => the search GROWS the set beyond the criterion: "
              f"{', '.join(search['new_gates'])}.")
    else:
        print("        => NO genuine register carries any gate beyond the "
              "criterion set: the conservation law is dimension-free as well as "
              "topology-free.")
    print(f"      dual-complex validity: every accepted move is gated on the "
          f"dual staying a valid cell complex (combinatorial manifold with "
          f"boundary); draws whose final state violates it: "
          f"{search['n_dual_invalid']}")
    check("no genuine register carries a gate beyond the criterion set",
          not search["grows"])
    check("every genuine register realizes exactly the criterion set",
          all(n == len(CANONICAL_SET) for n, _c in search["genuine_sizes"]))
    check("every scored draw keeps a valid dual complex",
          search["n_dual_invalid"] == 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--gate", default=None,
                    help="score ONE gate by name ('--gate help' lists them)")
    ap.add_argument("--h3", action="store_true",
                    help="validate H3 at the value level on the L_2 register")
    ap.add_argument("--retries", type=int, default=0,
                    help="randomized surgery-topology search draws (0 = skip)")
    ap.add_argument("--max-additional-vertices", type=int, default=20,
                    help="additive-growth cap per search draw (default 20)")
    ap.add_argument("--jobs", type=int, default=min(10, os.cpu_count() or 1),
                    help="parallel workers for the search (procs x threads <= 10)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default="/tmp/cobordism",
                    help="directory for the raw JSON table")
    args = ap.parse_args()
    jobs = max(1, min(args.jobs, 10))

    checks = []

    def check(label, passed):
        checks.append((label, bool(passed)))
        return bool(passed)

    _print_header()
    prog = BASE._progress()
    prog.phase("growing the L_2 register + synthesizing states")
    reg = RegisterL2()
    trace, anchor, stage1 = _emergence_and_anchor(reg, check)
    prog.finish("register ready")

    if args.gate is not None:
        resolved = BASE.resolve_gate(args.gate)
        if args.gate.lower() == "help" or resolved is None:
            if args.gate.lower() != "help":
                print(f"  unknown gate '{args.gate}'.\n")
            print("  Available gates (--gate <name>, slug-insensitive):")
            for name, _U, fam in BASE._gates():
                print(f"      {name:16} [{fam}]")
            raise SystemExit(0 if args.gate.lower() == "help" else 2)
        name, U, fam = resolved
        res, b2, leak = post_interaction(reg, U)
        realized = bool(res < REALIZE)
        print(f"\n  Single-gate solve -- {name} [{fam}] on the L_2 register:")
        print(f"      residual r(U) = {res:.3e}   leak |Sigma| = {leak:.3f}   "
              f"emergent b_2 = {b2}")
        print(f"        => {name} {'REALIZES' if realized else 'FLOORS'}.")
        check(f"{name} verdict matches the charge-conservation criterion",
              realized == BASE.conserves_charge(U))
        ok = all(passed for _l, passed in checks)
        print("\n  Verdict: " + ("SUPPORTED" if ok else "NOT SUPPORTED"))
        raise SystemExit(0 if ok else 1)

    prog.phase("battery sweep", total=len(BASE._gates()))
    rows = gate_sweep(reg, on_progress=prog.on_tick)
    prog.finish("battery scored")
    realized, floored = _print_sweep(rows, check)

    h3_payload = None
    if args.h3:
        prog.phase("H3 value sweep", total=len(BASE._gates()))
        h3_rows, info = h3_value_sweep(reg, on_progress=prog.on_tick)
        prog.finish("values read")
        prog.phase("bulk independence", total=4)
        inv = h3_invariance(on_progress=prog.on_tick)
        prog.finish("variants surveyed")
        worst = _print_h3(h3_rows, info, inv, check)
        h3_payload = {"rows": h3_rows,
                      "gram_dev": info["gram_dev"],
                      "metric_mean": info["metric_mean"],
                      "metric_uniform_dev": info["metric_uniform_dev"],
                      "worst_dev": worst, "invariance": inv}

    search = None
    if args.retries > 0:
        prog.phase("surgery-topology search", total=args.retries)
        search = surgery_search(args.retries, jobs, base_seed=args.seed,
                                max_add=args.max_additional_vertices,
                                on_progress=prog.on_tick)
        prog.finish("search scored")
        _print_search(search, check)

    if not args.no_write:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, "l2_register_realizability.json")
        with open(path, "w") as handle:
            json.dump({"register_trace": trace,
                       "register_constraint": reg.n.tolist(),
                       "identity_anchor": anchor, "stage1": stage1,
                       "gate_sweep": rows, "h3": h3_payload,
                       "surgery_search": search}, handle, indent=2)
        print(f"\n  raw table (PR artifact, not committed): {path}")

    ok = all(passed for _label, passed in checks)
    if not ok:
        print("\n  FAILED checks:")
        for label, passed in checks:
            if not passed:
                print(f"      - {label}")
    print("\n  Verdict: " + (
        "SUPPORTED -- the staged spectral synthesis on the S^3 hexagon-join "
        "register realizes exactly the charge-conservation criterion set, one "
        "degree up: tet surgery grows ker L_2 0 -> 2, the same 13 named gates "
        "realize at machine zero, and the spectral test agrees with the closed "
        "form on every gate. The register theorems are dimension-free, and this "
        "bulk is the substrate the Regge-mediated objective selects on."
        if ok else
        "NOT SUPPORTED -- a claim failed; inspect the FAILED checks above."))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
