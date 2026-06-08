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

"""Free interior connectivity extended to k=1: the triangle (2-simplex) search,
and the precise wall it certifies.

#201 made interior connectivity a free variable, but its candidate generator
proposes only **edges** (singleton specs) — correct at k=0, where the Hodge
Laplacian ``L_0 = D - A`` is assembled from the 1-skeleton alone, so 2-cells are
spectrally invisible. At k=1 the metric Hodge Laplacian ``L_1 = d_1^T d_1 +
d_2 d_2^T`` depends on the **2-simplices** through ``d_2``. This suite extends the
search to ALSO propose 2-simplex (triangle) attachments at k>=1 and pins down,
rigorously, what that does and does not buy:

  1. **2-cells are spectrally visible at k=1, invisible at k=0** (the principle).
     On a 2-complex, filling a triangle {0,1,2} (adding the 2-cell over the SAME
     three edges) shifts the k=1 residual but leaves the k=0 residual bit-exact —
     a clean isolation of d_2's effect, numpy-cross-checked.

  2. **On a pure d-complex the additive attach is spectrally inert at k>=1** (the
     mechanism). ``ChainComplex.fromSpacetime`` seeds its BFS from the TOP cells
     and takes their downward closure, so a dangling edge/triangle attached by
     ``attachInteriorVertex`` is dropped from ``L_k`` entirely: the operator
     dimension does not even change.

  3. **The spectrally-active move is boundary-locked at k>=1** (the lock). An
     additive *top-cell* (tetrahedron) attach would register, but it introduces
     new boundary edges incident to the new vertex, which the bit-exact ∂W guard
     rejects. So no additive move is BOTH spectrally active AND boundary-fixed at
     k>=1 — only the stellar Pachner subdivision (``growInterior``, a replace move)
     is.

  4. **The k=1 search therefore proposes triangles (bounded, logged, surfaced in
     the Verdict) but commits the Pachner cone** (the honest consequence).
     ``decideHarmonic(growth_mode=FREE_CONNECTIVITY)`` scores
     ``triangle_candidates > 0`` per growth step (0 at k=0 / cone), finds them
     inert, and falls back to the boundary-fixed cone — so the realized residual
     EQUALS cone growth's, ∂W stays bit-exact, and additive free connectivity does
     NOT enlarge the realizable set at k=1 (the opposite of the k=0 headline).
"""

import unittest

import numpy as np

import tessera

cob = tessera.cobordism
FREE = cob.RealizabilityOracle.GrowthMode.FREE_CONNECTIVITY
CONE = cob.RealizabilityOracle.GrowthMode.CONE


# --------------------------------------------------------------------------- #
# Fixtures.
# --------------------------------------------------------------------------- #
def _st(dim, topology=None):
    sig = tessera.Signature(dim, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    return tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED,
                             topology)


def _hollow_triangle():
    """Vertices 0,1,2 with the three edges 01,02,12 but NO 2-cell (a 1-cycle).
    A 2-complex stage (Signature 2) whose only difference from `_filled_triangle`
    is the absent 2-cell."""
    st = _st(2)
    v = [st.createVertex(i) for i in range(3)]
    st.createSimplex([v[0], v[1]])
    st.createSimplex([v[0], v[2]])
    st.createSimplex([v[1], v[2]])
    return st


def _filled_triangle():
    """The SAME 1-skeleton 01,02,12 PLUS the 2-cell {0,1,2}: createSimplex of the
    triangle materializes its three edges and the filling 2-simplex."""
    st = _st(2)
    v = [st.createVertex(i) for i in range(3)]
    st.createSimplex([v[0], v[1], v[2]])
    return st


def _build(topology):
    sig = tessera.Signature(topology.dimension(), tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, topology)
    st.build()
    return st


def _circle():
    return tessera.SimplexBoundarySphere(1)


def _solid_torus():
    """W = D^2 x S^1: a 3-manifold with boundary T^2 (b_1(W) = 1), pure (every
    cell is a face of a tetrahedron)."""
    return _build(tessera.SimplicialProduct(tessera.SolidSimplex(2), _circle()))


def _torus():
    return _build(tessera.SimplicialProduct(_circle(), _circle()))


def _pin_uniform(st, w=1.0, phase=0.0):
    for e in st.getEdgeList().toVector():
        e.setSquaredLength(w)
        e.setPhase(phase)


def _cvec(v):
    return [complex(z) for z in v]


def _edge_keys(st):
    out = set()
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        out.add((min(a, b), max(a, b)))
    return out


def _boundary_snapshot(st):
    return {(min(a, b), max(a, b)): (e.getSquaredLength(), e.getPhase())
            for e in st.getEdgeList().toVector()
            for a, b in [(e.getSource().getId(), e.getTarget().getId())]}


def _num_cells(st, k):
    return cob.ChainComplex.fromSpacetime(st).numSimplices(k)


# --------------------------------------------------------------------------- #
# numpy Hodge oracle for L_1 (symmetric metric Laplacian), the #176 assembly.
# --------------------------------------------------------------------------- #
def _numpy_L1(st):
    chain = cob.ChainComplex.fromSpacetime(st)
    nv, ne, nt = (chain.numSimplices(0), chain.numSimplices(1),
                  chain.numSimplices(2))
    d1 = np.asarray(chain.boundaryMatrix(1), float).reshape(nv, ne)
    hodge = cob.HodgeLaplacian(st)
    w1 = np.asarray(hodge.weights(1), float)
    b1 = d1 * (1.0 / np.sqrt(w1))[None, :]               # W_0 = I
    L = b1.T @ b1
    if nt > 0:
        d2 = np.asarray(chain.boundaryMatrix(2), float).reshape(ne, nt)
        w2 = np.asarray(hodge.weights(2), float)
        b2 = np.sqrt(w1)[:, None] * d2 * (1.0 / np.sqrt(w2))[None, :]
        L = L + b2 @ b2.T
    return L


def _residual_agnostic(L, psi):
    psi = np.asarray(psi, dtype=complex)
    psi = psi / np.linalg.norm(psi)
    Lp = L @ psi
    lam = np.vdot(psi, Lp).real
    return float(np.vdot(Lp - lam * psi, Lp - lam * psi).real)


# --------------------------------------------------------------------------- #
class TriangleSpectralVisibilityTest(unittest.TestCase):
    """(1) The principle: a 2-cell is spectrally visible at k=1, invisible at k=0,
    with the 1-skeleton held identical (so it is the 2-cell, not the edges). The
    two complexes share the edge set 01,02,12; the filled one adds the 2-cell
    {0,1,2}, which (being a top cell) registers in ChainComplex."""

    def _residual(self, st, k, seed=1):
        _pin_uniform(st)
        es = cob.EigenstateSynthesis(st, k)
        rng = np.random.default_rng(seed)
        psi = rng.standard_normal(es.order()) + 1j * rng.standard_normal(es.order())
        return es.residual(_cvec(psi)), es.order(), \
            [tuple(c) for c in es.cellSimplices()]

    def test_same_one_skeleton_one_extra_two_cell(self):
        self.assertEqual(_edge_keys(_hollow_triangle()),
                         _edge_keys(_filled_triangle()))
        self.assertEqual(_num_cells(_hollow_triangle(), 2), 0)
        self.assertEqual(_num_cells(_filled_triangle(), 2), 1)

    def test_two_cell_changes_k1_residual(self):
        r_hollow, o_h, c_h = self._residual(_hollow_triangle(), 1)
        r_filled, o_f, c_f = self._residual(_filled_triangle(), 1)
        self.assertEqual((o_h, c_h), (o_f, c_f))     # identical k=1 operator basis
        self.assertEqual(o_h, 3)                      # three edges
        # d_2 enters L_1 only in the filled complex: a finite residual shift.
        self.assertGreater(abs(r_hollow - r_filled), 1e-6)

    def test_two_cell_does_not_change_k0_residual(self):
        # L_0 = D - A has no d_2 term: same vertices + same edges => identical
        # operator, so the 2-cell is bit-exact invisible at k=0.
        r_hollow, o_h, _ = self._residual(_hollow_triangle(), 0)
        r_filled, o_f, _ = self._residual(_filled_triangle(), 0)
        self.assertEqual(o_h, o_f)
        self.assertAlmostEqual(r_hollow, r_filled, places=12)

    def test_k1_residual_matches_numpy_hodge(self):
        for factory in (_hollow_triangle, _filled_triangle):
            st = factory()
            _pin_uniform(st)
            es = cob.EigenstateSynthesis(st, 1)
            rng = np.random.default_rng(7)
            psi = rng.standard_normal(es.order()) + 1j * rng.standard_normal(es.order())
            self.assertAlmostEqual(es.residual(_cvec(psi)),
                                   _residual_agnostic(_numpy_L1(st), psi), places=8)


# --------------------------------------------------------------------------- #
class AdditiveAttachIsInertAtK1Test(unittest.TestCase):
    """(2)+(3) The mechanism + the lock: on a pure 3-complex the additive attach
    cannot do spectrally-active, boundary-fixed growth at k>=1."""

    def test_dangling_triangle_is_dropped_by_chaincomplex(self):
        # A triangle coned over an existing edge by attachInteriorVertex is NOT a
        # face of any tetrahedron, so ChainComplex (which closes downward from the
        # top cells) drops it: the operator dimension does not change.
        W = _solid_torus()
        _pin_uniform(W)
        es = cob.EigenstateSynthesis(W, 1)
        order0, two0 = es.order(), _num_cells(W, 2)
        edge = es.boundaryEdges()[0]
        self.assertTrue(es.attachInteriorVertex([[int(edge[0]), int(edge[1])]]))
        # The attach succeeded (boundary-safe: a 2-cell adds no tetrahedron), but
        # L_1 is blind to it — same operator dimension, same 2-cell count.
        self.assertEqual(es.order(), order0)
        self.assertEqual(_num_cells(W, 2), two0)

    def test_dangling_triangle_leaves_k1_residual_bit_exact(self):
        # Direct spectral confirmation of inertness: the residual of any fixed psi
        # is unchanged by the dangling triangle (it is the same L_1 operator).
        W = _solid_torus()
        _pin_uniform(W)
        es = cob.EigenstateSynthesis(W, 1)
        rng = np.random.default_rng(11)
        psi = _cvec(rng.standard_normal(es.order())
                    + 1j * rng.standard_normal(es.order()))
        r0 = es.residual(psi)
        edge = es.boundaryEdges()[3]
        es.attachInteriorVertex([[int(edge[0]), int(edge[1])]])
        self.assertEqual(es.residual(psi), r0)        # bit-exact: inert

    def test_additive_top_cell_attach_is_boundary_locked(self):
        # The spectrally-active move (cone the new vertex over an existing triangle
        # to form a TETRAHEDRON) is rejected: it would give the new vertex's faces
        # boundary status, introducing boundary edges not in the pinned ∂W.
        W = _solid_torus()
        _pin_uniform(W)
        es = cob.EigenstateSynthesis(W, 1)
        triangle = list(cob.ChainComplex.fromSpacetime(W).kSimplexVertices(2)[0])
        order0, edges0 = es.order(), _edge_keys(W)
        self.assertFalse(es.attachInteriorVertex([[int(x) for x in triangle]]))
        self.assertEqual(es.order(), order0)          # rolled back
        self.assertEqual(_edge_keys(W), edges0)

    def test_pachner_cone_is_the_one_active_boundary_fixed_move(self):
        # By contrast the stellar Pachner subdivision (growInterior) DOES enrich
        # L_1 (operator dimension grows by 4 edges) while pinning ∂W byte-fixed —
        # the move the k>=1 free search must fall back to.
        W = _solid_torus()
        _pin_uniform(W)
        es = cob.EigenstateSynthesis(W, 1)
        order0 = es.order()
        boundary0 = _boundary_snapshot(W)
        self.assertTrue(es.growInterior(7))
        self.assertEqual(es.order(), order0 + 4)
        after = _boundary_snapshot(W)
        for k in es.boundaryEdges():
            key = (min(k), max(k))
            self.assertEqual(after[key], boundary0[key])


# --------------------------------------------------------------------------- #
class K1FreeConnectivitySearchTest(unittest.TestCase):
    """(4) The consequence: the k=1 search proposes triangles (bounded, logged,
    surfaced), finds them inert, and falls back to the Pachner cone — so free
    connectivity EQUALS cone at k=1 (no enlargement of the realizable set)."""

    def _floored_target(self, W):
        """A boundary 1-form on Sigma = dW that floors at the bare seed: a basis
        harmonic of ker L_1(Sigma) (not the carried bulk harmonic), which
        decideHarmonic certifies non-realizable at the seed (#176)."""
        return cob.BoundaryStateSpace(_torus()).harmonics()[0]

    def test_k1_search_scores_triangle_candidates(self):
        W = _solid_torus()
        _pin_uniform(W)
        v = cob.RealizabilityOracle(W).decideHarmonic(
            self._floored_target(W), epsilon=1e-9, restarts=4, max_cones=1, seed=1,
            growth_mode=FREE, connectivity_candidates=8)
        self.assertGreater(v.triangle_candidates, 0)       # 2-simplices proposed
        self.assertLessEqual(v.triangle_candidates, 8)     # bounded by the cap
        self.assertGreater(v.connectivity_candidates, 0)   # edge fans too

    def test_cone_and_k0_report_zero_triangle_candidates(self):
        W = _solid_torus()
        _pin_uniform(W)
        cone = cob.RealizabilityOracle(W).decideHarmonic(
            self._floored_target(W), epsilon=1e-9, restarts=4, max_cones=1, seed=1,
            growth_mode=CONE)
        self.assertEqual(cone.triangle_candidates, 0)

        st = _hollow_triangle()
        _pin_uniform(st)
        U = [[1.0 + 0j, 2.0 + 0j]]
        k0 = cob.RealizabilityOracle(st).decide(
            _cvec(np.asarray(U, dtype=complex).reshape(-1)), 1, 2, epsilon=1e-10,
            restarts=4, max_cones=1, seed=0, growth_mode=FREE,
            connectivity_candidates=8)
        self.assertEqual(k0.triangle_candidates, 0)        # invisible at k=0

    def test_k1_free_grows_by_the_pachner_signature_boundary_fixed(self):
        # The additive arm being inert, the free step falls back to the stellar
        # Pachner subdivision: the witness grows by the Pachner signature (+4 edges
        # per cone — a dangling additive attach would be +0), ∂W stays byte-fixed,
        # and the disk-bounding target still floors (NOT realized). The exact
        # residual is not reproducible across calls (pre-existing AddMove global-
        # counter non-determinism, #201 — out of scope), so the structure is what
        # is asserted, not the residual value.
        W = _solid_torus()
        _pin_uniform(W)
        boundary0 = _boundary_snapshot(W)
        v = cob.RealizabilityOracle(W).decideHarmonic(
            self._floored_target(W), epsilon=1e-9, restarts=4, max_cones=1, seed=3,
            growth_mode=FREE, connectivity_candidates=8)

        self.assertGreaterEqual(v.cones_applied, 1)
        self.assertEqual(len(v.state), 27 + 4 * v.cones_applied)   # Pachner, not dangling
        self.assertFalse(v.realizable)                            # still floors
        self.assertGreater(v.residual, 1e-2)                     # far from epsilon
        # ∂W byte-fixed through the (Pachner) growth.
        after = _boundary_snapshot(W)
        es = cob.EigenstateSynthesis(W, 1)
        for k in es.boundaryEdges():
            key = (min(k), max(k))
            self.assertEqual(after[key], boundary0[key])

    def test_k1_free_equals_cone_without_a_growth_budget(self):
        # Without a growth budget the free path adds nothing (no growInterior, so no
        # global-counter non-determinism): free == cone EXACTLY, deterministically.
        # This is the clean statement that the free search does not perturb the
        # fixed-complex optimization — its only k>=1 effect is the (cone) Pachner.
        target = self._floored_target(_solid_torus())
        Wc = _solid_torus(); _pin_uniform(Wc)
        cone = cob.RealizabilityOracle(Wc).decideHarmonic(
            target, epsilon=1e-9, restarts=4, max_cones=0, seed=3, growth_mode=CONE)
        Wf = _solid_torus(); _pin_uniform(Wf)
        free = cob.RealizabilityOracle(Wf).decideHarmonic(
            target, epsilon=1e-9, restarts=4, max_cones=0, seed=3,
            growth_mode=FREE, connectivity_candidates=8)
        self.assertEqual(free.residual, cone.residual)
        self.assertEqual(free.realizable, cone.realizable)
        np.testing.assert_array_equal(np.asarray(free.state), np.asarray(cone.state))

    def test_k1_free_does_not_enlarge_the_realizable_set(self):
        # The disk-bounding ker-L_1(Sigma) basis harmonics each floor at the seed —
        # certified non-realizable — under BOTH free and cone (free's only growth is
        # the cone Pachner, so the realizable set is unchanged). Shown at the seed
        # (max_cones=0, deterministic) so the verdict is exact and reproducible.
        space = cob.BoundaryStateSpace(_torus())
        for i in range(space.harmonicDimension()):
            Wc = _solid_torus(); _pin_uniform(Wc)
            cone = cob.RealizabilityOracle(Wc).decideHarmonic(
                space.harmonics()[i], epsilon=1e-9, restarts=4, max_cones=0, seed=i)
            Wf = _solid_torus(); _pin_uniform(Wf)
            free = cob.RealizabilityOracle(Wf).decideHarmonic(
                space.harmonics()[i], epsilon=1e-9, restarts=4, max_cones=0, seed=i,
                growth_mode=FREE, connectivity_candidates=8)
            self.assertFalse(cone.realizable)
            self.assertFalse(free.realizable)
            self.assertEqual(free.residual, cone.residual)


if __name__ == "__main__":
    unittest.main()
