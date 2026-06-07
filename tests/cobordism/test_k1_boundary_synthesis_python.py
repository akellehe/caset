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

"""k=1 boundary-harmonic realizability synthesis on a 3-manifold-with-boundary (#176).

The v0.3 synthesis (`EigenstateSynthesis` / `RealizabilityOracle`) matched a k=0
boundary eigenvector of L = D - A on a 2-complex. The DW bridge (#174) lives in
the 3-manifold / k=1 setting: the spectral boundary qubit is the harmonic 1-forms
ker L_1(Sigma) of Sigma = dW (spec §5.2), prepared into the DW state space Z(Sigma)
by #175. This lifts the synthesis to k=1:

* `EigenstateSynthesis(W, degree=1)` scores a 1-form psi (length |C_1(W)|, the
  HodgeLaplacian/ChainComplex column order) by the HARMONIC residual
  r = ||L_1 psi||^2 (the metric Hodge Laplacian, built from the live simplex
  volumes; r = 0 iff psi in ker L_1). `boundaryStateIndices()` are the dW edges,
  the boundary support a target boundary harmonic occupies; the complement are the
  free interior amplitudes the fill solves for.
* `RealizabilityOracle.decideBoundaryHarmonic(target)` decides whether a target
  k=1 boundary harmonic extends to a harmonic of the bulk ker L_1(W) with dW
  pinned: realizable iff the harmonic residual is driven below epsilon, otherwise
  certified non-realizable by the obstruction floor (the v0.3 floor semantics).

Fixtures: the solid torus S^1 x D^2 (dW = T^2, the single-boundary realizability
verdict) and the thickened torus T^2 x I (genuine interior edges — the engine's
interior-completion demonstration).
"""

import unittest

import numpy as np

import tessera

cob = tessera.cobordism


# --------------------------------------------------------------------------- #
# Fixtures (the bulk-synthesis idiom: Signature(d) so d-cells register as top
# simplices; a uniform spacelike metric so the metric L_1 is a clean Euclidean
# Hodge Laplacian — ker L_1 = H_1).
# --------------------------------------------------------------------------- #
def _build(topology):
    sig = tessera.Signature(topology.dimension(), tessera.Lorentzian)
    st = tessera.Spacetime(tessera.Metric(True, sig), tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, topology)
    st.build()
    for e in st.getEdgeList().toVector():
        e.setSquaredLength(1.0)
        e.setPhase(0.0)
    return st


def _circle():
    return tessera.SimplexBoundarySphere(1)  # S^1 = boundary of a triangle


def _solid_torus():
    # S^1 x D^2: a 3-manifold with boundary dW = T^2 (one boundary component).
    return _build(tessera.SimplicialProduct(_circle(), tessera.SolidSimplex(2)))


def _thick_torus():
    # T^2 x I: a thickened surface with genuine interior edges; dW = T^2 ⊔ T^2.
    return _build(tessera.SimplicialProduct(
        tessera.SimplicialProduct(_circle(), _circle()), tessera.SolidSimplex(1)))


def _cvec(v):
    return [complex(z) for z in v]


def _bulk_harmonic(st):
    """The first bulk harmonic 1-form ker L_1(W) as a numpy vector (|C_1|)."""
    harmonics = cob.HodgeLaplacian(st).harmonics(1)
    assert harmonics, "fixture has trivial H_1; expected a bulk harmonic"
    return np.asarray(harmonics[0].coeffs())


# --------------------------------------------------------------------------- #
# The k=1 EigenstateSynthesis engine: the harmonic residual + indexing.
# --------------------------------------------------------------------------- #
class TestK1EngineResidual(unittest.TestCase):

    def test_degree_and_dimension(self):
        st = _solid_torus()
        es = cob.EigenstateSynthesis(st, 1)
        self.assertEqual(es.degree(), 1)
        # |C_1(W)| = the number of edges = the k=1 operator dimension.
        n_edges = cob.ChainComplex.fromSpacetime(st).numSimplices(1)
        self.assertEqual(es.dimension(), n_edges)
        # k=0 still reports the vertex count (backward compatible).
        self.assertEqual(cob.EigenstateSynthesis(st, 0).dimension(),
                         cob.ChainComplex.fromSpacetime(st).numSimplices(0))

    def test_state_index_partition(self):
        es = cob.EigenstateSynthesis(_solid_torus(), 1)
        bnd = list(es.boundaryStateIndices())
        inter = list(es.interiorStateIndices())
        # The two index sets partition [0, dimension()).
        self.assertEqual(sorted(bnd + inter), list(range(es.dimension())))
        self.assertEqual(len(set(bnd) & set(inter)), 0)
        self.assertEqual(len(bnd), es.numBoundaryEdges())
        self.assertEqual(len(inter), es.numInteriorEdges())

    def test_bulk_harmonics_have_zero_residual(self):
        # A genuine ker L_1(W) form is harmonic: r = ||L_1 psi||^2 ~ 0, lambda ~ 0.
        for fixture in (_solid_torus, _thick_torus):
            with self.subTest(fixture=fixture.__name__):
                st = fixture()
                es = cob.EigenstateSynthesis(st, 1)
                for h in cob.HodgeLaplacian(st).harmonics(1):
                    psi = list(h.coeffs())
                    self.assertLess(es.residual(psi), 1e-12)
                    self.assertLess(abs(es.rayleigh(psi)), 1e-9)
                    lp = np.asarray(es.apply(psi))
                    self.assertLess(float(np.sum(np.abs(lp) ** 2)), 1e-12)

    def test_nonharmonic_eigenvector_residual_is_eigenvalue_squared(self):
        # For an eigenvector with eigenvalue lambda, r = ||L_1 v||^2 = lambda^2
        # (v unit) — nonzero away from the kernel, isolating ker L_1.
        st = _solid_torus()
        es = cob.EigenstateSynthesis(st, 1)
        spec = cob.HodgeLaplacian(st).spectrum(1)
        evals = np.asarray(spec.eigenvalues()).real
        j = int(np.argmax(evals))           # the largest, clearly non-harmonic
        self.assertGreater(evals[j], 1.0)
        r = es.residual(list(spec[j].coeffs()))
        self.assertAlmostEqual(r, evals[j] ** 2, delta=1e-6 * evals[j] ** 2)

    def test_residual_rejects_wrong_length(self):
        es = cob.EigenstateSynthesis(_solid_torus(), 1)
        with self.assertRaises((RuntimeError, ValueError)):
            es.residual([1.0, 0.0])  # not |C_1|

    def test_negative_degree_raises(self):
        with self.assertRaises((RuntimeError, ValueError)):
            cob.EigenstateSynthesis(_solid_torus(), -1)


# --------------------------------------------------------------------------- #
# The interior completion: a boundary harmonic extends only via the interior.
# --------------------------------------------------------------------------- #
class TestInteriorCompletion(unittest.TestCase):
    """On a fixture with genuine interior edges, a bulk harmonic's boundary block
    alone is NOT harmonic — the interior amplitudes (the fill's free parameters)
    are what complete it. This is the structure RealizabilityOracle.fillInterior
    solves for at k=1."""

    def test_boundary_block_needs_the_interior(self):
        st = _thick_torus()
        es = cob.EigenstateSynthesis(st, 1)
        inter = list(es.interiorStateIndices())
        self.assertGreater(len(inter), 0)          # genuine interior edges
        psi = _bulk_harmonic(st)

        # The full harmonic (boundary block + its interior completion) is harmonic.
        self.assertLess(es.residual(list(psi)), 1e-12)
        # The harmonic genuinely lives on the interior (nonzero there).
        self.assertGreater(float(np.sum(np.abs(psi[inter]) ** 2)), 1e-6)
        # Zeroing the interior (keeping only the boundary block) breaks harmonicity:
        # the boundary block alone is far from ker L_1 — the fill must restore it.
        boundary_only = psi.copy()
        boundary_only[inter] = 0.0
        self.assertGreater(es.residual(list(boundary_only)), 1e-3)


# --------------------------------------------------------------------------- #
# The k=1 realizability oracle: verdict + obstruction floor on the solid torus.
# --------------------------------------------------------------------------- #
class TestK1RealizabilityVerdict(unittest.TestCase):

    def _target_from_bulk_harmonic(self, st, es):
        """The bulk harmonic's boundary block, ordered as boundaryStateIndices()."""
        psi = _bulk_harmonic(st)
        return [complex(psi[i]) for i in es.boundaryStateIndices()]

    def test_realizable_boundary_harmonic(self):
        # The boundary block of a genuine bulk harmonic ker L_1(W) is realizable:
        # it extends to a bulk harmonic with dW pinned, so r -> 0.
        st = _solid_torus()
        es = cob.EigenstateSynthesis(st, 1)
        target = self._target_from_bulk_harmonic(st, es)
        v = cob.RealizabilityOracle(st).decideBoundaryHarmonic(
            _cvec(target), epsilon=1e-9, restarts=8, max_cones=0, seed=1)

        self.assertTrue(v.realizable)
        self.assertLess(v.residual, 1e-9)
        self.assertEqual(v.floor, 0.0)
        self.assertLess(abs(v.eigenvalue), 1e-7)   # harmonic: lambda ~ 0
        # The witness is the realized bulk 1-form (length |C_1(W)|); its dW
        # boundary block matches the target (up to global phase/scale).
        self.assertEqual(len(v.state), es.dimension())
        block = np.asarray(v.state)[list(es.boundaryStateIndices())]
        tgt = np.asarray(target)
        overlap = abs(np.vdot(block / np.linalg.norm(block),
                              tgt / np.linalg.norm(tgt)))
        self.assertAlmostEqual(overlap, 1.0, places=8)

    def test_obstructed_boundary_form_floors(self):
        # A generic boundary 1-form is not a harmonic boundary block: it cannot be
        # extended to a bulk harmonic with dW pinned, so r floors away from 0 —
        # the obstruction floor IS the non-realizability certificate.
        st = _solid_torus()
        es = cob.EigenstateSynthesis(st, 1)
        n = es.numBoundaryEdges()
        rng = np.random.default_rng(0)
        target = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        v = cob.RealizabilityOracle(st).decideBoundaryHarmonic(
            _cvec(target), epsilon=1e-9, restarts=8, max_cones=0, seed=1)

        self.assertFalse(v.realizable)
        self.assertGreater(v.residual, 1e-2)
        self.assertEqual(v.floor, v.residual)

    def test_floor_is_seed_independent(self):
        # The certified floor is the genuine obstruction, not a seed-specific local
        # min: independent restart seeds reach the same floor.
        st = _solid_torus()
        n = cob.EigenstateSynthesis(st, 1).numBoundaryEdges()
        rng = np.random.default_rng(7)
        target = _cvec(rng.standard_normal(n) + 1j * rng.standard_normal(n))
        floors = []
        for s in (0, 13, 41):
            v = cob.RealizabilityOracle(st).decideBoundaryHarmonic(
                target, epsilon=1e-9, restarts=8, max_cones=0, seed=s)
            self.assertFalse(v.realizable)
            floors.append(v.floor)
        for f in floors[1:]:
            self.assertAlmostEqual(f, floors[0], delta=1e-6)

    def test_obstructed_with_growth_still_floors_after_coning(self):
        # The fixed-boundary cone-and-retry runs at k=1: an obstructed target grows
        # the interior (boundary-fixed Pachner) but still cannot be realized — the
        # floor certificate survives interior growth.
        st = _solid_torus()
        n = cob.EigenstateSynthesis(st, 1).numBoundaryEdges()
        rng = np.random.default_rng(3)
        target = _cvec(rng.standard_normal(n) + 1j * rng.standard_normal(n))
        v = cob.RealizabilityOracle(st).decideBoundaryHarmonic(
            target, epsilon=1e-9, restarts=8, max_cones=2, seed=1)
        self.assertFalse(v.realizable)
        self.assertGreater(v.floor, 1e-3)
        self.assertGreaterEqual(v.cones_applied, 1)   # growth was attempted

    def test_determinism(self):
        st = _solid_torus()
        es = cob.EigenstateSynthesis(st, 1)
        target = self._target_from_bulk_harmonic(st, es)
        a = cob.RealizabilityOracle(st).decideBoundaryHarmonic(
            _cvec(target), epsilon=1e-9, restarts=8, max_cones=0, seed=5)
        b = cob.RealizabilityOracle(st).decideBoundaryHarmonic(
            _cvec(target), epsilon=1e-9, restarts=8, max_cones=0, seed=5)
        self.assertEqual(a.realizable, b.realizable)
        self.assertEqual(a.residual, b.residual)
        np.testing.assert_array_equal(np.asarray(a.state), np.asarray(b.state))

    def test_target_wrong_length_raises(self):
        st = _solid_torus()
        with self.assertRaises((ValueError, RuntimeError)):
            cob.RealizabilityOracle(st).decideBoundaryHarmonic(
                [1.0 + 0j, 0.0 + 0j], max_cones=0)


if __name__ == "__main__":
    unittest.main()
