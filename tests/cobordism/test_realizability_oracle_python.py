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

"""§5.0 realizability oracle — spectral bulk synthesis for U (#138).

`RealizabilityOracle` decides whether an operation U : H_B -> H_A is realizable as
a bulk cobordism W_AB by *synthesizing the bulk spectrally*, not by TQFT
membership. U is bent to a boundary state via Choi-Jamiolkowski (vec(U), the
operator-as-state); the bulk's boundary dW is *pinned* (the synthesized geo's /
the output surface) and the interior is filled — its Hermitian edge weights and,
via the boundary-fixed Pachner add (#112), its topology — so the output-boundary
k=0 graph-Laplacian eigenvector matches the bent target, driving the §4b residual
r = ||(I - psi psi^dagger) L psi||^2 to 0.

The oracle is pure orchestration of merged classes: ChoiJamiolkowski (the bend),
EigenstateSynthesis (the fixed-boundary interior-fill engine #147), and
LevenbergMarquardt (the same multi-restart least-squares solver the §4b
GeometrySynthesizer drives, here over the interior parameters + free auxiliary
amplitudes).

Acceptance (#138):

  * **Realizable U realized with its W_AB (r < 1e-10).** A uniform U (vec(U) the
    constant vector — the exact zero mode of L = D - A for a phase-0 boundary) is
    realized at the minimal interior complexity: r < 1e-10, the witness's output-
    boundary block is proportional to vec(U), and the witness is a genuine
    Laplacian eigenstate (L psi parallel to psi). A second realizable U is matched
    only after the fixed-boundary cone-and-retry grows the interior (cones > 0).

  * **Obstructed U certified non-realizable by a residual floor.** A generic U on
    a one-interior-edge bulk floors at a positive residual bounded away from 0
    (the analogue of §4b's two-vertex floor), cross-checked against an independent
    numpy global-min oracle; the verdict is non-realizable.

  * **Determinism.** The seeded synthesis is reproducible.
"""

import math
import unittest

import numpy as np

import tessera

cob = tessera.cobordism


# --------------------------------------------------------------------------- #
# Fixtures (the #147 bulk-synthesis idiom: Signature(d) so the d-cells register
# as top simplices; built through the topology so the vertex-id counter advances).
# --------------------------------------------------------------------------- #
def _spacetime(dim, topology):
    sig = tessera.Signature(dim, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    return tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                             tessera.PREFERRED, topology)


def _solid_triangle():
    """Delta^2: a single triangle 0-1-2 whose boundary dW is the three sides
    (S^1); 0 interior edges (every edge is boundary)."""
    st = _spacetime(2, tessera.SolidSimplex(2))
    st.build()
    return st


def _bipyramid():
    """Two triangles 012 and 013 sharing the interior edge 01; the four outer
    edges 02,12,03,13 are the pinned boundary dW. The smallest fixed-boundary
    complex with exactly one interior parameter — the §5.0 analogue of §4b's
    two-vertex single edge, with the boundary pinned."""
    st = _solid_triangle()            # triangle 012, vertex-id counter at 3
    v = {x.getId(): x for x in st.getVertexList().toVector()}
    v3 = st.createVertex(3)
    st.createSimplex([v[0], v[1], v3])  # triangle 013
    return st


# --------------------------------------------------------------------------- #
# numpy oracle + helpers (the D - A magnitude convention, sorted-vertex-id order).
# --------------------------------------------------------------------------- #
def _cvec(v):
    return [complex(z) for z in v]


def _edge_map(st):
    out = {}
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        out[(min(a, b), max(a, b))] = e
    return out


def _pin_all(st, w=1.0, phase=0.0):
    """Pin every edge to a fixed Hermitian value. The oracle's fill only ever
    rewrites the interior edges, so this fixes dW; the interior values are merely
    a starting point the fill overwrites."""
    for e in st.getEdgeList().toVector():
        e.setSquaredLength(w)
        e.setPhase(phase)


def _np_L(st):
    ids = sorted(v.getId() for v in st.getVertexList().toVector())
    idx = {vid: i for i, vid in enumerate(ids)}
    n = len(ids)
    A = np.zeros((n, n), dtype=complex)
    D = np.zeros(n)
    for e in st.getEdgeList().toVector():
        s, t = e.getSource().getId(), e.getTarget().getId()
        if s == t:
            continue
        i, j = idx[s], idx[t]
        w = e.getSquaredLength().real
        z = w * np.exp(1j * e.getPhase())
        A[i, j] += z
        A[j, i] += np.conj(z)
        D[i] += abs(w)
        D[j] += abs(w)
    return np.diag(D).astype(complex) - A


def _np_residual(L, psi):
    psi = np.asarray(psi, dtype=complex)
    psi = psi / np.linalg.norm(psi)
    Lp = L @ psi
    lam = np.vdot(psi, Lp).real
    r = Lp - lam * psi
    return float(np.vdot(r, r).real)


def _vec(U):
    """Row-major vec(U) — the Choi bend; matches ChoiJamiolkowski.vectorize."""
    return np.asarray(U, dtype=complex).reshape(-1)


# --------------------------------------------------------------------------- #
class RealizableTest(unittest.TestCase):
    """A realizable U is realized with its W_AB: r < 1e-10, the output-boundary
    eigenvector matches vec(U), and the witness is a genuine eigenstate."""

    def test_uniform_operator_is_realized_at_minimal_complexity(self):
        # U = [[1,1],[1,1]] (= 2|+><+|, a rank-1 transition operation): vec(U) is
        # the constant vector, the exact zero mode of L = D - A for any phase-0
        # positive-weight boundary. Realizable with its W_AB at minimal interior
        # complexity — the fill only has to drive the lone interior edge to phase 0.
        st = _bipyramid()
        _pin_all(st, w=1.0, phase=0.0)
        boundary_before = {k: (e.getSquaredLength().real, e.getPhase())
                           for k, e in _edge_map(st).items()
                           if k != (0, 1)}

        U = [[1.0 + 0j, 1.0 + 0j], [1.0 + 0j, 1.0 + 0j]]
        oracle = cob.RealizabilityOracle(st)
        v = oracle.decide(_cvec(_vec(U)), 2, 2, epsilon=1e-10, restarts=48,
                          max_cones=0, seed=1)

        # Realizable, with r driven below the acceptance threshold.
        self.assertTrue(v.realizable)
        self.assertLess(v.residual, 1e-10)
        self.assertEqual(v.floor, 0.0)
        self.assertEqual(v.interior_vertex_count, 0)   # minimal W_AB: no growth

        # The bent target is vec(U) normalized.
        target = _vec(U) / np.linalg.norm(_vec(U))
        np.testing.assert_allclose(np.asarray(v.target), target, atol=1e-12)

        # The witness's output-boundary block (first dA*dB = 4 components) is
        # proportional to vec(U): the output-boundary eigenvector matches it.
        state = np.asarray(v.state)
        self.assertEqual(state.shape, (4,))            # no auxiliary vertices
        block = state[:4]
        overlap = abs(np.vdot(block / np.linalg.norm(block), target))
        self.assertAlmostEqual(overlap, 1.0, places=10)

        # The witness is a genuine Laplacian eigenstate of the realized W_AB:
        # the C++ residual certifies it, and an independent numpy check confirms
        # L psi = lambda psi (robust at the zero mode, where ||L psi|| ~ 0).
        es = cob.EigenstateSynthesis(v.witness)
        self.assertLess(es.residual(_cvec(state)), 1e-10)
        L = _np_L(v.witness)
        Lp = L @ state
        lam = np.vdot(state, Lp).real
        np.testing.assert_allclose(Lp, lam * state, atol=1e-5)
        self.assertAlmostEqual(lam, v.eigenvalue, places=8)
        self.assertAlmostEqual(es.rayleigh(_cvec(state)), v.eigenvalue, places=10)

        # dW was pinned: the boundary edges are byte-identical (the fill never
        # touched them).
        live = _edge_map(st)
        for k, (w, ph) in boundary_before.items():
            self.assertEqual((live[k].getSquaredLength().real, live[k].getPhase()),
                             (w, ph))

    def test_realizable_only_after_interior_growth(self):
        # A generic U : H_B(dim 3) -> H_A(dim 1) on the solid triangle (3 boundary
        # vertices, 0 interior edges at the seed): not an eigenvector of the bare
        # pinned triangle, so it floors at the seed complexity, but the fixed-
        # boundary cone-and-retry grows the interior and realizes it (the §5.0
        # analogue of §4b coning, with the boundary pinned).
        st = _solid_triangle()
        _pin_all(st, w=1.0, phase=0.0)

        U = [[1.0 + 0j, 0.3 + 0.5j, -0.8 + 0.2j]]   # 1 x 3
        oracle = cob.RealizabilityOracle(st)
        v = oracle.decide(_cvec(_vec(U)), 1, 3, epsilon=1e-10, restarts=80,
                          max_cones=4, seed=0)

        self.assertTrue(v.realizable)
        self.assertLess(v.residual, 1e-10)
        self.assertGreaterEqual(v.cones_applied, 1)    # growth was needed
        self.assertEqual(v.interior_vertex_count, v.cones_applied)

        # The witness is a genuine eigenstate, and its output-boundary block is
        # proportional to vec(U).
        state = np.asarray(v.state)
        es = cob.EigenstateSynthesis(v.witness)
        self.assertLess(es.residual(_cvec(state)), 1e-10)
        target = _vec(U) / np.linalg.norm(_vec(U))
        block = state[:3]
        overlap = abs(np.vdot(block / np.linalg.norm(block), target))
        self.assertAlmostEqual(overlap, 1.0, places=8)


class ObstructedFloorTest(unittest.TestCase):
    """A deliberately obstructed U is certified non-realizable by a residual floor
    bounded away from 0 — the analogue of §4b's two-vertex floor — cross-checked
    against an independent numpy global-min oracle."""

    def test_generic_operator_floors_and_is_certified_non_realizable(self):
        # U = [[1,2],[3,4]]: vec(U) = [1,2,3,4], a generic state not realizable as
        # a bipyramid-boundary eigenvector against the pinned boundary with a
        # single interior edge. With no growth budget the residual floors.
        st = _bipyramid()
        _pin_all(st, w=1.0, phase=0.0)

        U = [[1.0 + 0j, 2.0 + 0j], [3.0 + 0j, 4.0 + 0j]]
        oracle = cob.RealizabilityOracle(st)
        v = oracle.decide(_cvec(_vec(U)), 2, 2, epsilon=1e-10, restarts=40,
                          max_cones=0, seed=0)

        # Certified non-realizable: the residual floors, and floor == residual.
        self.assertFalse(v.realizable)
        self.assertGreater(v.residual, 1e-2)
        self.assertEqual(v.floor, v.residual)
        self.assertEqual(v.interior_vertex_count, 0)

        # Independent numpy oracle: the true global min over the single interior
        # edge (0,1) (grid + L-BFGS-B refine), boundary pinned at weight 1 phase 0.
        from scipy.optimize import minimize
        fresh = _bipyramid()
        _pin_all(fresh, w=1.0, phase=0.0)
        edge01 = _edge_map(fresh)[(0, 1)]
        target = _vec(U)

        def floor_at(x):
            edge01.setSquaredLength(x[0])
            edge01.setPhase(x[1])
            return _np_residual(_np_L(fresh), target)

        w_bounds = (0.1, 10.0)
        grid, gx = np.inf, None
        for w in np.linspace(w_bounds[0], w_bounds[1], 60):
            for th in np.linspace(-math.pi, math.pi, 60):
                val = floor_at([w, th])
                if val < grid:
                    grid, gx = val, [w, th]
        ref = minimize(floor_at, gx, method="L-BFGS-B",
                       bounds=[w_bounds, (-2 * math.pi, 2 * math.pi)])
        oracle_floor = float(ref.fun)

        # The certified floor is bounded away from zero and matches the hand
        # computation.
        self.assertGreater(oracle_floor, 1e-2)
        self.assertAlmostEqual(v.floor, oracle_floor, delta=2e-3)

    def test_floor_is_seed_independent(self):
        # The certified floor is the genuine global minimum, not a seed-specific
        # local min: independent restart seeds reach the same floor. (This is the
        # certificate's robustness — the spec decides non-realizability by the
        # floor, so the floor must be the true one.)
        U = [[1.0 + 0j, 2.0 + 0j], [3.0 + 0j, 4.0 + 0j]]
        floors = []
        for s in (0, 11, 23):
            st = _bipyramid()
            _pin_all(st, w=1.0, phase=0.0)
            v = cob.RealizabilityOracle(st).decide(
                _cvec(_vec(U)), 2, 2, epsilon=1e-10, restarts=40, max_cones=0,
                seed=s)
            self.assertFalse(v.realizable)
            floors.append(v.floor)
        for f in floors[1:]:
            self.assertAlmostEqual(f, floors[0], delta=1e-3)


class DeterminismTest(unittest.TestCase):
    """The seeded synthesis is reproducible: same seed -> identical verdict."""

    def test_same_seed_same_verdict(self):
        U = [[1.0 + 0j, 2.0 + 0j], [3.0 + 0j, 4.0 + 0j]]
        results = []
        for _ in range(2):
            st = _bipyramid()
            _pin_all(st, w=1.0, phase=0.0)
            v = cob.RealizabilityOracle(st).decide(
                _cvec(_vec(U)), 2, 2, epsilon=1e-10, restarts=24, max_cones=0,
                seed=7)
            results.append(v)
        self.assertEqual(results[0].realizable, results[1].realizable)
        self.assertEqual(results[0].residual, results[1].residual)
        np.testing.assert_array_equal(np.asarray(results[0].state),
                                      np.asarray(results[1].state))


class GuardrailTest(unittest.TestCase):
    """Input validation."""

    def test_null_bulk_raises(self):
        with self.assertRaises(Exception):
            cob.RealizabilityOracle(None)

    def test_dimension_mismatch_raises(self):
        st = _bipyramid()
        _pin_all(st)
        oracle = cob.RealizabilityOracle(st)
        with self.assertRaises(Exception):
            oracle.decide(_cvec([1, 2, 3]), 2, 2)   # 3 != 2*2

    def test_bulk_too_small_for_target_raises(self):
        # A triangle (3 vertices) cannot carry a dA*dB = 4 output boundary.
        st = _solid_triangle()
        _pin_all(st)
        oracle = cob.RealizabilityOracle(st)
        with self.assertRaises(Exception):
            oracle.decide(_cvec([1, 1, 1, 1]), 2, 2, max_cones=0)


if __name__ == "__main__":
    unittest.main()
