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

"""§5.0 realizability synthesis lifted to k=1 boundary harmonics on a
3-manifold-with-boundary (#176, the v0.4 bridge sub-ticket 2 of #174).

The v0.3 `RealizabilityOracle`/`EigenstateSynthesis` synthesize a k=0 boundary
eigenvector on a 2-complex. The DW bridge lives in the 3-manifold / k=1 setting:
the spectral boundary qubit is the harmonic 1-forms ker L_1(Sigma) (dim b_1).
`EigenstateSynthesis(W, k=1)` scores a target edge 1-form against the metric Hodge
Laplacian L_1(W); `RealizabilityOracle.decideHarmonic` pins the boundary surface
dW byte-fixed and fills the interior (interior edge squared-lengths -> the simplex
volumes -> W_k of L_1, plus boundary-fixed 1->4 Pachner growth in 3D), driving
r = ||(I - psi psi^dagger) L_1 psi||^2 to 0.

Fixture: the **solid torus** W = D^2 x S^1 = SolidSimplex(2) x S^1 (a 3-manifold
with boundary T^2; b_1(W) = 1) whose boundary surface Sigma = T^2 = S^1 x S^1
(b_1 = 2, the natural qubit). The two share the product vertex-id scheme, so
dW == Sigma edge-for-edge.

Acceptance (#176), the topological dichotomy of H_1(Sigma) -> H_1(W):

  * **Realizable** — the boundary harmonic the manifold *carries*: the restriction
    of the bulk harmonic ker L_1(W) (the solid torus's longitude / core circle),
    expressed in the prepared DW basis via BoundaryStateSpace. decideHarmonic
    drives r -> 0 with the boundary pinned byte-fixed; the witness is a genuine
    L_1(W) harmonic (lambda ~ 0) whose boundary block matches the target.
  * **Obstructed** — the meridian (the boundary cycle that bounds a disk in W, so
    it dies in H_1(W)): the residual floors away from 0, a certified obstruction
    cross-checked against an independent numpy Hodge oracle.
  * **Growth** — the boundary-fixed 1->4 Pachner add runs in 3D (dW byte-fixed),
    and the interior fill genuinely cones + re-optimizes, driving a floored
    target's residual down.

numpy/Hodge cross-checks throughout: the C++ L_1 apply, the harmonic basis, and
the residual floor are all reproduced from an independent numpy assembly.
"""

import itertools
import unittest

import numpy as np

import tessera

cob = tessera.cobordism


# --------------------------------------------------------------------------- #
# Fixtures (the cobordism idiom: Signature(d) so the d-cells register as top
# simplices; built through the topology so the vertex-id counter advances).
# --------------------------------------------------------------------------- #
def _build(topology):
    sig = tessera.Signature(topology.dimension(), tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, topology)
    st.build()
    return st


def _circle():
    return tessera.SimplexBoundarySphere(1)  # S^1 = boundary of a triangle


def _solid_torus():
    """W = D^2 x S^1 = SolidSimplex(2) x S^1: a 3-manifold with boundary T^2."""
    return _build(tessera.SimplicialProduct(tessera.SolidSimplex(2), _circle()))


def _torus():
    """Sigma = T^2 = S^1 x S^1, the boundary surface (b_1 = 2)."""
    return _build(tessera.SimplicialProduct(_circle(), _circle()))


def _pin_uniform(st, w=1.0, phase=0.0):
    """Pin every edge to a fixed Hermitian value; decideHarmonic's fill only
    rewrites interior edges, so this fixes dW (and Sigma)."""
    for e in st.getEdgeList().toVector():
        e.setSquaredLength(w)
        e.setPhase(phase)


def _pinned_solid_torus():
    """A fresh solid torus with every edge pinned uniform — the synthesis bulk."""
    W = _solid_torus()
    _pin_uniform(W)
    return W


# --------------------------------------------------------------------------- #
# numpy Hodge oracle for L_1 (symmetric metric Laplacian) + helpers.
# --------------------------------------------------------------------------- #
def _numpy_L1(st):
    """L_1^sym = B_1^T B_1 + B_2 B_2^T, B_k = W_{k-1}^{1/2} d_k W_k^{-1/2}, in the
    canonical ChainComplex k=1 cell order (matches HodgeLaplacian.laplacian(1))."""
    chain = cob.ChainComplex.fromSpacetime(st)
    nv, ne, nt = (chain.numSimplices(0), chain.numSimplices(1),
                  chain.numSimplices(2))
    d1 = np.asarray(chain.boundaryMatrix(1), float).reshape(nv, ne)
    d2 = np.asarray(chain.boundaryMatrix(2), float).reshape(ne, nt)
    hodge = cob.HodgeLaplacian(st)
    w1 = np.asarray(hodge.weights(1), float)
    w2 = np.asarray(hodge.weights(2), float)
    b1 = d1 * (1.0 / np.sqrt(w1))[None, :]          # W_0 = I
    b2 = np.sqrt(w1)[:, None] * d2 * (1.0 / np.sqrt(w2))[None, :]
    return b1.T @ b1 + b2 @ b2.T


def _residual_agnostic(L, psi):
    psi = np.asarray(psi, dtype=complex)
    psi = psi / np.linalg.norm(psi)
    Lp = L @ psi
    lam = np.vdot(psi, Lp).real
    return float(np.vdot(Lp - lam * psi, Lp - lam * psi).real)


def _boundary_edges(W):
    """Edges of W on dW (in some boundary triangle), as sorted id pairs."""
    bnd = set()
    for facet in cob.Cobordism.boundaryFaces(W):
        for e in itertools.combinations(sorted(facet), 2):
            bnd.add(e)
    return bnd


def _cvec(v):
    return [complex(z) for z in v]


def _Cochain(simplices, coeffs):
    return cob.Cochain(1, simplices, np.asarray(coeffs, dtype=complex))


def _embed_on_W(form, cells):
    """Map a degree-1 Cochain over Sigma's edges onto W's k=1 cell order."""
    index = {tuple(c): i for i, c in enumerate(cells)}
    out = np.zeros(len(cells), dtype=complex)
    for c, s in zip(np.asarray(form.coeffs()), form.simplices()):
        out[index[tuple(s)]] = c
    return out


def _longitude_and_meridian(W, space):
    """The two distinguished boundary harmonics of the solid torus:

      * longitude = the restriction of the bulk harmonic ker L_1(W) to Sigma,
        expressed in the prepared DW basis (the cycle that survives in H_1(W));
      * meridian  = its orthogonal complement in ker L_1(Sigma) (the cycle that
        bounds a disk in W, dying in H_1(W)).
    """
    sig_simpl = space.harmonics()[0].simplices()
    bulk_h = cob.HodgeLaplacian(W).harmonics(1)[0]   # b_1(W) = 1
    restriction = _Cochain(
        sig_simpl,
        [complex(bulk_h.amplitudeFor(list(e))) for e in sig_simpl])
    prepared = space.prepare(restriction)
    longitude = prepared.readout()
    coords = np.array([complex(prepared.generatorAmplitude(i)) for i in range(2)])
    coords = coords / np.linalg.norm(coords)
    harmonics = np.column_stack([np.asarray(h.coeffs()) for h in space.harmonics()])
    meridian = _Cochain(sig_simpl, harmonics @ np.array([coords[1], -coords[0]]))
    return longitude, meridian


# --------------------------------------------------------------------------- #
class SolidTorusFixtureTest(unittest.TestCase):
    """The 3-manifold-with-boundary fixture and its boundary surface."""

    def test_solid_torus_is_three_manifold_with_torus_boundary(self):
        W = _solid_torus()
        chain = cob.ChainComplex.fromSpacetime(W)
        self.assertEqual(chain.dimension(), 3)                 # tetrahedra
        self.assertEqual(chain.bettiNumbers(), [1, 1, 0, 0])   # solid torus
        # dW is a single T^2 component: chi(dW) = 0, all triangles in one tet.
        facets = cob.Cobordism.boundaryFaces(W)
        self.assertTrue(all(len(f) == 3 for f in facets))

    def test_boundary_matches_standalone_torus(self):
        # dW edge-for-edge equals the standalone Sigma = T^2 (shared product
        # vertex-id scheme), so a boundary harmonic of Sigma lands on dW.
        self.assertEqual(_boundary_edges(_solid_torus()),
                         set(tuple(e) for e in cob.ChainComplex.fromSpacetime(
                             _torus()).kSimplexVertices(1)))

    def test_boundary_qubit_is_b1_two(self):
        space = cob.BoundaryStateSpace(_torus())
        self.assertEqual(space.harmonicDimension(), 2)         # b_1(T^2) = 2
        self.assertEqual(space.boundaryDimension(), 4)         # Z(T^2) = C^4


# --------------------------------------------------------------------------- #
class K1EngineTest(unittest.TestCase):
    """EigenstateSynthesis at k=1: the metric Hodge L_1 residual core, with a
    numpy cross-check, and the 3D boundary-fixed Pachner growth."""

    def test_degree_order_and_cells(self):
        W = _solid_torus()
        _pin_uniform(W)
        es = cob.EigenstateSynthesis(W, 1)
        self.assertEqual(es.degree(), 1)
        # psi is an edge 1-form: order() = |C_1(W)| = 27, and every edge is a
        # boundary edge of the minimal solid torus (0 interior edges at the seed).
        self.assertEqual(es.order(), 27)
        self.assertEqual(len(es.cellSimplices()), 27)
        self.assertEqual(es.numInteriorEdges(), 0)
        self.assertEqual(es.numBoundaryEdges(), 27)
        self.assertTrue(all(len(c) == 2 for c in es.cellSimplices()))  # edges

    def test_apply_and_residual_match_numpy_hodge(self):
        # L_1(W) apply and the eigenvalue-agnostic residual reproduce an
        # independent numpy assembly of the symmetric metric Hodge Laplacian.
        W = _solid_torus()
        _pin_uniform(W)
        es = cob.EigenstateSynthesis(W, 1)
        L1 = _numpy_L1(W)
        rng = np.random.default_rng(176)
        psi = rng.standard_normal(27) + 1j * rng.standard_normal(27)
        np.testing.assert_allclose(np.asarray(es.apply(_cvec(psi))), L1 @ psi,
                                   atol=1e-8)
        self.assertAlmostEqual(es.residual(_cvec(psi)),
                               _residual_agnostic(L1, psi), places=8)

    def test_bulk_harmonic_has_zero_residual(self):
        # ker L_1(W) is the exact zero mode: r ~ 0 and Rayleigh lambda ~ 0.
        W = _solid_torus()
        _pin_uniform(W)
        es = cob.EigenstateSynthesis(W, 1)
        h = np.asarray(cob.HodgeLaplacian(W).harmonics(1)[0].coeffs())
        # The harmonic Cochain is in the same k=1 order as the engine.
        self.assertLess(es.residual(_cvec(h)), 1e-15)
        self.assertAlmostEqual(es.rayleigh(_cvec(h)), 0.0, places=9)

    def test_boundary_fixed_pachner_growth_is_3d(self):
        # growInterior is the 1->4 stellar add on a tetrahedron: +1 interior
        # vertex, +4 interior edges, dW byte-fixed, degree unchanged.
        W = _solid_torus()
        _pin_uniform(W)
        es = cob.EigenstateSynthesis(W, 1)
        boundary_before = set(tuple(t) for t in es.boundaryEdges())
        self.assertEqual(len(boundary_before), 27)
        self.assertTrue(es.growInterior(7))
        self.assertEqual(es.degree(), 1)
        self.assertEqual(es.interiorVertexCount(), 1)
        self.assertEqual(es.numInteriorEdges(), 4)             # apex -> 4 verts
        self.assertEqual(es.order(), 31)                       # |C_1| grew by 4
        # dW is byte-fixed: the boundary edge set is unchanged.
        self.assertEqual(set(tuple(t) for t in es.boundaryEdges()),
                         boundary_before)


# --------------------------------------------------------------------------- #
class RealizableTest(unittest.TestCase):
    """The boundary harmonic the solid torus carries (the longitude) is realized:
    r -> 0, the witness is a genuine L_1(W) harmonic, dW byte-fixed."""

    def test_longitude_is_realizable(self):
        W = _solid_torus()
        _pin_uniform(W)
        space = cob.BoundaryStateSpace(_torus())
        longitude, _ = _longitude_and_meridian(W, space)

        # The target is a genuine boundary harmonic (the readout of a prepared
        # DW boundary state), annihilated by L_1(Sigma).
        L1_sigma = _numpy_L1(_torus())
        np.testing.assert_allclose(
            L1_sigma @ np.asarray(longitude.coeffs()), 0.0, atol=1e-7)

        boundary_before = {(min(a, b), max(a, b)):
                           (e.getSquaredLength().real, e.getPhase())
                           for e in W.getEdgeList().toVector()
                           for a, b in [(e.getSource().getId(),
                                         e.getTarget().getId())]}

        oracle = cob.RealizabilityOracle(W)
        v = oracle.decideHarmonic(longitude, epsilon=1e-9, restarts=8,
                                  max_cones=0, seed=1)

        # Realizable, r driven below the threshold, floor 0, no growth needed
        # (the manifold already carries it at the seed metric).
        self.assertTrue(v.realizable)
        self.assertLess(v.residual, 1e-9)
        self.assertEqual(v.floor, 0.0)
        self.assertEqual(v.interior_vertex_count, 0)

        # The witness is a genuine harmonic: lambda ~ 0 and L_1(W) psi ~ 0.
        state = np.asarray(v.state)
        self.assertEqual(state.shape, (27,))
        self.assertAlmostEqual(v.eigenvalue, 0.0, places=7)
        L1_W = _numpy_L1(W)
        np.testing.assert_allclose(L1_W @ state, 0.0, atol=1e-6)

        # The witness's boundary block matches the target (all 27 cells are
        # boundary cells): proportional to the embedded target form.
        target_on_W = _embed_on_W(longitude, [tuple(c)
                                              for c in cob.EigenstateSynthesis(
                                                  W, 1).cellSimplices()])
        overlap = abs(np.vdot(state / np.linalg.norm(state),
                              target_on_W / np.linalg.norm(target_on_W)))
        self.assertAlmostEqual(overlap, 1.0, places=7)

        # dW was pinned byte-identical (the fill never touched the boundary).
        live = {(min(a, b), max(a, b)): (e.getSquaredLength().real, e.getPhase())
                for e in W.getEdgeList().toVector()
                for a, b in [(e.getSource().getId(), e.getTarget().getId())]}
        self.assertEqual(live, boundary_before)

    def test_realizable_is_deterministic(self):
        results = []
        for _ in range(2):
            W = _solid_torus()
            _pin_uniform(W)
            longitude, _ = _longitude_and_meridian(W, cob.BoundaryStateSpace(
                _torus()))
            v = cob.RealizabilityOracle(W).decideHarmonic(
                longitude, epsilon=1e-9, restarts=8, max_cones=0, seed=5)
            results.append(v)
        self.assertEqual(results[0].realizable, results[1].realizable)
        self.assertEqual(results[0].residual, results[1].residual)
        np.testing.assert_array_equal(np.asarray(results[0].state),
                                      np.asarray(results[1].state))


# --------------------------------------------------------------------------- #
class ObstructedTest(unittest.TestCase):
    """The meridian (a cycle that bounds in W) is certified non-realizable by a
    residual floor bounded away from 0, cross-checked against numpy."""

    def test_meridian_floors_and_is_certified_non_realizable(self):
        W = _solid_torus()
        _pin_uniform(W)
        space = cob.BoundaryStateSpace(_torus())
        _, meridian = _longitude_and_meridian(W, space)

        oracle = cob.RealizabilityOracle(W)
        v = oracle.decideHarmonic(meridian, epsilon=1e-9, restarts=8,
                                  max_cones=0, seed=0)

        self.assertFalse(v.realizable)
        self.assertGreater(v.residual, 1e-2)            # bounded away from 0
        self.assertEqual(v.floor, v.residual)
        self.assertEqual(v.interior_vertex_count, 0)

        # Independent numpy oracle: at the seed there are no interior cells, so
        # the floor is exactly r(meridian) on L_1(W) (the global min over an
        # empty parameter set), to which the certified floor must agree.
        cells = [tuple(c) for c in cob.EigenstateSynthesis(W, 1).cellSimplices()]
        mer_on_W = _embed_on_W(meridian, cells)
        self.assertAlmostEqual(v.floor, _residual_agnostic(_numpy_L1(W), mer_on_W),
                               delta=1e-6)

    def test_raw_boundary_harmonics_also_floor(self):
        # The bare ker L_1(Sigma) basis 1-forms are not the carried harmonic
        # either; each floors against the pinned solid-torus boundary.
        W = _solid_torus()
        _pin_uniform(W)
        space = cob.BoundaryStateSpace(_torus())
        for i in range(space.harmonicDimension()):
            v = cob.RealizabilityOracle(W).decideHarmonic(
                space.harmonics()[i], epsilon=1e-9, restarts=4, max_cones=0,
                seed=i)
            self.assertFalse(v.realizable)
            self.assertGreater(v.floor, 1e-2)


# --------------------------------------------------------------------------- #
class GrowthFillTest(unittest.TestCase):
    """The interior fill genuinely cones (boundary-fixed 1->4 Pachner) and
    re-optimizes in 3D, driving a floored target's residual down while keeping
    dW byte-fixed."""

    def test_growth_reduces_a_floored_residual(self):
        W = _solid_torus()
        _pin_uniform(W)
        space = cob.BoundaryStateSpace(_torus())
        _, meridian = _longitude_and_meridian(W, space)

        seed_floor = cob.RealizabilityOracle(_pinned_solid_torus()).decideHarmonic(
            meridian, epsilon=1e-9, restarts=4, max_cones=0, seed=2).floor

        boundary_before = set(_boundary_edges(W))
        v = cob.RealizabilityOracle(W).decideHarmonic(
            meridian, epsilon=1e-9, restarts=4, max_cones=1, seed=2)

        # The fill coned in an interior vertex (boundary-fixed 1->4 Pachner) and
        # re-optimized: the witness grew and the residual dropped well below the
        # bare-seed floor (genuine LM + interior fill work in 3D).
        self.assertGreaterEqual(v.cones_applied, 1)
        self.assertEqual(v.interior_vertex_count, v.cones_applied)
        self.assertEqual(len(v.state), 27 + 4 * v.cones_applied)
        self.assertLess(v.residual, seed_floor)

        # dW is byte-fixed through the growth: the boundary edge set is unchanged.
        self.assertEqual(set(_boundary_edges(W)), boundary_before)


if __name__ == "__main__":
    unittest.main()
