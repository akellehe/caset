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

"""BoundaryStateSpace + PreparedBoundaryState value objects (#187).

`BoundaryStateSpace` is the per-Sigma factory, conceptually Z(Sigma) =
C[H^1(Sigma; Z_2)]: it owns Sigma + the cached `Cochain` harmonic basis of
ker L_1(Sigma) (HodgeLaplacian at k=1) and manufactures the value objects that
live in it. Two distinct objects sit on a closed surface Sigma:

* the Hodge harmonic 1-forms ker L_1(Sigma) — the *spectral qubit*, dimension
  b_1, an orthonormal basis of degree-1 `Cochain`s; and
* the DW boundary Hilbert space Z(Sigma) = C[H^1(Sigma; Z_2)] — the
  flat-connection-class basis, dimension 2^{b_1}.

The b_1 vs 2^{b_1} reconciliation lives in the space: H^1(Sigma; Z_2) =
(Z_2)^{b_1} has b_1 single-generator classes at gf2Span indices 2^0..2^{b_1-1}.
`prepare(form)` embeds the i-th harmonic 1-form onto the amplitude at index 2^i,
returning a `PreparedBoundaryState`; `PreparedBoundaryState.readout()` is the
adjoint. Because the harmonic basis is orthonormal, prepare is an isometry and
readout o prepare = id on ker L_1.

Anchored on T^2 (b_1 = 2: ker L_1(T^2) = C^2 the qubit, Z(T^2) = C^4):

* numpy/Hodge cross-check of the cached `Cochain` basis (metric + combinatorial);
* the b_1 vs 2^{b_1} reconciliation (harmonic i -> index 2^i);
* round-trip readout(prepare(psi)) = psi and the prepare isometry via overlap();
* the PreparedBoundaryState value-object ops (coeffs/indexing/norm/overlap/space);
* consistency with DijkgraafWitten.amplitude and the trivial cylinder
  Sigma x [0,T]: amplitude(prepare(psi), prepare(phi)) = <psi|phi>.
"""

import unittest

import numpy as np

import tessera

cobordism = tessera.cobordism
BoundaryStateSpace = cobordism.BoundaryStateSpace
PreparedBoundaryState = cobordism.PreparedBoundaryState
Cochain = cobordism.Cochain
DijkgraafWitten = cobordism.DijkgraafWitten
Cocycle = cobordism.Cocycle


# --------------------------------------------------------------------------- #
# Fixtures (mirror test_dijkgraaf_witten_boundary_python.py).
# --------------------------------------------------------------------------- #
def _build(topology):
    signature = tessera.Signature(topology.dimension(), tessera.Lorentzian)
    metric = tessera.Metric(True, signature)
    spacetime = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                                  tessera.PREFERRED, topology)
    spacetime.build()
    return spacetime


def _circle():
    return tessera.SimplexBoundarySphere(1)  # S^1 = boundary of a triangle


def _torus_topology():
    return tessera.SimplicialProduct(_circle(), _circle())  # T^2 = S^1 x S^1


def _torus():
    return _build(_torus_topology())


def _sphere2():
    return _build(tessera.SimplexBoundarySphere(2))  # S^2: b_1 = 0


def _torus_cylinder():
    # W = T^2 x [0,T], the trivial cobordism T^2 -> T^2 (boundary T^2 ⊔ T^2).
    return _build(tessera.SimplicialProduct(_torus_topology(),
                                            tessera.SolidSimplex(1)))


# --------------------------------------------------------------------------- #
# Helpers bridging the value objects to numpy.
# --------------------------------------------------------------------------- #
def _harmonics_matrix(space):
    """The cached basis as a |C_1| x b_1 numpy matrix (columns = h_i.coeffs())."""
    columns = [np.asarray(h.coeffs()) for h in space.harmonics()]
    if not columns:
        return np.zeros((space.numEdges(), 0), dtype=complex)
    return np.column_stack(columns)


def _edge_ordering(space):
    """The degree-1 simplex ordering a harmonic 1-form Cochain lives over."""
    harmonics = space.harmonics()
    if harmonics:
        return harmonics[0].simplices()
    return cobordism.ChainComplex.fromSpacetime(
        _sphere2()).kSimplexVertices(1)


def _form(space, coeffs):
    """A harmonic 1-form Cochain = sum_i coeffs[i] h_i over Sigma's edges."""
    vec = _harmonics_matrix(space) @ np.asarray(coeffs, dtype=complex)
    return Cochain(1, space.harmonics()[0].simplices(), vec)


# --------------------------------------------------------------------------- #
# Independent numpy Hodge oracle for ker L_1 (symmetric metric Laplacian).
# --------------------------------------------------------------------------- #
def _numpy_L1_symmetric(spacetime, metric):
    """L_1^sym = B_1^T B_1 + B_2 B_2^T with B_k = W_{k-1}^{1/2} d_k W_k^{-1/2}."""
    chain = cobordism.ChainComplex.fromSpacetime(spacetime)
    num_verts = chain.numSimplices(0)
    num_edges = chain.numSimplices(1)
    num_tris = chain.numSimplices(2)
    d1 = np.asarray(chain.boundaryMatrix(1), float).reshape(num_verts, num_edges)
    d2 = np.asarray(chain.boundaryMatrix(2), float).reshape(num_edges, num_tris)
    hodge = cobordism.HodgeLaplacian(spacetime)
    if metric:
        w1 = np.asarray(hodge.weights(1), float)
        w2 = np.asarray(hodge.weights(2), float)
    else:
        w1 = np.ones(num_edges)
        w2 = np.ones(num_tris)
    b1mat = d1 * (1.0 / np.sqrt(w1))[None, :]            # W_0 = I
    b2mat = np.sqrt(w1)[:, None] * d2 * (1.0 / np.sqrt(w2))[None, :]
    return b1mat.T @ b1mat + b2mat @ b2mat.T, num_edges


# --------------------------------------------------------------------------- #
# The cached harmonic basis: numpy/Hodge cross-check + stable ordering.
# --------------------------------------------------------------------------- #
class TestHarmonicBasisHodgeCrossCheck(unittest.TestCase):

    def test_torus_harmonic_count_is_b1(self):
        space = BoundaryStateSpace(_torus())
        self.assertEqual(space.harmonicDimension(), 2)       # b_1(T^2) = 2
        self.assertEqual(space.boundaryDimension(), 4)       # 2^{b_1} = 4
        self.assertEqual(space.numEdges(), 27)               # |C_1(T^2)|

    def test_harmonics_are_degree1_cochains(self):
        space = BoundaryStateSpace(_torus())
        harmonics = space.harmonics()
        self.assertEqual(len(harmonics), 2)
        for h in harmonics:
            self.assertEqual(h.degree(), 1)
            self.assertEqual(h.size(), space.numEdges())

    def test_harmonics_orthonormal(self):
        space = BoundaryStateSpace(_torus())
        harmonics = _harmonics_matrix(space)
        gram = harmonics.conj().T @ harmonics
        np.testing.assert_allclose(gram, np.eye(space.harmonicDimension()),
                                   atol=1e-9)

    def test_harmonics_match_numpy_hodge_subspace(self):
        # The C++ harmonics span exactly ker L_1 of an independent numpy assembly,
        # for both volume (metric) and unit (combinatorial) Hodge weights.
        torus = _torus()
        for metric in (True, False):
            with self.subTest(metric=metric):
                space = BoundaryStateSpace(torus, 1e-9, metric)
                harmonics = _harmonics_matrix(space)
                lap, num_edges = _numpy_L1_symmetric(torus, metric)
                self.assertEqual(harmonics.shape, (num_edges, 2))
                # nullity == b_1
                evals = np.linalg.eigvalsh(lap)
                self.assertEqual(int(np.sum(np.abs(evals) < 1e-9)), 2)
                # harmonics annihilated by L_1
                np.testing.assert_allclose(lap @ harmonics, 0.0, atol=1e-7)
                # projector equality: same subspace as the numpy kernel
                vals, vecs = np.linalg.eigh(lap)
                kernel = vecs[:, np.abs(vals) < 1e-9]
                np.testing.assert_allclose(kernel @ kernel.conj().T,
                                           harmonics @ harmonics.conj().T,
                                           atol=1e-7)

    def test_harmonic_basis_ordering_is_stable(self):
        # Deterministic, reproducible basis (the order prepare/readout rely on).
        a = _harmonics_matrix(BoundaryStateSpace(_torus()))
        b = _harmonics_matrix(BoundaryStateSpace(_torus()))
        np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------- #
# The b_1 vs 2^{b_1} reconciliation: harmonic i lands on flat-class index 2^i.
# --------------------------------------------------------------------------- #
class TestBasesReconciliation(unittest.TestCase):

    def test_generator_indices_are_powers_of_two(self):
        space = BoundaryStateSpace(_torus())
        self.assertEqual(list(space.generatorIndices()), [1, 2])  # 2^0, 2^1

    def test_each_harmonic_lands_on_its_generator_class(self):
        space = BoundaryStateSpace(_torus())
        for i, gen in enumerate(space.generatorIndices()):
            # Prepare the i-th basis 1-form Cochain directly.
            prepared = space.prepare(space.harmonics()[i])
            expected = np.zeros(space.boundaryDimension(), dtype=complex)
            expected[gen] = 1.0  # unit on its own class, zero elsewhere
            np.testing.assert_allclose(np.asarray(prepared.coeffs()), expected,
                                       atol=1e-9)

    def test_trivial_and_multi_generator_classes_stay_empty(self):
        # An arbitrary harmonic form never excites the trivial class (index 0) or
        # any multi-generator class (a non-power-of-two index, e.g. 3 = both).
        space = BoundaryStateSpace(_torus())
        prepared = space.prepare(_form(space, [0.7 + 0.2j, -0.4 + 1.1j]))
        coeffs = np.asarray(prepared.coeffs())
        generators = set(space.generatorIndices())
        for idx in range(space.boundaryDimension()):
            if idx not in generators:
                self.assertAlmostEqual(coeffs[idx], 0.0, places=12)


# --------------------------------------------------------------------------- #
# Round-trip and the prepare isometry on ker L_1 (via the value objects).
# --------------------------------------------------------------------------- #
class TestRoundTripAndIsometry(unittest.TestCase):

    def test_readout_prepare_is_identity_on_ker_L1(self):
        for metric in (True, False):
            with self.subTest(metric=metric):
                space = BoundaryStateSpace(_torus(), 1e-9, metric)
                rng = np.random.default_rng(187)
                # Basis forms and random complex combinations, all in ker L_1.
                forms = [_form(space, [1.0, 0.0]), _form(space, [0.0, 1.0])]
                for _ in range(5):
                    c = rng.standard_normal(2) + 1j * rng.standard_normal(2)
                    forms.append(_form(space, c))
                for form in forms:
                    back = space.prepare(form).readout()
                    self.assertEqual(back.degree(), 1)
                    np.testing.assert_allclose(np.asarray(back.coeffs()),
                                               np.asarray(form.coeffs()),
                                               atol=1e-9)

    def test_prepare_is_an_isometry(self):
        # overlap(prepare(psi), prepare(phi)) = <psi|phi> for psi, phi in ker L_1.
        space = BoundaryStateSpace(_torus())
        rng = np.random.default_rng(424242)
        for _ in range(6):
            a = rng.standard_normal(2) + 1j * rng.standard_normal(2)
            b = rng.standard_normal(2) + 1j * rng.standard_normal(2)
            psi, phi = _form(space, a), _form(space, b)
            prepared_inner = space.prepare(psi).overlap(space.prepare(phi))
            self.assertAlmostEqual(prepared_inner,
                                   np.vdot(np.asarray(psi.coeffs()),
                                           np.asarray(phi.coeffs())),
                                   places=9)

    def test_prepare_then_readout_projects_onto_generators(self):
        # The other composition: prepare o readout zeroes the non-generator slots
        # and is the identity on the generator subspace.
        space = BoundaryStateSpace(_torus())
        rng = np.random.default_rng(99)
        raw = rng.standard_normal(4) + 1j * rng.standard_normal(4)
        projected = np.asarray(space.prepare(space.state(raw).readout()).coeffs())
        expected = np.zeros(4, dtype=complex)
        for gen in space.generatorIndices():
            expected[gen] = raw[gen]
        np.testing.assert_allclose(projected, expected, atol=1e-9)


# --------------------------------------------------------------------------- #
# The PreparedBoundaryState value object: coeffs / indexing / norm / overlap.
# --------------------------------------------------------------------------- #
class TestPreparedBoundaryStateOps(unittest.TestCase):

    def setUp(self):
        self.space = BoundaryStateSpace(_torus())

    def test_coeffs_length_and_len(self):
        prepared = self.space.prepare(_form(self.space, [0.5, -0.5j]))
        self.assertEqual(len(prepared), 4)
        self.assertEqual(prepared.size(), 4)
        self.assertEqual(np.asarray(prepared.coeffs()).shape, (4,))

    def test_indexing_and_generator_amplitude(self):
        raw = np.array([0.1, 0.2 + 1j, 0.3, -0.4j], dtype=complex)
        state = self.space.state(raw)
        for idx in range(4):
            self.assertAlmostEqual(state[idx], raw[idx], places=12)
            self.assertAlmostEqual(state.amplitude(idx), raw[idx], places=12)
        # generatorAmplitude(i) == coeffs[2^i] (the convention owned by the space).
        for i, gen in enumerate(self.space.generatorIndices()):
            self.assertAlmostEqual(state.generatorAmplitude(i), raw[gen],
                                   places=12)

    def test_norm_matches_coeffs(self):
        state = self.space.state(np.array([1.0, 2.0, 3.0, 4.0], dtype=complex))
        self.assertAlmostEqual(state.norm(),
                               float(np.linalg.norm(np.asarray(state.coeffs()))),
                               places=12)

    def test_overlap_is_vdot(self):
        rng = np.random.default_rng(2024)
        for _ in range(5):
            a = rng.standard_normal(4) + 1j * rng.standard_normal(4)
            b = rng.standard_normal(4) + 1j * rng.standard_normal(4)
            overlap = self.space.state(a).overlap(self.space.state(b))
            self.assertAlmostEqual(overlap, np.vdot(a, b), places=12)

    def test_space_handle_is_its_space(self):
        prepared = self.space.prepare(_form(self.space, [1.0, 0.0]))
        self.assertEqual(prepared.space().boundaryDimension(),
                         self.space.boundaryDimension())
        self.assertEqual(prepared.space().harmonicDimension(),
                         self.space.harmonicDimension())

    def test_readout_returns_degree1_cochain(self):
        readout = self.space.prepare(_form(self.space, [0.3, 0.7])).readout()
        self.assertEqual(readout.degree(), 1)
        self.assertEqual(readout.size(), self.space.numEdges())

    def test_amplitude_index_out_of_range_raises(self):
        state = self.space.state(np.zeros(4, dtype=complex))
        with self.assertRaises((IndexError, ValueError)):
            state.amplitude(4)

    def test_generator_amplitude_out_of_range_raises(self):
        state = self.space.state(np.zeros(4, dtype=complex))
        with self.assertRaises((IndexError, ValueError)):
            state.generatorAmplitude(2)  # b_1 = 2, so 0 and 1 only

    def test_overlap_dimension_mismatch_raises(self):
        torus_state = self.space.state(np.zeros(4, dtype=complex))
        sphere_state = BoundaryStateSpace(_sphere2()).state(
            np.zeros(1, dtype=complex))
        with self.assertRaises((ValueError, RuntimeError)):
            torus_state.overlap(sphere_state)


# --------------------------------------------------------------------------- #
# Consistency with DijkgraafWitten.amplitude and the trivial cylinder.
# --------------------------------------------------------------------------- #
class TestAmplitudeConsistencyAndCylinder(unittest.TestCase):

    def setUp(self):
        self.space = BoundaryStateSpace(_torus())

    def test_prepared_state_feeds_amplitude(self):
        # A PreparedBoundaryState fed to amplitude() reproduces the direct
        # flat-connection-basis contraction conj(psi_A) . Z(W) . psi_B.
        dw = DijkgraafWitten(_torus_cylinder(), Cocycle.Trivial)
        zmap = np.asarray(dw.map())
        self.assertEqual(zmap.shape, (4, 4))
        rng = np.random.default_rng(1870)
        for _ in range(5):
            v_a = rng.standard_normal(4) + 1j * rng.standard_normal(4)
            v_b = rng.standard_normal(4) + 1j * rng.standard_normal(4)
            amplitude = dw.amplitude(self.space.state(v_a),
                                     self.space.state(v_b))
            self.assertAlmostEqual(amplitude, v_a.conj() @ zmap @ v_b, places=9)

    def test_cylinder_amplitude_is_harmonic_inner_product(self):
        # Trivial cobordism Z(W)=id: amplitude(prepare(psi), prepare(phi)) =
        # <psi|phi> (the harmonic inner product on ker L_1).
        dw = DijkgraafWitten(_torus_cylinder(), Cocycle.Trivial)
        rng = np.random.default_rng(8675309)
        for _ in range(6):
            psi = _form(self.space, rng.standard_normal(2)
                        + 1j * rng.standard_normal(2))
            phi = _form(self.space, rng.standard_normal(2)
                        + 1j * rng.standard_normal(2))
            amplitude = dw.amplitude(self.space.prepare(psi),
                                     self.space.prepare(phi))
            self.assertAlmostEqual(amplitude,
                                   np.vdot(np.asarray(psi.coeffs()),
                                           np.asarray(phi.coeffs())),
                                   places=9)

    def test_cylinder_diagonal_is_the_norm(self):
        # The explicit trivial-cobordism check: <psi|Z(W)|psi> = ||psi||^2, and
        # the prepared state's own norm reproduces it (prepare is an isometry).
        dw = DijkgraafWitten(_torus_cylinder(), Cocycle.Trivial)
        for i in range(self.space.harmonicDimension()):
            psi = self.space.harmonics()[i]
            prepared = self.space.prepare(psi)
            norm_squared = float(np.vdot(np.asarray(psi.coeffs()),
                                         np.asarray(psi.coeffs())).real)
            self.assertAlmostEqual(dw.amplitude(prepared, prepared),
                                   norm_squared, places=9)
            self.assertAlmostEqual(prepared.norm() ** 2, norm_squared, places=9)

    def test_sign_cocycle_cylinder_agrees(self):
        # The cup cube vanishes on the cylinder, so the Sign twist also returns
        # the harmonic inner product.
        dw = DijkgraafWitten(_torus_cylinder(), Cocycle.Sign)
        psi = _form(self.space, [0.6 - 0.3j, 0.2 + 0.9j])
        prepared = self.space.prepare(psi)
        self.assertAlmostEqual(dw.amplitude(prepared, prepared),
                               np.vdot(np.asarray(psi.coeffs()),
                                       np.asarray(psi.coeffs())),
                               places=9)


# --------------------------------------------------------------------------- #
# Degenerate surface and guards.
# --------------------------------------------------------------------------- #
class TestEdgeCasesAndGuards(unittest.TestCase):

    def test_sphere_has_trivial_boundary_space(self):
        # b_1(S^2) = 0: the spectral qubit is a point and Z(S^2) is 1-dimensional.
        space = BoundaryStateSpace(_sphere2())
        self.assertEqual(space.harmonicDimension(), 0)
        self.assertEqual(space.boundaryDimension(), 1)
        self.assertEqual(list(space.generatorIndices()), [])
        self.assertEqual(list(space.harmonics()), [])
        # prepare maps the (empty) harmonic data to the single trivial class.
        zero_form = Cochain(1, _edge_ordering(space),
                            np.zeros(space.numEdges(), dtype=complex))
        prepared = space.prepare(zero_form)
        self.assertEqual(list(np.asarray(prepared.coeffs())), [0.0])
        # readout of the single trivial class is the zero 1-form.
        back = space.state(np.array([1.0], dtype=complex)).readout()
        self.assertEqual(back.degree(), 1)
        np.testing.assert_array_equal(np.asarray(back.coeffs()),
                                      np.zeros(space.numEdges()))

    def test_prepare_rejects_wrong_form_length(self):
        space = BoundaryStateSpace(_torus())
        wrong = Cochain(1, [[0, 1], [0, 2]], np.array([1.0, 0.0], dtype=complex))
        with self.assertRaises((ValueError, RuntimeError)):
            space.prepare(wrong)

    def test_prepare_rejects_wrong_degree(self):
        space = BoundaryStateSpace(_torus())
        ordering = _edge_ordering(space)
        bad_degree = Cochain(0, ordering,
                             np.zeros(space.numEdges(), dtype=complex))
        with self.assertRaises((ValueError, RuntimeError)):
            space.prepare(bad_degree)

    def test_state_rejects_wrong_length(self):
        space = BoundaryStateSpace(_torus())
        with self.assertRaises((ValueError, RuntimeError)):
            space.state(np.array([1.0, 0.0, 0.0], dtype=complex))  # not 2^{b_1}=4

    def test_null_surface_raises(self):
        with self.assertRaises((RuntimeError, TypeError)):
            BoundaryStateSpace(None)


if __name__ == "__main__":
    unittest.main()
