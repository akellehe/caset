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

"""Spectral->DW boundary preparation map (#175): ker L_1(Sigma) -> C[H^1(Sigma;Z_2)].

`BoundaryStatePrep` makes explicit (and invertible) the preparation that
`DijkgraafWitten.amplitude()` uses internally. Two distinct objects live on a
closed surface Sigma:

* the Hodge harmonic 1-forms ker L_1(Sigma) — the *spectral qubit*, dimension
  b_1, an orthonormal basis of 1-forms (HodgeLaplacian at k=1); and
* the DW boundary Hilbert space Z(Sigma) = C[H^1(Sigma; Z_2)] — the
  flat-connection-class basis, dimension 2^{b_1}.

The b_1 vs 2^{b_1} reconciliation: H^1(Sigma; Z_2) = (Z_2)^{b_1} has b_1
single-generator classes at gf2Span indices 2^0, 2^1, ..., 2^{b_1-1}. prepare
embeds the i-th harmonic 1-form onto the amplitude at index 2^i; the trivial
class and every multi-generator class carry 0. Because the harmonic basis is
orthonormal, prepare is an isometry and readout o prepare = id on ker L_1.

Anchored on T^2 (b_1 = 2: ker L_1(T^2) = C^2 the qubit, Z(T^2) = C^4):

* numpy/Hodge cross-check of the harmonic basis (metric + combinatorial weights),
  with a stable, documented ordering;
* the b_1 vs 2^{b_1} reconciliation (harmonic i -> index 2^i);
* round-trip readout(prepare(psi)) = psi and the prepare isometry on ker L_1;
* consistency with amplitude() and the trivial cylinder Sigma x [0,T]:
  amplitude(prepare(psi), prepare(phi)) = <psi|phi>.
"""

import unittest

import numpy as np

import tessera

cobordism = tessera.cobordism
BoundaryStatePrep = cobordism.BoundaryStatePrep
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
# The harmonic basis: numpy/Hodge cross-check + stable ordering.
# --------------------------------------------------------------------------- #
class TestHarmonicBasisHodgeCrossCheck(unittest.TestCase):

    def test_torus_harmonic_count_is_b1(self):
        prep = BoundaryStatePrep(_torus())
        self.assertEqual(prep.harmonicDimension(), 2)        # b_1(T^2) = 2
        self.assertEqual(prep.boundaryDimension(), 4)        # 2^{b_1} = 4
        self.assertEqual(prep.numEdges(), 27)                # |C_1(T^2)|

    def _harmonics_matrix(self, prep):
        return np.asarray(prep.harmonics()).reshape(prep.numEdges(),
                                                    prep.harmonicDimension())

    def test_harmonics_orthonormal(self):
        prep = BoundaryStatePrep(_torus())
        harmonics = self._harmonics_matrix(prep)
        gram = harmonics.conj().T @ harmonics
        np.testing.assert_allclose(gram, np.eye(prep.harmonicDimension()),
                                   atol=1e-9)

    def test_harmonics_match_numpy_hodge_subspace(self):
        # The C++ harmonics span exactly ker L_1 of an independent numpy assembly,
        # for both volume (metric) and unit (combinatorial) Hodge weights.
        torus = _torus()
        for metric in (True, False):
            with self.subTest(metric=metric):
                prep = BoundaryStatePrep(torus, 1e-9, metric)
                harmonics = self._harmonics_matrix(prep)
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
        a = np.asarray(BoundaryStatePrep(_torus()).harmonics())
        b = np.asarray(BoundaryStatePrep(_torus()).harmonics())
        np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------- #
# The b_1 vs 2^{b_1} reconciliation: harmonic i lands on flat-class index 2^i.
# --------------------------------------------------------------------------- #
class TestBasesReconciliation(unittest.TestCase):

    def test_generator_indices_are_powers_of_two(self):
        prep = BoundaryStatePrep(_torus())
        self.assertEqual(list(prep.generatorIndices()), [1, 2])  # 2^0, 2^1

    def test_each_harmonic_lands_on_its_generator_class(self):
        prep = BoundaryStatePrep(_torus())
        harmonics = np.asarray(prep.harmonics()).reshape(prep.numEdges(),
                                                         prep.harmonicDimension())
        for i, gen in enumerate(prep.generatorIndices()):
            state = np.asarray(prep.prepare(list(harmonics[:, i])))
            # Unit amplitude on its own generator class, zero everywhere else.
            expected = np.zeros(prep.boundaryDimension(), dtype=complex)
            expected[gen] = 1.0
            np.testing.assert_allclose(state, expected, atol=1e-9)

    def test_trivial_and_multi_generator_classes_stay_empty(self):
        # An arbitrary harmonic form never excites the trivial class (index 0) or
        # any multi-generator class (a non-power-of-two index, e.g. 3 = both).
        prep = BoundaryStatePrep(_torus())
        harmonics = np.asarray(prep.harmonics()).reshape(prep.numEdges(),
                                                         prep.harmonicDimension())
        form = harmonics @ np.array([0.7 + 0.2j, -0.4 + 1.1j])
        state = np.asarray(prep.prepare(list(form)))
        generators = set(prep.generatorIndices())
        for idx in range(prep.boundaryDimension()):
            if idx not in generators:
                self.assertAlmostEqual(state[idx], 0.0, places=12)


# --------------------------------------------------------------------------- #
# Round-trip and the prepare isometry on ker L_1.
# --------------------------------------------------------------------------- #
class TestRoundTripAndIsometry(unittest.TestCase):

    def _setup(self, metric=True):
        prep = BoundaryStatePrep(_torus(), 1e-9, metric)
        harmonics = np.asarray(prep.harmonics()).reshape(prep.numEdges(),
                                                         prep.harmonicDimension())
        return prep, harmonics

    def test_readout_prepare_is_identity_on_ker_L1(self):
        for metric in (True, False):
            with self.subTest(metric=metric):
                prep, harmonics = self._setup(metric)
                rng = np.random.default_rng(175)
                # Basis forms and random complex combinations, all in ker L_1.
                forms = [harmonics[:, 0], harmonics[:, 1]]
                for _ in range(5):
                    coeffs = rng.standard_normal(2) + 1j * rng.standard_normal(2)
                    forms.append(harmonics @ coeffs)
                for form in forms:
                    back = np.asarray(prep.readout(prep.prepare(list(form))))
                    np.testing.assert_allclose(back, form, atol=1e-9)

    def test_prepare_is_an_isometry(self):
        # <prepare(psi)|prepare(phi)> = <psi|phi> for psi, phi in ker L_1.
        prep, harmonics = self._setup()
        rng = np.random.default_rng(424242)
        for _ in range(6):
            a = rng.standard_normal(2) + 1j * rng.standard_normal(2)
            b = rng.standard_normal(2) + 1j * rng.standard_normal(2)
            psi, phi = harmonics @ a, harmonics @ b
            prepared_inner = np.vdot(np.asarray(prep.prepare(list(psi))),
                                     np.asarray(prep.prepare(list(phi))))
            self.assertAlmostEqual(prepared_inner, np.vdot(psi, phi), places=9)

    def test_prepare_then_readout_projects_onto_generators(self):
        # The other composition: prepare o readout zeroes the non-generator slots
        # and is the identity on the generator subspace.
        prep, _ = self._setup()
        rng = np.random.default_rng(99)
        state = rng.standard_normal(4) + 1j * rng.standard_normal(4)
        projected = np.asarray(prep.prepare(prep.readout(list(state))))
        expected = np.zeros(4, dtype=complex)
        for gen in prep.generatorIndices():
            expected[gen] = state[gen]
        np.testing.assert_allclose(projected, expected, atol=1e-9)


# --------------------------------------------------------------------------- #
# Consistency with amplitude() and the trivial cylinder Sigma x [0,T].
# --------------------------------------------------------------------------- #
class TestAmplitudeConsistencyAndCylinder(unittest.TestCase):

    def setUp(self):
        self.prep = BoundaryStatePrep(_torus())
        self.harmonics = np.asarray(self.prep.harmonics()).reshape(
            self.prep.numEdges(), self.prep.harmonicDimension())

    def _form(self, coeffs):
        return self.harmonics @ np.asarray(coeffs, dtype=complex)

    def test_prepared_state_feeds_amplitude(self):
        # A state prepared via the new map, fed to amplitude(), reproduces the
        # direct flat-connection-basis contraction conj(psi_A) . Z(W) . psi_B.
        dw = DijkgraafWitten(_torus_cylinder(), Cocycle.Trivial)
        zmap = np.asarray(dw.map())
        self.assertEqual(zmap.shape, (4, 4))
        rng = np.random.default_rng(1750)
        for _ in range(5):
            psi = self._form(rng.standard_normal(2) + 1j * rng.standard_normal(2))
            phi = self._form(rng.standard_normal(2) + 1j * rng.standard_normal(2))
            prep_a = np.asarray(self.prep.prepare(list(psi)))
            prep_b = np.asarray(self.prep.prepare(list(phi)))
            amplitude = dw.amplitude(list(prep_a), list(prep_b))
            direct = prep_a.conj() @ zmap @ prep_b
            self.assertAlmostEqual(amplitude, direct, places=9)

    def test_cylinder_amplitude_is_harmonic_inner_product(self):
        # Trivial cobordism Z(W)=id: amplitude(prepare(psi), prepare(phi)) =
        # <psi|phi> (the harmonic inner product on ker L_1).
        dw = DijkgraafWitten(_torus_cylinder(), Cocycle.Trivial)
        rng = np.random.default_rng(8675309)
        for _ in range(6):
            psi = self._form(rng.standard_normal(2) + 1j * rng.standard_normal(2))
            phi = self._form(rng.standard_normal(2) + 1j * rng.standard_normal(2))
            amplitude = dw.amplitude(list(self.prep.prepare(list(psi))),
                                     list(self.prep.prepare(list(phi))))
            self.assertAlmostEqual(amplitude, np.vdot(psi, phi), places=9)

    def test_cylinder_diagonal_is_the_norm(self):
        # The explicit trivial-cobordism check: <psi|Z(W)|psi> = ||psi||^2.
        dw = DijkgraafWitten(_torus_cylinder(), Cocycle.Trivial)
        for i in range(self.prep.harmonicDimension()):
            psi = self.harmonics[:, i]
            prepared = list(self.prep.prepare(list(psi)))
            self.assertAlmostEqual(dw.amplitude(prepared, prepared),
                                   float(np.vdot(psi, psi).real), places=9)

    def test_sign_cocycle_cylinder_agrees(self):
        # The cup cube vanishes on the cylinder, so the Sign twist also returns
        # the harmonic inner product.
        dw = DijkgraafWitten(_torus_cylinder(), Cocycle.Sign)
        psi = self._form([0.6 - 0.3j, 0.2 + 0.9j])
        amplitude = dw.amplitude(list(self.prep.prepare(list(psi))),
                                 list(self.prep.prepare(list(psi))))
        self.assertAlmostEqual(amplitude, np.vdot(psi, psi), places=9)


# --------------------------------------------------------------------------- #
# Degenerate surface and guards.
# --------------------------------------------------------------------------- #
class TestEdgeCasesAndGuards(unittest.TestCase):

    def test_sphere_has_trivial_boundary_space(self):
        # b_1(S^2) = 0: the spectral qubit is a point and Z(S^2) is 1-dimensional.
        prep = BoundaryStatePrep(_sphere2())
        self.assertEqual(prep.harmonicDimension(), 0)
        self.assertEqual(prep.boundaryDimension(), 1)
        self.assertEqual(list(prep.generatorIndices()), [])
        # prepare maps the (empty) harmonic data to the single trivial class.
        state = prep.prepare([0.0] * prep.numEdges())
        self.assertEqual(list(state), [0.0])
        self.assertEqual(list(prep.readout([1.0])), [0.0] * prep.numEdges())

    def test_prepare_rejects_wrong_form_length(self):
        prep = BoundaryStatePrep(_torus())
        with self.assertRaises((ValueError, RuntimeError)):
            prep.prepare([1.0, 0.0])

    def test_readout_rejects_wrong_state_length(self):
        prep = BoundaryStatePrep(_torus())
        with self.assertRaises((ValueError, RuntimeError)):
            prep.readout([1.0, 0.0, 0.0])  # not 2^{b_1} = 4

    def test_null_surface_raises(self):
        with self.assertRaises((RuntimeError, TypeError)):
            BoundaryStatePrep(None)


if __name__ == "__main__":
    unittest.main()
