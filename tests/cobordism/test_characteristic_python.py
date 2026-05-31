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

"""Capability A: characteristic numbers (#65).

Euler characteristic and signature (as Observables) plus the aggregate
CharacteristicNumbers. The signature is validated on the intersection form:
the 4-sphere gives the empty form, and S^2 x S^2 gives the hyperbolic form
(signature 0 but rank 2 -- which only happens if the cup product is computed
correctly). The signature = +1 case (the complex projective plane) and the
Stiefel-Whitney numbers are pending follow-ups.
"""

import unittest

import tessera

cobordism = tessera.cobordism


def _build(topology):
    signature = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, signature)
    spacetime = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                                  tessera.PREFERRED, topology)
    spacetime.build()
    return spacetime


def _sphere(n):
    return tessera.SimplexBoundarySphere(n)


def _ball(n):
    return tessera.SolidSimplex(n)


def _s2_cross_s2():
    return tessera.SimplicialProduct(_sphere(2), _sphere(2))


def _torus():
    return tessera.SimplicialProduct(_sphere(1), _sphere(1))


def _s1_cross_s2():
    return tessera.SimplicialProduct(_sphere(1), _sphere(2))


class TestEulerCharacteristic(unittest.TestCase):

    def test_spheres(self):
        # chi(S^n) = 1 + (-1)^n.
        euler = cobordism.EulerCharacteristic()
        for n in range(1, 7):
            with self.subTest(sphere=n):
                self.assertEqual(euler.compute(_build(_sphere(n))),
                                 float(1 + (-1) ** n))

    def test_balls_are_contractible(self):
        # A solid simplex is contractible, so chi = 1.
        euler = cobordism.EulerCharacteristic()
        for n in range(1, 7):
            with self.subTest(ball=n):
                self.assertEqual(euler.compute(_build(_ball(n))), 1.0)

    def test_named_manifolds(self):
        euler = cobordism.EulerCharacteristic()
        cases = [
            (tessera.RealProjectivePlane(), 1),
            (_torus(), 0),
            (_s1_cross_s2(), 0),
            (_s2_cross_s2(), 4),
        ]
        for topology, expected in cases:
            with self.subTest(manifold=type(topology).__name__):
                self.assertEqual(euler.compute(_build(topology)), float(expected))

    def test_matches_alternating_sum_of_betti_numbers(self):
        # chi should equal sum_k (-1)^k b_k for every complex (a consistency
        # check between the f-vector count and homology).
        euler = cobordism.EulerCharacteristic()
        for topology in (_sphere(2), _sphere(4), _ball(3),
                         tessera.RealProjectivePlane(), _torus(), _s2_cross_s2()):
            with self.subTest(manifold=type(topology).__name__):
                spacetime = _build(topology)
                chain = cobordism.ChainComplex.fromSpacetime(spacetime)
                betti = chain.bettiNumbers()
                from_betti = sum((-1) ** k * b for k, b in enumerate(betti))
                self.assertEqual(euler.compute(spacetime), float(from_betti))

    def test_empty_complex(self):
        empty = tessera.Spacetime()
        self.assertEqual(cobordism.EulerCharacteristic().compute(empty), 0.0)


class TestSignature(unittest.TestCase):

    def test_four_sphere_has_empty_form(self):
        spacetime = _build(_sphere(4))
        chain = cobordism.ChainComplex.fromSpacetime(spacetime)
        self.assertEqual(chain.bettiNumbers()[2], 0)
        self.assertEqual(list(chain.intersectionForm()), [])
        self.assertEqual(cobordism.Signature().compute(spacetime), 0.0)

    def test_s2_cross_s2_is_hyperbolic(self):
        # The intersection form of S^2 x S^2 is the hyperbolic form [[0,1],[1,0]]:
        # signature 0 but rank 2 (nondegenerate, indefinite).
        spacetime = _build(_s2_cross_s2())
        chain = cobordism.ChainComplex.fromSpacetime(spacetime)
        self.assertEqual(chain.bettiNumbers()[2], 2)
        self.assertEqual(cobordism.Signature().compute(spacetime), 0.0)
        form = list(chain.intersectionForm())
        self.assertEqual(len(form), 4)
        diag0, off01, off10, diag1 = form
        self.assertAlmostEqual(diag0, 0.0, places=6)
        self.assertAlmostEqual(diag1, 0.0, places=6)
        self.assertAlmostEqual(off01, off10, places=6)        # symmetric
        self.assertGreater(abs(off01), 1e-6)                  # nonzero crossing
        self.assertLess(diag0 * diag1 - off01 * off10, 0.0)   # det < 0 (indefinite)

    def test_intersection_form_is_symmetric_and_sized_by_b2(self):
        spacetime = _build(_s2_cross_s2())
        chain = cobordism.ChainComplex.fromSpacetime(spacetime)
        b2 = chain.bettiNumbers()[2]
        form = list(chain.intersectionForm())
        self.assertEqual(len(form), b2 * b2)
        for i in range(b2):
            for j in range(b2):
                self.assertAlmostEqual(form[i * b2 + j], form[j * b2 + i], places=6)

    def test_zero_below_dimension_four(self):
        # The signature here is defined for 4-manifolds; lower dimensions give 0.
        for topology in (_sphere(1), _sphere(2), _sphere(3),
                         tessera.RealProjectivePlane(), _torus(), _s1_cross_s2()):
            with self.subTest(manifold=type(topology).__name__):
                self.assertEqual(cobordism.Signature().compute(_build(topology)), 0.0)


class TestCharacteristicNumbers(unittest.TestCase):

    def test_four_manifolds_have_signature_and_pontryagin(self):
        for topology in (_sphere(4), _s2_cross_s2()):
            with self.subTest(manifold=type(topology).__name__):
                numbers = cobordism.CharacteristicNumbers.of(_build(topology))
                self.assertIsNotNone(numbers.signature)
                # Hirzebruch signature theorem: p1 = 3 * signature.
                self.assertEqual(numbers.pontryagin_numbers["p1"], 3 * numbers.signature)

    def test_euler_agrees_with_observable(self):
        euler = cobordism.EulerCharacteristic()
        for topology in (_sphere(2), _sphere(4), _ball(4),
                         tessera.RealProjectivePlane(), _s2_cross_s2()):
            with self.subTest(manifold=type(topology).__name__):
                spacetime = _build(topology)
                numbers = cobordism.CharacteristicNumbers.of(spacetime)
                self.assertEqual(float(numbers.euler), euler.compute(spacetime))

    def test_below_dimension_four_has_no_signature_or_pontryagin(self):
        for topology in (tessera.RealProjectivePlane(), _torus(), _sphere(3)):
            with self.subTest(manifold=type(topology).__name__):
                numbers = cobordism.CharacteristicNumbers.of(_build(topology))
                self.assertIsNone(numbers.signature)
                self.assertEqual(dict(numbers.pontryagin_numbers), {})

    def test_stiefel_whitney_numbers_pending(self):
        # Documented as not-yet-computed; the map is currently empty.
        numbers = cobordism.CharacteristicNumbers.of(_build(_s2_cross_s2()))
        self.assertEqual(dict(numbers.stiefel_whitney_numbers), {})


if __name__ == "__main__":
    unittest.main()
