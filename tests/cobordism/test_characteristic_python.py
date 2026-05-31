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

"""Capability A — characteristic numbers (#65).

Euler characteristic and signature as Observables, plus the CharacteristicNumbers
aggregate (Pontryagin p1 = 3*sigma). The signature is validated on the
intersection form: S^4 (empty), S^2 x S^2 (hyperbolic -> sigma 0, rank 2).
The sigma = +1 case (CP^2) and Stiefel-Whitney numbers are pending follow-ups.
"""

import unittest

import tessera

cob = tessera.cobordism


def _build(topology):
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, topology)
    st.build()
    return st


def _s2xs2():
    return tessera.SimplicialProduct(tessera.SimplexBoundarySphere(2),
                                     tessera.SimplexBoundarySphere(2))


class TestEulerCharacteristic(unittest.TestCase):

    def test_matches_known_values(self):
        cases = [
            (tessera.SimplexBoundarySphere(1), 0),   # S^1
            (tessera.SimplexBoundarySphere(2), 2),   # S^2
            (tessera.SimplexBoundarySphere(3), 0),   # S^3
            (tessera.SimplexBoundarySphere(4), 2),   # S^4
            (tessera.SolidSimplex(4), 1),            # D^4
            (tessera.RealProjectivePlane(), 1),      # RP^2
            (_s2xs2(), 4),                           # S^2 x S^2
        ]
        chi = cob.EulerCharacteristic()
        for topology, expected in cases:
            with self.subTest(topology=type(topology).__name__):
                self.assertEqual(chi.compute(_build(topology)), float(expected))


class TestSignature(unittest.TestCase):

    def test_sphere_signature_zero(self):
        # S^4 has b_2 = 0, so the intersection form is empty and sigma = 0.
        st = _build(tessera.SimplexBoundarySphere(4))
        self.assertEqual(cob.Signature().compute(st), 0.0)
        cc = cob.ChainComplex.fromSpacetime(st)
        self.assertEqual(cc.bettiNumbers()[2], 0)
        self.assertEqual(list(cc.intersectionForm()), [])

    def test_s2xs2_is_hyperbolic(self):
        # S^2 x S^2: intersection form is the hyperbolic form [[0,1],[1,0]] ->
        # rank 2, signature 0. This exercises the cup product non-trivially.
        st = _build(_s2xs2())
        cc = cob.ChainComplex.fromSpacetime(st)
        self.assertEqual(cc.bettiNumbers()[2], 2)
        self.assertEqual(cob.Signature().compute(st), 0.0)
        Q = list(cc.intersectionForm())
        self.assertEqual(len(Q), 4)  # 2x2
        # Hyperbolic: zero diagonal, equal nonzero off-diagonal, det < 0.
        a, b, c, d = Q
        self.assertAlmostEqual(a, 0.0, places=6)
        self.assertAlmostEqual(d, 0.0, places=6)
        self.assertAlmostEqual(b, c, places=6)
        self.assertGreater(abs(b), 1e-6)
        self.assertLess(a * d - b * c, 0.0)  # nondegenerate, indefinite

    def test_signature_undefined_below_dim4(self):
        # Signature is only defined (here) for n = 4; lower dims give 0.
        for topology in (tessera.SimplexBoundarySphere(2),
                         tessera.RealProjectivePlane()):
            with self.subTest(topology=type(topology).__name__):
                self.assertEqual(cob.Signature().compute(_build(topology)), 0.0)


class TestCharacteristicNumbers(unittest.TestCase):

    def test_dim4_fills_signature_and_pontryagin(self):
        for topology in (tessera.SimplexBoundarySphere(4), _s2xs2()):
            with self.subTest(topology=type(topology).__name__):
                cn = cob.CharacteristicNumbers.of(_build(topology))
                self.assertIsNotNone(cn.signature)
                # Hirzebruch signature theorem: <p_1,[K]> = 3 sigma.
                self.assertEqual(cn.pontryagin["p1"], 3 * cn.signature)

    def test_euler_matches_observable(self):
        for topology in (tessera.SimplexBoundarySphere(2), _s2xs2(),
                         tessera.RealProjectivePlane()):
            with self.subTest(topology=type(topology).__name__):
                st = _build(topology)
                cn = cob.CharacteristicNumbers.of(st)
                self.assertEqual(float(cn.euler),
                                 cob.EulerCharacteristic().compute(st))

    def test_below_dim4_has_no_signature(self):
        cn = cob.CharacteristicNumbers.of(_build(tessera.RealProjectivePlane()))
        self.assertIsNone(cn.signature)
        self.assertEqual(dict(cn.pontryagin), {})


if __name__ == "__main__":
    unittest.main()
