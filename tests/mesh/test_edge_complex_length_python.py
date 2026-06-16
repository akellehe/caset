# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Causal character is read from the (complex) edge LENGTH, not the fragile
`sign(l^2)`: an edge is timelike iff its length has a nonzero imaginary part.
`getLength()` is the metric DOF (distinct from the U(1) `getPhase()`), real for
spacelike, imaginary for timelike, and squares back to the squared length."""

import unittest

from tessera import Vertex, Edge


def _edge(sq):
    # the Edge ctor takes l^2 directly (stored exactly); the complex length is
    # derived as its sqrt (real = spacelike, imaginary = timelike)
    return Edge(Vertex(1, [0.0, 0.0, 0.0, 0.0]),
                Vertex(2, [0.0, 0.0, 0.0, 1.0]), complex(sq, 0.0))


class EdgeComplexLengthTest(unittest.TestCase):
    def test_spacelike_has_real_length(self):
        e = _edge(25.0)
        L = e.getLength()
        self.assertAlmostEqual(L.real, 5.0, places=12)
        self.assertAlmostEqual(L.imag, 0.0, places=12)
        self.assertTrue(e.isSpacelike())
        self.assertFalse(e.isTimelike())
        self.assertFalse(e.isNull())

    def test_timelike_has_imaginary_length(self):
        e = _edge(-4.0)
        L = e.getLength()
        self.assertAlmostEqual(L.real, 0.0, places=12)
        self.assertAlmostEqual(L.imag, 2.0, places=12)   # length = i*2
        self.assertTrue(e.isTimelike())
        self.assertFalse(e.isSpacelike())
        self.assertFalse(e.isNull())

    def test_null_has_zero_length(self):
        e = _edge(0.0)
        self.assertAlmostEqual(abs(e.getLength()), 0.0, places=12)
        self.assertTrue(e.isNull())
        self.assertFalse(e.isTimelike())
        self.assertFalse(e.isSpacelike())

    def test_squaring_length_recovers_squared_length(self):
        for sq in (25.0, -4.0, 1.0, -1.0, 100.0, -0.5):
            length = _edge(sq).getLength()
            sq2 = length * length
            self.assertAlmostEqual(sq2.real, sq, places=9)
            self.assertAlmostEqual(sq2.imag, 0.0, places=12)

    def test_causal_character_is_a_length_test_not_a_magnitude_test(self):
        # tiny-but-nonzero squared lengths still resolve cleanly by imaginary part
        self.assertTrue(_edge(-1e-6).isTimelike())
        self.assertTrue(_edge(1e-6).isSpacelike())


if __name__ == "__main__":
    unittest.main()
