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

"""Causal character is read from the (complex) edge LENGTH, not the fragile
`sign(l^2)`: an edge is timelike iff its length has a nonzero imaginary part.
`getLength()` is the metric DOF (distinct from the U(1) `getPhase()`), real for
spacelike, imaginary for timelike, and squares back to the squared length."""

import unittest

from tessera import Vertex, Edge


def _edge(sq):
    return Edge(Vertex(1, [0.0, 0.0, 0.0, 0.0]),
                Vertex(2, [0.0, 0.0, 0.0, 1.0]), sq)


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
            sq2 = _edge(sq).getLength() ** 2
            self.assertAlmostEqual(sq2.real, sq, places=9)
            self.assertAlmostEqual(sq2.imag, 0.0, places=12)

    def test_causal_character_is_a_length_test_not_a_magnitude_test(self):
        # tiny-but-nonzero squared lengths still resolve cleanly by imaginary part
        self.assertTrue(_edge(-1e-6).isTimelike())
        self.assertTrue(_edge(1e-6).isSpacelike())


if __name__ == "__main__":
    unittest.main()
