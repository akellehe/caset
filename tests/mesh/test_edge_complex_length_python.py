# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Direct complex squared-length storage plus legacy presentation checks.

Causal character is read from the ARGUMENT of the squared edge LENGTH: an
edge is spacelike at `arg(l^2) ~ 0`, timelike at `~ +/-pi`, lightlike at
`~ +/-pi/2`, and mixed anywhere else (#870).
`squaredLength()` is the exact complex metric datum. `getLength()` is only a
legacy principal-root view and is not used as a replay identity."""

import cmath
import math
import unittest

from tessera import Vertex, Edge


def _edge(sq):
    # The Edge ctor stores the full complex squared length verbatim.
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

    def test_zero_length_is_degenerate_not_null(self):
        """An absent edge is DEGENERATE; a null edge is a lightlike ray.

        These were conflated while causal type was read from the Euclidean
        modulus, which can only vanish when the edge itself vanishes (#870).
        Reading `arg(l^2)` separates them -- see the lightlike test below.
        """
        e = _edge(0.0)
        self.assertAlmostEqual(abs(e.getLength()), 0.0, places=12)
        self.assertTrue(e.isDegenerate())
        self.assertFalse(e.isNull())
        self.assertFalse(e.isTimelike())
        self.assertFalse(e.isSpacelike())
        self.assertFalse(e.isMixed())

    def test_equal_parts_are_lightlike_on_an_edge_of_nonzero_extent(self):
        """`Re(l) == Im(l) > 0` is the non-trivial lightlike case.

        Asserts the argument and the nonzero extent SEPARATELY, so the test
        cannot pass by the edge having collapsed to nothing.
        """
        component = math.sqrt(0.5)
        e = _edge(1.0)
        e.setLength(complex(component, component))
        self.assertGreater(abs(e.getLength()), 0.5)          # not degenerate
        self.assertFalse(e.isDegenerate())
        self.assertAlmostEqual(e.squaredArgument(), math.pi / 2.0, places=12)
        self.assertTrue(e.isNull())
        self.assertFalse(e.isTimelike())
        self.assertFalse(e.isSpacelike())
        self.assertFalse(e.isMixed())

    def test_a_generic_argument_is_mixed(self):
        """A generic argument is NOT snapped to the nearest definite type."""
        e = _edge(1.0)
        e.setLength(cmath.exp(0.3j))
        self.assertTrue(e.isMixed())
        for definite in (e.isSpacelike(), e.isTimelike(), e.isNull(),
                         e.isDegenerate()):
            self.assertFalse(definite)

    def test_squaring_length_recovers_squared_length(self):
        for sq in (25.0, -4.0, 1.0, -1.0, 100.0, -0.5):
            length = _edge(sq).getLength()
            sq2 = length * length
            self.assertAlmostEqual(sq2.real, sq, places=9)
            self.assertAlmostEqual(sq2.imag, 0.0, places=12)

    def test_arbitrary_complex_squared_length_round_trips_exactly(self):
        z = complex(-2.75, 4.125)
        e = Edge(Vertex(7), Vertex(11), z)
        self.assertEqual(complex(e.squaredLength()), z)
        replacement = complex(0.375, -9.5)
        e.setSquaredLength(replacement)
        self.assertEqual(complex(e.squaredLength()), replacement)

    def test_causal_character_is_a_length_test_not_a_magnitude_test(self):
        # tiny-but-nonzero squared lengths still resolve cleanly by imaginary part
        self.assertTrue(_edge(-1e-6).isTimelike())
        self.assertTrue(_edge(1e-6).isSpacelike())


if __name__ == "__main__":
    unittest.main()
