# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The CDT temporal orientation class (renamed from SimplexOrientation, #412).

`TemporalOrientation` is the Ambjorn-Loll causal (time-slice) orientation: the
`(t_i, t_f)` split = how many of a simplex's vertices lie on the initial vs the
final time slice. It is purely TEMPORAL -- it carries no spatial +-1 orientation
sign (that is `ChainComplex::endSignCovector`, the induced spatial orientation).
The rename disentangles the two notions that were both called "orientation".
"""

import unittest

import tessera


class TemporalOrientationTest(unittest.TestCase):
    def test_renamed_class_is_exposed(self):
        self.assertTrue(hasattr(tessera, "TemporalOrientation"))
        # the old, conflated name is gone
        self.assertFalse(hasattr(tessera, "SimplexOrientation"))

    def test_constructs_and_reports_the_time_split(self):
        # (t_i, t_f) = (2 initial-slice vertices, 1 final-slice vertex): a (3,1) CDT
        # tetrahedron face split.
        o = tessera.TemporalOrientation(2, 1)
        self.assertEqual(tuple(o.numeric()), (2, 1))

    def test_orientation_is_purely_temporal(self):
        # The class encodes only the (t_i, t_f) time-slice split -- the three causal
        # cases t_i > t_f, t_i == t_f, t_i < t_f -- and carries no spatial sign.
        for ti, tf in [(2, 1), (1, 1), (1, 2)]:
            self.assertEqual(tuple(tessera.TemporalOrientation(ti, tf).numeric()),
                             (ti, tf))

    def test_flip_swaps_initial_and_final(self):
        o = tessera.TemporalOrientation(3, 1)
        # equality is by (t_i, t_f); a (3,1) and a (1,3) are distinct orientations.
        self.assertNotEqual(o, tessera.TemporalOrientation(1, 3))
        self.assertEqual(o, tessera.TemporalOrientation(3, 1))


if __name__ == "__main__":
    unittest.main()
