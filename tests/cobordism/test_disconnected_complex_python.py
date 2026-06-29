# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""A deliberately DISCONNECTED complex (b0 > 1) must flow through every objective
primitive without raising.

The isolated-block construction (#506) splices each solved input/output block into the
host as its own DISCONNECTED component, so the betti read, the emergent-hole read, the
Regge action+gradient, and — critically — the `dualComplexValid` move gate must all
tolerate b0 > 1. This pins that contract: two vertex-disjoint 4-simplices give b0 = 2,
and none of the primitives blow up.
"""
import math
import unittest

import tessera as T

cob = T.cobordism
_DIM = 4


def _two_disjoint_simplices():
    """Two vertex-disjoint 4-simplices: a complex with b0 = 2 (two components)."""
    cells = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    st = T.Spacetime.fromCells(_DIM, cells, 1.0, 0.0)
    # Generic non-degenerate edge lengths, mirroring build_closed_s4 so the Regge
    # primitives have well-defined hinge areas to integrate.
    for i, e in enumerate(st.getEdgeList().toVector()):
        e.setSquaredLength(1.0 + 0.01 * (i % 6))
    return st


class DisconnectedComplexTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.st = _two_disjoint_simplices()
        cls.CXX = cob.MultiCobordism

    def test_complex_is_actually_disconnected(self):
        # Guard the premise: if this stops being b0 > 1 the test is vacuous.
        self.assertEqual(list(self.CXX.betti(self.st))[0], 2)

    def test_betti_runs_on_b0_gt_1(self):
        b = list(self.CXX.betti(self.st))
        self.assertEqual(b[0], 2)
        self.assertTrue(all(math.isfinite(x) for x in b))

    def test_emergent_holes_runs_on_b0_gt_1(self):
        # Pure read; must not raise on a multi-component boundary.
        holes = [tuple(h) for h in self.CXX.emergent_holes(self.st, 3)]
        self.assertEqual(len(holes), 2)  # one hole per disjoint top cell

    def test_regge_action_and_gradient_run_on_b0_gt_1(self):
        rs = T.ReggeSolver(self.st, T.MatterConfiguration())
        self.assertTrue(math.isfinite(rs.reggeAction()))
        grad = list(rs.actionGradientExact())
        self.assertTrue(grad)
        self.assertTrue(all(math.isfinite(abs(z)) for z in grad))

    def test_dual_complex_valid_accepts_b0_gt_1(self):
        # The move gate must ACCEPT a disconnected complex — splicing blocks in as
        # disconnected components (#506) depends on this.
        ok, _why = cob.EigenstateSynthesis(self.st, 3).dualComplexValid()
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
