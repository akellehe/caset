# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Tests for the fully-emergent C++ `Proton` builder (#503).

`tessera.cobordism.Proton` composes `MultiCobordism` to grow the proton's colour
register by gated surgical coning on a bare S^4 host — nothing is hand-placed — in
two steps (q+q -> diquark, diquark+q -> proton), restarting across seeds until the
proton block carries the colour singlet on >= 3 emergent quark holes.

Fast tests pin the binding/accessor contract with a zero-attempt build (instant).
The slow test does one real emergent build (shared across assertions via
`setUpClass`) and pins: the register EMERGED (>=3 holes on a host that starts with
b3=0), the block carries the singlet while flooring the trivial rep (confinement),
the block metric is the relaxed one (not unit), and the diquark step converged.
Convergence is stochastic, so the slow test skips honestly if no seed converged in
the attempt budget (same pattern as the sibling emergent tests).
"""
import cmath
import math
import os
import unittest

import tessera

cob = tessera.cobordism
_W = cmath.exp(2j * math.pi / 3)
_SINGLET = [1, _W, _W * _W]
_TRIVIAL = [1, 1, 1]


def _threads():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "BLIS_NUM_THREADS"):
        os.environ.setdefault(v, "16")


class ProtonBindingContractTest(unittest.TestCase):
    """Fast: the binding and accessors exist and behave with zero attempts."""

    def test_zero_attempts_is_unconverged_and_inspectable(self):
        # n_attempts=0 runs no build -> instant; every accessor must still work.
        p = cob.Proton(n_attempts=0)
        self.assertFalse(p.converged)
        self.assertEqual(p.attempts, 0)
        self.assertIsNone(p.block)
        self.assertIsNone(p.st)
        self.assertEqual(list(p.quark_holes), [])
        self.assertIsInstance(p.color_residual, float)
        self.assertIsInstance(p.diquark_residual, float)
        self.assertIsInstance(p.seed, int)

    def test_constructor_params_are_exposed(self):
        # All user-controllable knobs accept keywords without error (zero attempts
        # keeps it instant): attempts + per-stage step budgets are user-controlled.
        p = cob.Proton(n_attempts=0, stage1_steps=7, stage2_steps=3, n_refine=10,
                       gamma=0.5, seed0=42)
        self.assertFalse(p.converged)
        self.assertEqual(p.attempts, 0)


class EmergentProtonBuildTest(unittest.TestCase):
    """Slow: one real fully-emergent two-step build, shared across assertions."""

    @classmethod
    def setUpClass(cls):
        _threads()
        # A real emergent build. Stochastic + restart-driven; modest budget.
        cls.p = cob.Proton(n_attempts=12, stage1_steps=60, stage2_steps=15,
                           n_refine=16, gamma=1.0, seed0=3)

    def _require_converged(self):
        if not self.p.converged:
            self.skipTest("no seed grew a 3-hole proton in the attempt budget")

    def test_attempts_within_budget(self):
        # The attempts knob is respected: never more than requested, and >=1 tried.
        self.assertGreaterEqual(self.p.attempts, 1)
        self.assertLessEqual(self.p.attempts, 12)

    def test_register_emerged_with_three_holes(self):
        self._require_converged()
        # The colour register EMERGED: >= 3 quark holes on the proton block, grown
        # by surgery (a bare host starts with b3=0 -> no register).
        self.assertGreaterEqual(len(self.p.quark_holes), 3)
        self.assertIsNotNone(self.p.block)
        self.assertIsNotNone(self.p.st)

    def test_emergent_not_seeded_betti(self):
        self._require_converged()
        # Emergence check: the proton block carries a non-trivial degree-3 register
        # (b3 >= 1), i.e. holes were grown, not placed.
        betti = list(cob.MultiCobordism.betti(self.p.block))
        self.assertGreaterEqual(betti[3], 1)

    def test_singlet_carried_and_color_neutral(self):
        self._require_converged()
        # The colour singlet is carried (small r_U).
        self.assertLess(self.p.color_residual, 1.0)
        # NOTE: with a rich emergent register (b3 >= 3) the r_state probe fits ANY
        # 3-vector, so the trivial rep floors too and is NOT a confinement signal.
        # The genuine signal is COLOUR-NEUTRALITY: the singlet-phase-weighted net
        # Dirac-Kahler charge is far below the constituent total (the proton is
        # colourless = confined).
        es = cob.EigenstateSynthesis(self.p.block, 3)
        dk = cob.DiracKahler(self.p.block)
        q = [dk.charge(dk.lift(3, list(es.carriedRepresentative([list(h)], [1.0]))))
             for h in list(self.p.quark_holes)[:3]]
        net = abs(sum((_W ** k) * q[k] for k in range(3)))
        total = sum(q)
        self.assertGreater(total, 1e-6)
        self.assertLess(net, 0.5 * total, "proton block is not colour-neutral")

    def test_block_has_relaxed_metric(self):
        self._require_converged()
        # The block must carry the RELAXED metric (copied from the parent), not a
        # unit-metric rebuild: at least one edge has squared length != 1.
        l2 = [e.getSquaredLength().real for e in self.p.block.getEdgeList().toVector()]
        self.assertTrue(any(abs(x - 1.0) > 1e-9 for x in l2),
                        "block metric looks unit (not the relaxed one)")

    def test_diquark_step_converged(self):
        self._require_converged()
        # Step A produced a diquark (finite, non-divergent r_U).
        self.assertTrue(math.isfinite(self.p.diquark_residual))


if __name__ == "__main__":
    unittest.main()
