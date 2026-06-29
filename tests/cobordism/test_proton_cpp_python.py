# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The canonical two-step Proton build (#503): a footgun-free MultiCobordism proton.

A proton is three quarks in a colorless bound state, so it is built in two steps:
Step A (recombination) makes a *colored* diquark `{1, ω}`; Step B (formation) makes
the proton *color singlet* `{1, ω, ω²}`. These tests confirm the constants and that
`Proton` actually assembles a converged proton — its formation block carries the
singlet with at least three emergent color holes, on the relaxed (non-unit) metric.
"""
import cmath
import math
import unittest

import pytest

import tessera


def _close(a, b, tol=1e-9):
    return abs(a - b) < tol


class ProtonConstantsTest(unittest.TestCase):
    """Fast: the baked-in color constants, no build."""

    def test_omega_is_cube_root_of_unity(self):
        w = tessera.cobordism.Proton.omega()
        self.assertTrue(_close(w, cmath.exp(2j * math.pi / 3)))
        self.assertTrue(_close(w * w * w, 1.0))            # ω³ = 1
        self.assertTrue(_close(1 + w + w * w, 0.0))        # 1 + ω + ω² = 0 (colorless)

    def test_singlet_is_one_omega_omegasq(self):
        s = tessera.cobordism.Proton.singlet()
        w = tessera.cobordism.Proton.omega()
        self.assertEqual(len(s), 3)
        self.assertTrue(_close(s[0], 1.0))
        self.assertTrue(_close(s[1], w))
        self.assertTrue(_close(s[2], w * w))


@pytest.mark.slow
class ProtonBuildTest(unittest.TestCase):
    """Slow: the real two-step emergent build (Step A then Step B, restarts)."""

    # (seed, params) pinned to a configuration verified to converge: seed 3 at
    # refinement 14 with 30 stage-1 steps reaches the proton singlet (colorR ~1e-31,
    # 3 holes) on the first attempt in ~35s. max_restarts is headroom for round-off
    # differences on other machines; the happy path stops at the first attempt.
    SEED = 3
    REFINE = 14
    MAX_RESTARTS = 8
    STAGE1_STEPS = 30
    STAGE1_CANDIDATES = 8
    STAGE2_ITERS = 10
    COLOR_TOL = 0.5

    @classmethod
    def setUpClass(cls):
        cls.p = tessera.cobordism.Proton(seed=cls.SEED, host_refinement=cls.REFINE)
        cls.p.build(max_restarts=cls.MAX_RESTARTS, stage1_max_steps=cls.STAGE1_STEPS,
                    stage1_candidates=cls.STAGE1_CANDIDATES,
                    stage2_max_iters=cls.STAGE2_ITERS,
                    color_tolerance=cls.COLOR_TOL, min_quark_holes=3)

    def test_proton_converged(self):
        self.assertTrue(
            self.p.converged(),
            f"proton did not converge: colorR={self.p.color_residual()}, "
            f"holes={len(self.p.quark_holes())}")

    def test_block_carries_the_singlet(self):
        # Step B's proton block carries the 3-vector color singlet (r_state → 0).
        self.assertLess(self.p.color_residual(), self.COLOR_TOL)

    def test_block_has_at_least_three_quark_holes(self):
        self.assertGreaterEqual(len(self.p.quark_holes()), 3)

    def test_block_carries_the_relaxed_metric(self):
        # The block is carved with the relaxed metric copied in, NOT the unit-metric
        # subOf: at least one edge length must differ from the unit 1.0+0j.
        block = self.p.block()
        self.assertIsNotNone(block)
        squared = [e.getSquaredLength() for e in block.getEdgeList().toVector()]
        self.assertTrue(squared, "block has no edges")
        self.assertTrue(any(abs(l - complex(1.0, 0.0)) > 1e-9 for l in squared),
                        "block metric is unit — relaxed lengths were not copied in")

    def test_step_a_diquark_recombination_ran(self):
        # Step A's r_U is finite (the diquark recombination was co-optimized).
        self.assertTrue(math.isfinite(self.p.diquark_residual()))


if __name__ == "__main__":
    unittest.main()
