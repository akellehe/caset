# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The canonical two-step Proton build (#503): a footgun-free MultiCobordism proton.

A proton is three quarks in a colorless bound state, so it is built in two steps:
Step A (recombination) makes a *colored* diquark `{1, ω}`; Step B (formation) makes
the proton *color singlet* `{1, ω, ω²}`. These tests confirm the constants and that
`Proton` actually assembles a converged proton — the whole formation cobordism
carries the singlet with at least three emergent color holes, on the relaxed
(non-unit) metric.
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
    """Slow: the real two-step emergent build (Step A recombination then Step B
    formation, with restarts)."""

    # Single-Δ⁴-simplex-seed build (one pentatope; nothing is pre-built — all topology
    # emerges from one simplex via the trap door). It converges reliably for this seed,
    # but the runtime is VARIABLE: threaded eigensolves reorder floating-point sums, so
    # the "best move" — and thus how soon r_U carries — varies run to run (this build can
    # finish in ~1 min or take several). max_restarts is cross-machine headroom.
    SEED = 1
    MAX_RESTARTS = 2
    INIT_STEPS = 180
    EVOLVE_STEPS = 60
    STAGE1_CANDIDATE_MOVES = 8
    STAGE2_ITERS = 10
    COLOR_TOL = 0.5

    @classmethod
    def setUpClass(cls):
        cls.p = tessera.cobordism.Proton(seed=cls.SEED)
        cls.p.build(max_restarts=cls.MAX_RESTARTS, init_steps=cls.INIT_STEPS,
                    evolve_steps=cls.EVOLVE_STEPS,
                    stage1_candidate_moves=cls.STAGE1_CANDIDATE_MOVES,
                    stage2_max_iters=cls.STAGE2_ITERS,
                    color_tolerance=cls.COLOR_TOL, min_quark_holes=3)

    def test_proton_converged(self):
        self.assertTrue(
            self.p.converged(),
            f"proton did not converge: colorR={self.p.color_residual()}, "
            f"holes={len(self.p.quark_holes())}")

    def test_whole_cobordism_carries_the_singlet(self):
        # The proton is the harmonic of the WHOLE relaxed step-B cobordism: its singlet
        # residual r_state({1, ω, ω²}) → 0 (the inputs are held by their own residual;
        # the bulk evolves to carry the colorless 3-vector).
        self.assertLess(self.p.color_residual(), self.COLOR_TOL)

    def test_has_at_least_three_quark_holes(self):
        self.assertGreaterEqual(len(self.p.quark_holes()), 3)

    def test_proton_carries_the_relaxed_metric(self):
        # block() IS the whole relaxed cobordism (not a unit-metric carve): at least one
        # edge length must differ from the unit 1.0+0j.
        block = self.p.block()
        self.assertIsNotNone(block)
        squared = [e.getSquaredLength() for e in block.getEdgeList().toVector()]
        self.assertTrue(squared, "proton has no edges")
        self.assertTrue(any(abs(l - complex(1.0, 0.0)) > 1e-9 for l in squared),
                        "proton metric is unit — the relaxed geometry was lost")

    def test_step_a_diquark_recombination_ran(self):
        # Step A's r_U is finite (the diquark recombination was co-optimized) — a
        # separate physical claim from the proton's formation.
        self.assertTrue(math.isfinite(self.p.diquark_residual()))


if __name__ == "__main__":
    unittest.main()
