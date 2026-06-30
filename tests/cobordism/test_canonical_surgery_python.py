# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The canonical directed surgery lives on MultiCobordism (#549).

`build_step` (the policy hook) and the directed cone-out/cone-in probes — the unit a
search policy (Proton's build, a greedy driver, or the RL agent) composes — are methods
of the *engine* (`MultiCobordism`), not of `Proton`. The consolidation removed the
`Proton` duplicate; `Proton` keeps only the `should_use_directed_surgery` flag and calls
the engine. These tests pin the API surface, the consolidation, the probes' `rU`-monotone
invariant (they only ever commit moves that lower `rU`), and an end-to-end converged
proton driven through the canonical directed path.
"""
import unittest

import pytest

import tessera

cob = tessera.cobordism


class CanonicalSurgeryApiTest(unittest.TestCase):
    """Fast: the canonical surgery API lives on MultiCobordism, not Proton."""

    def test_multicobordism_exposes_buildstep_and_probes(self):
        for name in ("build_step", "directed_cone_out", "directed_cone_in"):
            self.assertTrue(hasattr(cob.MultiCobordism, name),
                            f"MultiCobordism.{name} missing")

    def test_build_action_enum_has_every_solve_action(self):
        ba = cob.MultiCobordism.BuildAction
        for name in ("GROW", "EVOLVE", "RELAX", "CONE_OUT", "CONE_IN"):
            self.assertTrue(hasattr(ba, name), f"BuildAction.{name} missing")

    def test_hole_placement_strategy_enum(self):
        hp = cob.MultiCobordism.HolePlacementStrategy
        for name in ("ADJACENT_HOLES_FIRST", "ADJACENT_HOLES_LAST"):
            self.assertTrue(hasattr(hp, name), f"HolePlacementStrategy.{name} missing")

    def test_proton_surgery_api_was_consolidated_away(self):
        # The directed surgery + buildStep moved DOWN to the engine; Proton only drives it.
        for name in ("BuildAction", "HolePlacementStrategy", "build_step",
                     "directed_cone_out", "directed_cone_in"):
            self.assertFalse(
                hasattr(cob.Proton, name),
                f"Proton.{name} should be gone (canonical home is MultiCobordism)")

    def test_pinned_boundary_vertices_is_engine_internal(self):
        # Reverted to private once the directed probe moved into the engine — only the move
        # gate and the probe consult it, both internal to MultiCobordism.
        self.assertFalse(hasattr(cob.MultiCobordism, "pinned_boundary_vertices"))


@pytest.mark.slow
class DirectedProbeInvariantTest(unittest.TestCase):
    """Slow (grows a seeded node): the directed probes commit a move ONLY when it lowers
    `rU`, so `rU` is non-increasing across either probe, and `build_step` dispatches every
    action without raising."""

    def test_probes_are_ru_monotone_and_build_step_dispatches(self):
        BA = cob.MultiCobordism.BuildAction
        HP = cob.MultiCobordism.HolePlacementStrategy
        node = cob.Proton(seed=0).formation_node(1)   # a seeded single-Δ⁴ node
        node.build_step(BA.GROW, max_steps=60, n_candidate_moves=8, patience=15)

        before = node.r_u(node.st)
        opened = node.directed_cone_out(HP.ADJACENT_HOLES_LAST)
        self.assertGreaterEqual(opened, 0)
        self.assertLessEqual(node.r_u(node.st), before + 1e-6,
                             "directed_cone_out must not raise rU (commits only openers that lower it)")

        before = node.r_u(node.st)
        closed = node.directed_cone_in()
        self.assertGreaterEqual(closed, 0)
        self.assertLessEqual(node.r_u(node.st), before + 1e-6,
                             "directed_cone_in must not raise rU")

        # Every BuildAction dispatches without raising.
        node.build_step(BA.RELAX, stage2_max_iters=3)
        node.build_step(BA.EVOLVE, max_steps=5)
        node.build_step(BA.CONE_OUT, hole_placement_strategy=HP.ADJACENT_HOLES_FIRST)
        node.build_step(BA.CONE_IN)


@pytest.mark.slow
class DirectedSurgeryProtonBuildTest(unittest.TestCase):
    """Slow: a full two-step Proton build driven through the canonical directed-surgery
    path (`should_use_directed_surgery=True`) converges — the whole formation cobordism
    carries the singlet with at least three emergent color holes."""

    @classmethod
    def setUpClass(cls):
        cls.p = cob.Proton(seed=0, should_use_directed_surgery=True)
        cls.p.build(max_restarts=3, init_steps=180, evolve_steps=60,
                    stage2_max_iters=10, color_tolerance=0.5, min_quark_holes=3)

    def test_converged_via_canonical_path(self):
        self.assertTrue(
            self.p.converged(),
            f"did not converge: colorR={self.p.color_residual()}, "
            f"holes={len(self.p.quark_holes())}")

    def test_three_quark_holes(self):
        self.assertGreaterEqual(len(self.p.quark_holes()), 3)

    def test_singlet_carried(self):
        self.assertLess(self.p.color_residual(), 0.5)


if __name__ == "__main__":
    unittest.main()
