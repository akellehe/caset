# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Smoke test for the ProtonIngredients (emergent-arm) animation (#555).

Drives a few optimization chunks of the emergent arm's Step A (the canonical
recombination node) and Step B (formation with NOTHING pinned) headless (Agg) and checks
that the per-frame history advances through the init → evolve → stage2 phases across the
node boundary, the inherited panels draw, and the verdict is OBSERVATIONAL — a
(stationary, singlet_diagnostic, holes) read with no pass/fail gate on the singlet. The
charts are `multicobordism_animation.py`'s, reused by subclassing, so this also guards
that the two examples stay panel-compatible.
"""
import importlib.util
import os
import sys
import unittest

import matplotlib

matplotlib.use("Agg")

_EX = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "cobordism")


def _load(name):
    sys.path.insert(0, _EX)
    spec = importlib.util.spec_from_file_location(name, os.path.join(_EX, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class EmergentProtonAnimationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ep = _load("emergent_proton")

    def _tiny_anim(self, nodes):
        # One chunk per phase per node: init → evolve → stage2 = 3 frames/node.
        return self.ep.EmergentProtonAnimator(nodes, init_steps=2, init_chunk=2,
                                              evolve_steps=2, evolve_chunk=2,
                                              stage2_iters=1)

    def test_step_b_pins_nothing(self):
        nodes = self.ep.build_ingredients_nodes(seed=3)
        self.assertEqual(len(nodes), 2)
        self.assertEqual(len(nodes[1][0].outputs), 0)   # the emergent arm's one delta

    def test_frames_advance_and_verdict_is_observational(self):
        nodes = self.ep.build_ingredients_nodes(seed=3)
        anim = self._tiny_anim(nodes)
        self.assertEqual(anim._frames, 6)
        for f in range(6):
            anim._advance(f)
        self.assertEqual(len(anim.hist["F"]), 6)
        self.assertEqual(anim.hist["phase"],
                         ["init", "evolve", "stage2", "init", "evolve", "stage2"])
        self.assertEqual(anim.hist["node"], [0, 0, 0, 1, 1, 1])
        self.assertEqual(anim._boundaries, [3])          # Step B began at frame index 3
        stationary, singlet_diagnostic, holes = anim.verdict()
        self.assertIsInstance(stationary, bool)
        self.assertIsInstance(singlet_diagnostic, float)
        self.assertIsInstance(holes, int)

    def test_panels_draw_headless(self):
        import matplotlib.pyplot as plt
        nodes = self.ep.build_ingredients_nodes(seed=3)
        anim = self._tiny_anim(nodes)
        anim._setup(plt)
        for f in range(anim._frames):                    # update() advances AND redraws
            anim.update(f)
        title = anim.fig._suptitle.get_text()
        self.assertIn("ProtonIngredients", title)
        # observational, not a gate; the #586 verdict tag spells the singlet
        # diagnostic "r_state(singlet)=", in BOTH title branches (#588)
        self.assertIn("r_state(singlet)", title)
        plt.close(anim.fig)

    def test_fast_path_reports_what_emerged(self):
        nodes = self.ep.build_ingredients_nodes(seed=3)
        result = self.ep.run_build(nodes, visualize=False, init_steps=2, evolve_steps=2,
                                   stage2_iters=1)
        labels = [label for label, _metrics in result]
        self.assertEqual(len(labels), 3)                 # step A, step B, what emerged
        self.assertEqual(labels[-1], "what emerged")
        summary = result[-1][1]
        for key in ("stationary", "persistent", "persistence_passes", "registers",
                    "b3", "singlet_diagnostic"):
            self.assertIn(key, summary)


if __name__ == "__main__":
    unittest.main()
