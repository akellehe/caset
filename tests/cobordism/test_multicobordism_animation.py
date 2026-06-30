# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Smoke test for the two-step Proton animation (#522).

Drives a few optimization steps of the proton's Step A (recombination) and Step B
(formation) nodes headless (Agg) and checks that the per-step history advances across
the node boundary, the MDS layout produces 2-D coordinates, and a GIF is written. The
animation only reads the public `Proton` (node factories) + `MultiCobordism` API, so this
also guards that single-step `run_stage1`/`run_stage2` keep advancing the optimizer state
and that the proton's whole topology grows from a single simplex.
"""
import importlib.util
import os
import sys
import tempfile
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


class MultiCobordismAnimationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mca = _load("multicobordism_animation")

    def test_steps_advance_across_both_nodes(self):
        # Two Proton nodes (Step A, Step B); per_node = stage1_steps + stage2_iters = 4,
        # so node 0 is frames 0-3 (3 surgery + 1 relax) and node 1 begins at frame 4.
        nodes = self.mca.build_proton_nodes(seed=3)
        anim = self.mca.ProtonAnimator(nodes, stage1_steps=3, stage2_iters=1)
        for f in range(6):                  # all of node 0, then 2 surgery frames of node 1
            anim._advance(f)
        self.assertEqual(len(anim.hist["F"]), 6)
        for key in ("gradN2", "rU", "b3", "holes", "stage", "node"):
            self.assertEqual(len(anim.hist[key]), 6)
        # surgery×3 then relax on node 0, then node 1's surgery
        self.assertEqual(anim.hist["stage"], [1, 1, 1, 2, 1, 1])
        self.assertEqual(anim.hist["node"], [0, 0, 0, 0, 1, 1])
        self.assertEqual(anim._boundaries, [4])   # Step B began at step index 4
        self.assertTrue(all(isinstance(v, float) for v in anim.hist["F"]))

    def test_default_is_no_visualization(self):
        # visualize defaults OFF: run_build takes the fast batched path (one run_stage1 +
        # one run_stage2 per node) and returns a list of (label, metrics) — one entry per
        # step (Step A, Step B) — rather than an animation.
        nodes = self.mca.build_proton_nodes(seed=3)
        res = self.mca.run_build(nodes, stage1_steps=2, stage2_iters=1)
        self.assertEqual(len(res), 2)
        for label, metrics in res:
            self.assertIsInstance(label, str)
            for key in ("F", "gradN2", "rU", "b3", "holes"):
                self.assertIn(key, metrics)

    def test_mds_layout_is_2d(self):
        nodes = self.mca.build_proton_nodes(seed=3)
        coords = self.mca._mds_layout(nodes[0][0].st)   # the single-Δ⁴ seed (5 vertices)
        self.assertGreater(len(coords), 0)
        for v in coords.values():
            self.assertEqual(v.shape, (2,))

    def test_headless_save_writes_gif(self):
        nodes = self.mca.build_proton_nodes(seed=3)
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "proton.gif")
            self.mca.animate(nodes, save=out, stage1_steps=2, stage2_iters=1)
            self.assertTrue(os.path.exists(out))
            self.assertGreater(os.path.getsize(out), 0)


if __name__ == "__main__":
    unittest.main()
