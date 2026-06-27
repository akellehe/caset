# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Smoke test for the real-time MultiCobordism animation (#493).

Drives a few optimization steps headless (Agg) and checks that the per-step
history advances, the MDS layout produces 2-D coordinates, and a GIF is written.
The animation only reads the public MultiCobordism API, so this also guards that
single-step `run_stage1`/`relax_stage2` keep advancing the optimizer state.
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
        cls.opt = cls.mca.build_demo_merge(seed=3, n_refine=12)

    def test_steps_advance_history(self):
        anim = self.mca.MultiCobordismAnimator(
            self.opt, stage1_steps=3, stage2_iters=3)
        for f in range(4):
            anim._advance(f)
        self.assertEqual(len(anim.hist["F"]), 4)
        # every recorded metric series advances in lock-step
        for key in ("gradN2", "rU", "b3", "holes", "stage"):
            self.assertEqual(len(anim.hist[key]), 4)
        # stage flips from 1 (surgery) to 2 (relaxation) at the boundary
        self.assertEqual(anim.hist["stage"], [1, 1, 1, 2])
        self.assertTrue(all(isinstance(v, float) for v in anim.hist["F"]))

    def test_default_is_no_visualization(self):
        # visualize defaults to OFF: run_optimization takes the fast batched path
        # (one run_stage1 + one relax_stage2, no per-step plotting) and returns the
        # final metrics dict rather than an animation.
        res = self.mca.run_optimization(self.opt, stage1_steps=2, stage2_iters=2)
        self.assertIsInstance(res, dict)
        for key in ("F", "gradN2", "rU", "b3", "holes"):
            self.assertIn(key, res)

    def test_mds_layout_is_2d(self):
        coords = self.mca._mds_layout(self.opt.st)
        self.assertGreater(len(coords), 0)
        for v in coords.values():
            self.assertEqual(v.shape, (2,))

    def test_headless_save_writes_gif(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "anim.gif")
            self.mca.animate(self.opt, save=out, stage1_steps=2, stage2_iters=2)
            self.assertTrue(os.path.exists(out))
            self.assertGreater(os.path.getsize(out), 0)


if __name__ == "__main__":
    unittest.main()
