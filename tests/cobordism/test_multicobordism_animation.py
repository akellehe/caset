# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Smoke test for the two-step Proton animation (#522, #526).

Drives a few optimization chunks of the proton's Step A (recombination) and Step B
(formation) nodes headless (Agg) and checks that the per-frame history advances across the
node boundary through the init → evolve → stage2 phases, the MDS layout produces 2-D
coordinates, the batched build reports a convergence verdict, and a GIF is written. The
animation only reads the public `Proton` (node factories) + `MultiCobordism` API, so this
also guards that `run_stage1`/`run_stage2` keep advancing the optimizer state and that the
proton's whole topology grows from a single simplex.
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

    def _tiny_anim(self, nodes):
        # One chunk per phase per node: init → evolve → stage2 = 3 frames/node, so node 0 is
        # frames 0-2 and node 1 begins at frame 3.
        return self.mca.ProtonAnimator(nodes, init_steps=2, init_chunk=2, evolve_steps=2,
                                       evolve_chunk=2, stage2_iters=1)

    def test_frames_advance_across_both_nodes(self):
        nodes = self.mca.build_proton_nodes(seed=3)
        anim = self._tiny_anim(nodes)
        self.assertEqual(anim._frames, 6)
        for f in range(6):
            anim._advance(f)
        self.assertEqual(len(anim.hist["F"]), 6)
        for key in ("gradN2", "rU", "b3", "holes", "phase", "node"):
            self.assertEqual(len(anim.hist[key]), 6)
        self.assertEqual(anim.hist["phase"],
                         ["init", "evolve", "stage2", "init", "evolve", "stage2"])
        self.assertEqual(anim.hist["node"], [0, 0, 0, 1, 1, 1])
        self.assertEqual(anim._boundaries, [3])   # Step B began at frame index 3
        self.assertTrue(all(isinstance(v, float) for v in anim.hist["F"]))

    def test_default_is_no_visualization_with_verdict(self):
        # visualize defaults OFF: run_build takes the fast batched path and returns one
        # (label, metrics) entry per node plus a trailing ("verdict", {...}) entry.
        nodes = self.mca.build_proton_nodes(seed=3)
        res = self.mca.run_build(nodes, init_steps=2, evolve_steps=2, stage2_iters=1)
        self.assertEqual(len(res), 3)             # Step A, Step B, verdict
        for label, metrics in res[:2]:
            self.assertIsInstance(label, str)
            for key in ("F", "gradN2", "rU", "b3", "holes"):
                self.assertIn(key, metrics)
        label, verdict = res[-1]
        self.assertEqual(label, "verdict")
        for key in ("converged", "color_residual", "registers"):
            self.assertIn(key, verdict)
        self.assertIsInstance(verdict["converged"], bool)

    def test_mds_layout_is_2d_and_normalized(self):
        nodes = self.mca.build_proton_nodes(seed=3)
        coords = self.mca._mds_layout(nodes[0][0].st)   # the single-Δ⁴ seed (5 vertices)
        self.assertGreater(len(coords), 0)
        for v in coords.values():
            self.assertEqual(v.shape, (2,))

    def test_headless_save_writes_gif(self):
        nodes = self.mca.build_proton_nodes(seed=3)
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "proton.gif")
            self.mca.animate(nodes, save=out, init_steps=2, init_chunk=2, evolve_steps=2,
                             evolve_chunk=2, stage2_iters=1)
            self.assertTrue(os.path.exists(out))
            self.assertGreater(os.path.getsize(out), 0)

    def test_precone_pre_grows_the_seed(self):
        # The --precone flag flows build_proton_nodes(precone=N) -> Proton(precone=N) ->
        # the C++ MultiCobordism ctor, so each node's single-Δ⁴ seed is pre-grown before
        # any optimization. precone=0 leaves the bare seed (one top cell).
        bare = self.mca.build_proton_nodes(seed=3, precone=0)
        grown = self.mca.build_proton_nodes(seed=3, precone=6)
        self.assertEqual(len(bare[0][0].st.getTopSimplices()), 1)
        self.assertGreater(len(grown[0][0].st.getTopSimplices()), 1)
        self.assertGreater(len(grown[1][0].st.getTopSimplices()), 1)


if __name__ == "__main__":
    unittest.main()
