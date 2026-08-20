# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Focused correctness tests for the experimental one-step proton animation."""

import cmath
import importlib.util
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import matplotlib
import numpy as np

matplotlib.use("Agg")

_EX = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "cobordism")


def _load():
    sys.path.insert(0, _EX)
    path = os.path.join(_EX, "proton_animation.py")
    spec = importlib.util.spec_from_file_location("direct_proton_animation", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Vertex:
    def __init__(self, identifier):
        self.identifier = identifier

    def getId(self):
        return self.identifier


class _Cell:
    def __init__(self, *vertices):
        self.vertices = [_Vertex(identifier) for identifier in vertices]

    def getVertices(self):
        return self.vertices


class _Edge:
    def __init__(self, source, target, squared_length):
        self.source = _Vertex(source)
        self.target = _Vertex(target)
        self.length = cmath.sqrt(squared_length)

    def getSource(self):
        return self.source

    def getTarget(self):
        return self.target

    def getLength(self):
        return self.length


class _Vector:
    def __init__(self, values):
        self.values = values

    def toVector(self):
        return self.values


class _Spacetime:
    def __init__(self, cells, squared_lengths=None):
        self.cells = [_Cell(*cell) for cell in cells]
        squared_lengths = squared_lengths or {}
        self.edges = [_Edge(source, target, value)
                      for (source, target), value in squared_lengths.items()]

    def getTopSimplices(self):
        return self.cells

    def getEdgeList(self):
        return _Vector(self.edges)


class ProtonAnimationCorrectnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pa = _load()

    def test_preimport_threads_accepts_split_and_equals_forms(self):
        environment = {"OMP_NUM_THREADS": "3"}
        self.assertEqual(
            self.pa._configure_preimport_threads(["p.py", "--threads", "7"], environment),
            "7")
        self.assertEqual(environment["OMP_NUM_THREADS"], "7")

        environment = {"OMP_NUM_THREADS": "3"}
        self.assertEqual(
            self.pa._configure_preimport_threads(["p.py", "--threads=9"], environment),
            "9")
        self.assertEqual(environment["OMP_NUM_THREADS"], "9")

    def test_invalid_thread_value_preserves_existing_environment(self):
        environment = {"OMP_NUM_THREADS": "3"}
        effective = self.pa._configure_preimport_threads(
            ["p.py", "--threads=0"], environment)
        self.assertEqual(effective, "3")

    def test_schedule_rejects_nonpositive_chunks_instead_of_hanging(self):
        with self.assertRaisesRegex(ValueError, "init_chunk must be positive"):
            self.pa.ProtonAnimator._make_schedule(1, 1, 0, 0, 1)
        with self.assertRaisesRegex(ValueError, "evolve_chunk must be positive"):
            self.pa.ProtonAnimator._make_schedule(1, 1, 1, 0, -1)
        with self.assertRaisesRegex(ValueError, "n_nodes must be positive"):
            self.pa.ProtonAnimator._make_schedule(0, 1, 1, 1, 1)

    def test_retries_require_one_engine_iteration_per_frame(self):
        with self.assertRaisesRegex(ValueError, "one run iteration per frame"):
            self.pa.ProtonAnimator([(object(), "node")], init_steps=2, init_chunk=2,
                                   evolve_steps=0, max_lookahead_tries=2)

    def test_run_build_rejects_relaxation_chunk_for_interleaved_drive(self):
        with self.assertRaisesRegex(ValueError, "requires no_combinatorial_moves"):
            self.pa.run_build([(object(), "unused")], relax_chunk=2)

    def test_complex_gram_diagnostic_is_scale_invariant_and_not_real_projected(self):
        # Orthogonal tetrahedron multiplied by i: G = i I. Its real projection
        # is identically zero, but the complex induced metric is nondegenerate.
        base = {(0, 1): 1j, (0, 2): 1j, (0, 3): 1j,
                (1, 2): 2j, (1, 3): 2j, (2, 3): 2j}
        first = self.pa._min_abs_gram_dets(_Spacetime([(0, 1, 2, 3)], base))
        scaled = self.pa._min_abs_gram_dets(
            _Spacetime([(0, 1, 2, 3)], {edge: 17 * value
                                        for edge, value in base.items()}))
        self.assertGreater(first[0], 0.0)
        self.assertGreater(first[1], 0.0)
        np.testing.assert_allclose(first, scaled, rtol=1e-12, atol=1e-12)

    def test_curvature_cache_refreshes_on_commits_and_topology_change_once_per_frame(self):
        animator = self.pa.ProtonAnimator.__new__(self.pa.ProtonAnimator)
        animator.hist = {"F": [1.0], "lookahead": [0]}
        animator._active = 0
        animator._frames = 10
        animator._curv_cache = {}
        animator._cell_curvature = mock.Mock(return_value={})
        spacetime = _Spacetime([(0, 1, 2, 3, 4)])

        animator._cell_curvature_cached(0, spacetime)
        animator._cell_curvature_cached(0, spacetime)
        self.assertEqual(animator._cell_curvature.call_count, 1)

        spacetime.cells.append(_Cell(0, 1, 2, 3, 5))
        animator.hist["F"].append(0.9)
        animator.hist["lookahead"].append(0)
        animator._cell_curvature_cached(0, spacetime)
        animator._cell_curvature_cached(0, spacetime)
        self.assertEqual(animator._cell_curvature.call_count, 2)

        animator.hist["F"].append(0.8)
        animator.hist["lookahead"].append(1)
        animator._cell_curvature_cached(0, spacetime)
        animator._cell_curvature_cached(0, spacetime)
        self.assertEqual(animator._cell_curvature.call_count, 3)

        animator.hist["F"].append(0.7)
        animator.hist["lookahead"].append(0)
        self.assertEqual(animator._curvature_age_tag(0), "  (heat frame 3)")

    def test_relaxation_only_paint_does_not_report_a_stage1_stall(self):
        animator = self.pa.ProtonAnimator.__new__(self.pa.ProtonAnimator)
        animator._done = False
        animator._frames = 2
        animator.lookahead_depth = 10
        animator.hist = {"lookahead": [None], "tries": [1]}
        animator.fig = mock.Mock()
        animator._redraw = mock.Mock()
        animator._draw_extras = mock.Mock()
        animator._frame_label = mock.Mock(return_value="fixed topology")

        animator._paint(0)
        title = animator.fig.suptitle.call_args.args[0]
        self.assertIn("stage-1 disabled", title)
        self.assertNotIn("stalled", title)

    def test_target_verdict_does_not_claim_numerical_convergence(self):
        animator = self.pa.ProtonAnimator.__new__(self.pa.ProtonAnimator)
        animator.nodes = [(SimpleNamespace(last_stage2_stationary=True), "node")]
        tag = animator._verdict_tag(True, 0.01, 3)
        self.assertIn("TARGET CARRIED", tag)
        self.assertIn("optimizer stationary=yes", tag)
        self.assertNotIn("CONVERGED", tag)

    def test_headless_save_advances_each_scheduled_frame_once(self):
        pa = self.pa

        class FakeAnimator:
            _TITLE_PREFIX = "test animation"

            def __init__(self, _nodes, **_kw):
                self._frames = 3
                self.hist = {"frames": []}

            def _setup(self, plt):
                self.fig, self.ax = plt.subplots()

            def update(self, frame):
                self.hist["frames"].append(frame)
                self.ax.clear()
                self.ax.plot([0, frame])
                return []

        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "animation.gif")
            with mock.patch.object(pa, "ProtonAnimator", FakeAnimator):
                animator = pa.animate([], save=output, visualize=False)
            self.assertEqual(animator.hist["frames"], [0, 1, 2])
            self.assertGreater(os.path.getsize(output), 0)


if __name__ == "__main__":
    unittest.main()
