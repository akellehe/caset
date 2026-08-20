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

    def test_wick_fit_recovers_a_common_interval_axis(self):
        intervals = {(0, 1): 1.0j, (0, 2): 2.0j, (1, 2): 3.0j}
        phase, projected, off_axis = self.pa._fit_real_interval_axis(intervals)
        self.assertAlmostEqual(phase, np.pi / 2.0, places=12)
        self.assertAlmostEqual(off_axis, 0.0, places=12)
        np.testing.assert_allclose([value.real for value in projected.values()],
                                   [1.0, 2.0, 3.0], atol=1e-12)
        np.testing.assert_allclose([value.imag for value in projected.values()],
                                   0.0, atol=1e-12)

    def test_projected_gram_signature_distinguishes_euclidean_and_lorentzian(self):
        cell = (0, 1, 2, 3, 4)

        def intervals(points, metric):
            result = {}
            for first in range(len(points)):
                for second in range(first + 1, len(points)):
                    delta = points[first] - points[second]
                    result[(first, second)] = delta @ metric @ delta
            return result

        origin = np.zeros(4)
        euclidean_points = np.vstack((origin, np.eye(4)))
        lorentzian_points = np.vstack((origin, np.diag([2.0, 1.0, 1.0, 1.0])))
        euclidean = intervals(euclidean_points, np.eye(4))
        lorentzian = intervals(lorentzian_points, np.diag([-1.0, 1.0, 1.0, 1.0]))
        self.assertEqual(
            self.pa._gram_signature(self.pa._simplex_gram(cell, euclidean)),
            ((0, 0, 4), "euclidean"))
        self.assertEqual(
            self.pa._gram_signature(self.pa._simplex_gram(cell, lorentzian)),
            ((1, 0, 3), "lorentzian"))

    def test_spacetime_development_glues_two_simplex_charts_exactly(self):
        points = np.asarray([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, -1.0],
        ])
        intervals = {}
        for first in range(len(points)):
            for second in range(first + 1, len(points)):
                delta = points[first] - points[second]
                intervals[(first, second)] = np.dot(delta, delta)
        cells = [(0, 1, 2, 3, 4), (0, 1, 2, 3, 5)]
        developed = self.pa._develop_spacetime_data(cells, intervals, [(0, 1)])
        self.assertEqual(developed["signature_names"], ["euclidean", "euclidean"])
        self.assertEqual(developed["physical"].shape, (2, 4))
        self.assertLess(developed["local_residual"], 1e-10)
        self.assertLess(developed["closure_residual"], 1e-10)
        self.assertTrue(np.all(np.isfinite(developed["physical"])))

    def test_spacetime_layout_removes_observer_axis_flips(self):
        cells = [(index,) for index in range(5)]
        physical = np.asarray([
            [-1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 1.0],
            [-0.5, 0.5, 0.5, 0.5],
        ])
        rotation = np.asarray([[0.0, -1.0, 0.0],
                               [1.0, 0.0, 0.0],
                               [0.0, 0.0, 1.0]])
        changed = np.column_stack((-physical[:, 0], physical[:, 1:] @ rotation))
        layout = self.pa._StableSpacetimeLayout(ease=1.0)
        expected = layout.coords(cells, physical)
        actual = layout.coords(cells, changed)
        for cell in cells:
            np.testing.assert_allclose(actual[cell], expected[cell], atol=1e-12)

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
