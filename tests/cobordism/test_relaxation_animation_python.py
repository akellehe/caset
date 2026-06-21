# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Smoke test for the W_ABC relaxation animation (#361).

Exercises the capture + render pipeline of
``examples/cobordism/wabc_relaxation_animation.py`` on a couple of low-iteration
frames: the stepped relax reads the emergent geometry, the frames render to valid
RGBA images, and a null (photon) edge emerges as the Lorentzian worldlines relax.
"""

import importlib.util
import os
import pathlib
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")  # headless

import numpy as np

_EXAMPLE = (pathlib.Path(__file__).resolve().parents[2]
            / "examples" / "cobordism" / "wabc_relaxation_animation.py")
_spec = importlib.util.spec_from_file_location("wabc_relaxation_animation", _EXAMPLE)
_anim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_anim)


class RelaxationAnimationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.iters = [0, 2]
        (cls.geoms, cls.trace, cls.layout,
         cls.windows, cls.stride) = _anim.capture(cls.iters)

    def test_capture_yields_a_frame_per_iteration(self):
        self.assertEqual(len(self.geoms), len(self.iters))
        self.assertEqual(len(self.trace), len(self.iters))

    def test_geometry_is_read_off_the_relaxed_complex(self):
        edges, defmap, k, resid, nnull = self.geoms[-1]
        self.assertGreater(len(edges), 0)          # edges with l^2 read off the complex
        self.assertGreater(len(defmap), 0)         # per-hinge curvature read off the complex
        self.assertTrue(np.isfinite(resid))

    def test_photon_null_edge_emerges_under_relax(self):
        # Lorentzian worldlines start timelike; a worldline relaxing through l^2~0 is a
        # null edge (the photon). The null count is 0 at the seed and grows.
        null_by_iter = {t[0]: t[2] for t in self.trace}
        self.assertEqual(null_by_iter[0], 0)
        self.assertGreater(max(t[2] for t in self.trace), 0)

    def test_both_views_render_to_valid_rgba_images(self):
        for view in ("full", "pants"):
            with tempfile.TemporaryDirectory() as d:
                out = os.path.join(d, f"anim_{view}.gif")
                frames = _anim.render(self.geoms, self.trace, self.layout,
                                      self.windows, self.stride, out, view=view)
                self.assertEqual(len(frames), len(self.iters), view)
                for f in frames:
                    self.assertEqual(f.ndim, 3, view)
                    self.assertEqual(f.shape[2], 4, view)   # RGBA
                    self.assertGreater(f.shape[0] * f.shape[1], 0, view)
                self.assertTrue(os.path.exists(out) and os.path.getsize(out) > 0, view)

    def test_pants_view_uses_a_clean_sphere_embedding(self):
        # the base surface embeds onto ~the unit sphere (clearer than the prism blob)
        scoords, sidx = _anim._sphere_coords(self.geoms[-1][0], self.stride)
        radii = np.linalg.norm(scoords, axis=1)
        self.assertTrue(np.allclose(radii, 1.0, atol=1e-6))
        self.assertEqual(len(sidx), self.stride)        # all base vertices embedded


if __name__ == "__main__":
    unittest.main()
