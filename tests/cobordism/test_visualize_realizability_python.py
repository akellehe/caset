# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Smoke test for examples/cobordism/visualize_realizability.py (#166).

Asserts the realizability-visualization script runs end-to-end and writes a
non-empty composite PNG. Image bytes are NOT committed -- the test renders into
a throwaway temp dir and only checks the produced file path is non-empty and the
file is on disk with content (per the project convention: the script is the
committed artifact, the PNG is attached to the PR).
"""

import importlib.util
import os
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPT = os.path.normpath(os.path.join(
    _HERE, "..", "..", "examples", "cobordism", "visualize_realizability.py"))


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "visualize_realizability", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class VisualizeRealizabilitySmokeTest(unittest.TestCase):
    """The script renders the four-panel composite for a realizable U."""

    def test_script_exists(self):
        self.assertTrue(os.path.isfile(_SCRIPT), _SCRIPT)

    def test_render_writes_nonempty_png(self):
        mod = _load_module()
        with tempfile.TemporaryDirectory() as out_dir:
            res = mod.render(out_dir=out_dir)

            # The pipeline realized U: the four panels have a coherent story to
            # tell (r -> 0 on the glued bulk).
            self.assertTrue(res["realizable"])
            self.assertLess(res["residual"], 1e-10)

            # A non-empty PNG path was produced and the file is on disk.
            png = res["composite"]
            self.assertTrue(png)                          # path is non-empty
            self.assertTrue(png.endswith(".png"))
            self.assertTrue(os.path.isfile(png))
            self.assertGreater(os.path.getsize(png), 0)   # non-empty bytes
            # Written under the requested (temp) directory, not the repo tree.
            self.assertEqual(os.path.dirname(os.path.abspath(png)),
                             os.path.abspath(out_dir))

    def test_pipeline_objects_are_distinct_manifolds(self):
        # The three synthesized objects exist and A, B are visibly distinct
        # boundary manifolds (different |V|), and W_AB has a filled interior.
        mod = _load_module()
        res = mod.run_pipeline()
        self.assertGreaterEqual(res["stA"].getVertexCount(), 2)
        self.assertGreaterEqual(res["stB"].getVertexCount(), 2)
        self.assertNotEqual(res["stA"].getVertexCount(),
                            res["stB"].getVertexCount())
        # W_AB carries at least one interior cell to "fill" (vertex or edge).
        self.assertGreater(res["witness"].getVertexCount(), 0)
        self.assertGreaterEqual(res["interior_vertex_count"], 1)


if __name__ == "__main__":
    unittest.main()
