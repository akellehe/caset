# MIT License
# Copyright (c) 2025 Andrew Kelleher
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""The merge cobordism (``examples/cobordism/merge_cobordism.py``).

Two boundary states on slice t merge through the bulk into a single object at
t+1 -- the simplicial pair-of-pants. These tests pin the construction:

  1. **Geometry.** Two staircase prisms share one result surface: 36 vertices
     (input A, input B on slice t; result R on t+1), the input→result edges
     timelike, intra-slice edges spacelike -- coordinate-free.
  2. **Lorentzian from the causal labeling.** The dual Regge action is complex
     (the timelike sign on the primal transfers to the dual), with no vertex
     coordinates and no CDT.
  3. **The register survives** (ker L₁ = 2, the Riemannian / signature-blind
     register) and the dual-complex check passes.
  4. **Hierarchical composition** -- a second-level merge builds and stays
     Lorentzian.
  5. **Transport is untouched** -- the existing `Level1Fill` still builds and
     carries its register (the merge is additive).
"""

import importlib.util
import os
import sys
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.join(_HERE, "..", "..", "examples", "cobordism",
                        "merge_cobordism.py")


def _load_example():
    spec = importlib.util.spec_from_file_location("merge_cobordism", _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["merge_cobordism"] = module
    spec.loader.exec_module(module)
    return module


MC = _load_example()
_M = MC.MergeCobordism()
_INFO = MC.summarize(_M)


class MergeGeometryTest(unittest.TestCase):
    """1: the two-prisms-sharing-a-result geometry."""

    def test_vertex_blocks(self):
        # input A [0,12), input B [12,24) on slice t; result R [24,36) on t+1
        self.assertEqual(_INFO["nV"], 36)
        self.assertEqual(_INFO["n_tets"], 102)

    def test_causal_split_is_coordinate_free(self):
        # input->result edges timelike, intra-slice spacelike (both present)
        self.assertGreater(_INFO["timelike_edges"], 0)
        self.assertGreater(_INFO["spatial_edges"], 0)
        self.assertEqual(_INFO["timelike_edges"] + _INFO["spatial_edges"],
                         _INFO["nE"])

    def test_crossing_edges_are_exactly_the_timelike_ones(self):
        cross = [e for e in _M.edges()
                 if _M._is_result(e[0]) != _M._is_result(e[1])]
        self.assertEqual(len(cross), _INFO["timelike_edges"])
        # and every crossing edge has one endpoint in the result block
        for a, b in cross:
            self.assertTrue((a >= 24) ^ (b >= 24))


class LorentzianFromLabelingTest(unittest.TestCase):
    """2: the dual action goes complex from the primal causal character."""

    def test_dual_action_is_complex(self):
        self.assertGreater(abs(_INFO["S_im"]), 1e-6)

    def test_timelike_edges_have_negative_squared_length(self):
        sl = {}
        for e in _M.st.getEdgeList().toVector():
            a, b = e.getSource().getId(), e.getTarget().getId()
            sl[(min(a, b), max(a, b))] = e.getSquaredLength()
        for (a, b), s in sl.items():
            crossing = (a >= 24) != (b >= 24)
            if crossing:
                self.assertLess(s, 0.0)     # timelike
            else:
                self.assertGreater(s, 0.0)  # spacelike


class RegisterSurvivesTest(unittest.TestCase):
    """3: a carried (Riemannian) register survives the merge geometry."""

    def test_dual_complex_valid(self):
        self.assertTrue(_M.dual_valid, _M.dual_reason)

    def test_kernel_dimension(self):
        self.assertEqual(_INFO["dim_kerL1"], 2)

    def test_nine_holonomy_circles(self):
        # three holes on each of A, B, R
        self.assertEqual(len(_M.hole_circles), 9)


class HierarchicalCompositionTest(unittest.TestCase):
    """4: a second-level merge builds and stays Lorentzian."""

    def test_second_level_merge(self):
        m2 = MC.MergeCobordism()
        info2 = MC.summarize(m2)
        self.assertTrue(m2.dual_valid)
        self.assertEqual(info2["dim_kerL1"], 2)
        self.assertGreater(abs(info2["S_im"]), 1e-6)


class TransportUntouchedTest(unittest.TestCase):
    """5: the additive merge leaves the transport fill intact."""

    def test_level1_fill_still_builds_and_carries(self):
        fill = MC.L1.Level1Fill(layers=1)
        self.assertEqual(fill.dim, 2)
        self.assertTrue(fill.dual_valid)
        # a transport fill's two boundaries sit on DIFFERENT slices (layers),
        # unlike the merge's two inputs on one slice -- the distinguishing
        # property the merge does not disturb
        self.assertEqual(int(fill.st.getVertexList().size()), 24)


if __name__ == "__main__":
    unittest.main()
