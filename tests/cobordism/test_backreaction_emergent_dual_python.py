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

"""Backreaction on the merge substrate
(``examples/cobordism/backreaction_emergent_dual.py``).

The carried state's stress-energy shifts the action-selected,
damping-regulated emergent dual. These tests pin the well-posed picture:

  1. **Realizable, Lorentzian carriers.** Every merge carrier in the family
     realizes the merge (period residual machine-zero — realizability is
     topological), and the dual Regge action is complex (Lorentzian, the
     spacelike-hinge boosts).
  2. **The conformal runaway.** With action + damping only (kappa=0) the
     emergent dual sits on the grid edge — Re S and |Im S| fall monotonically
     with the conformal scale, so neither pins it.
  3. **The matter is the regulator.** The matter energy E has an interior
     minimum (the restoring force), and it sources the worldtube scale far
     more than the bulk scale (charge curves the fill near the charge).
  4. **It pins and shifts the emergent dual.** A finite coupling pins the
     emergent dual to a finite interior geometry, its worldtube scale tracks
     the matter's preferred scale, and it shifts with kappa.
"""

import importlib.util
import os
import sys
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.join(_HERE, "..", "..", "examples", "cobordism",
                        "backreaction_emergent_dual.py")


def _load_example():
    spec = importlib.util.spec_from_file_location("backreaction_emergent_dual",
                                                  _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["backreaction_emergent_dual"] = module
    spec.loader.exec_module(module)
    return module


BE = _load_example()
# one shared scan (deterministic; the construction has no randomness)
_ED = BE.EmergentDual()
_SCAN = _ED.scan()
_GRID = BE.GRID
_LO, _HI = _GRID[0], _GRID[-1]


def _is_edge(w):
    return w[0] in (_LO, _HI) or w[1] in (_LO, _HI)


class RealizableLorentzianTest(unittest.TestCase):
    """1: every carrier is a realizable, Lorentzian merge."""

    def test_all_carriers_realizable(self):
        self.assertLess(max(v["residual"] for v in _SCAN.values()), 1e-9)

    def test_action_is_complex_everywhere(self):
        self.assertGreater(min(abs(v["ImS"]) for v in _SCAN.values()), 1e-6)


class ConformalRunawayTest(unittest.TestCase):
    """2: action + damping alone don't pin the scale."""

    def test_kappa0_on_the_edge(self):
        self.assertTrue(_is_edge(BE.EmergentDual.select(_SCAN, 0.0)))

    def test_ReS_falls_with_scale(self):
        # Re S more negative at the large-scale corner than the small one
        big = _SCAN[(round(float(_HI), 3), round(float(_HI), 3))]["ReS"]
        small = _SCAN[(round(float(_LO), 3), round(float(_LO), 3))]["ReS"]
        self.assertLess(big, small)


class MatterRegulatorTest(unittest.TestCase):
    """3: the matter has an interior minimum and is worldtube-dominated."""

    def test_energy_interior_minimum(self):
        wmin = min(_SCAN, key=lambda k: _SCAN[k]["E"])
        self.assertNotIn(wmin[0], (_LO, _HI))
        self.assertNotIn(wmin[1], (_LO, _HI))

    def test_matter_sources_worldtube_more_than_bulk(self):
        mid = _GRID[len(_GRID) // 2]
        e_wt = [_SCAN[(round(float(s), 3), round(float(mid), 3))]["E"] for s in _GRID]
        e_bulk = [_SCAN[(round(float(mid), 3), round(float(s), 3))]["E"] for s in _GRID]
        span_wt = max(e_wt) - min(e_wt)
        span_bulk = max(e_bulk) - min(e_bulk)
        self.assertGreater(span_wt, 3 * span_bulk)


class EmergentDualShiftTest(unittest.TestCase):
    """4: the matter pins and shifts the emergent dual."""

    @classmethod
    def setUpClass(cls):
        cls.traj = [(k, BE.EmergentDual.select(_SCAN, k))
                    for k in (0.0, 300.0, 2500.0, 5000.0)]

    def test_finite_coupling_pins_an_interior_geometry(self):
        self.assertTrue(any(not _is_edge(w) for _k, w in self.traj))

    def test_emergent_dual_shifts_with_kappa(self):
        self.assertNotEqual(self.traj[0][1], self.traj[-1][1])

    def test_pinned_worldtube_scale_tracks_the_matter_minimum(self):
        wEmin = min(_SCAN, key=lambda k: _SCAN[k]["E"])
        final = self.traj[-1][1]
        self.assertLessEqual(abs(final[0] - wEmin[0]),
                             (_GRID[1] - _GRID[0]) + 1e-9)


if __name__ == "__main__":
    unittest.main()
