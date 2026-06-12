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

"""Hierarchical synthesis, level 1
(``examples/cobordism/level1_fill_realizability.py``).

Two completed level-0 registers serve as the boundary carriers of a new
3-dimensional fill; the level-1 state space is the pair of end periods
V (+) V, and a fill realizes the transport u' iff graph(u') lies in the
restriction R of its ker L_1. These tests pin the hand-derivable structure:

  1. **No bulk, no interaction.** The disjoint union is saturated
     (ker L_1 = V (+) V): every transport trivially "carried".
  2. **The level-1 anchor.** The trivial fill (prism W x I) keeps
     ker L_1 two-dimensional, conserves each end's signed charge, has
     R = the diagonal -- the graph of the identity -- and its battery
     realizes EXACTLY the identity, with every floored transport
     certified by a period leak.
  3. **Mapping-class transport.** The gamma- and gamma^2-twisted fills
     realize exactly their hole 3-cycles: the C_3 (the hole triple's full
     setwise stabilizer) is transported by twisting, and nothing else is.
  4. **Rigidity under interior topology change.** Gated interior surgery
     and stellar growth on the 3-layer fill (thin prisms have no interior
     tets -- every tet spans adjacent layers) leave the transport at the
     identity: the level-1 realizable set on prism-class fills is the C_3
     of mapping-class transport.
"""

import importlib.util
import os
import sys
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.join(_HERE, "..", "..", "examples", "cobordism",
                        "level1_fill_realizability.py")


def _load_example():
    spec = importlib.util.spec_from_file_location("level1_fill_realizability",
                                                  _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["level1_fill_realizability"] = module
    spec.loader.exec_module(module)
    return module


L1 = _load_example()

# Shared fixtures: the trivial fill and its battery (deterministic builds).
_CYL = L1.Level1Fill(layers=1)
_ROWS = L1.level1_battery(_CYL)


class NoBulkNoInteractionTest(unittest.TestCase):
    """1: the disconnected union is the saturated control."""

    def test_union_is_saturated(self):
        uc = L1.union_control()
        self.assertEqual(uc["dim"], 4)
        self.assertTrue(uc["saturated"])


class TrivialFillAnchorTest(unittest.TestCase):
    """2: the level-1 T1 anchor on the prism W x I."""

    def test_fill_keeps_a_valid_dual_complex(self):
        self.assertTrue(_CYL.dual_valid, _CYL.dual_reason)

    def test_ker_l1_is_two_dimensional(self):
        self.assertEqual(_CYL.dim, 2)
        self.assertEqual(_CYL.rank, 2)

    def test_both_ends_conserve_signed_charge(self):
        self.assertLess(_CYL.end_charge_leak(), 1e-9)

    def test_restriction_is_the_diagonal(self):
        dev = float(np.max(np.abs(_CYL.P6[:, 0:3] * _CYL.sign0
                                  - _CYL.P6[:, 3:6] * _CYL.sign1)))
        self.assertLess(dev, 1e-9)

    def test_emergent_gate_is_the_identity(self):
        self.assertEqual(L1.match_gate(_CYL.emergent_gate()), "Identity")

    def test_battery_realizes_exactly_the_identity(self):
        realized = [r["gate"] for r in _ROWS if r["realizable"]]
        self.assertEqual(realized, ["Identity"])

    def test_identity_residual_is_machine_zero(self):
        row = next(r for r in _ROWS if r["gate"] == "Identity")
        self.assertLess(row["residual"], L1.REALIZE)

    def test_floored_transports_sit_above_the_floor_with_leaks(self):
        for row in _ROWS:
            if row["realizable"]:
                continue
            self.assertGreater(row["residual"], L1.CERT_FLOOR, msg=row["gate"])
            self.assertGreater(row["leak"], 1e-6, msg=row["gate"])


class MappingClassTransportTest(unittest.TestCase):
    """3: twisted fills realize exactly their hole 3-cycles."""

    def _twist_case(self, twist, expected):
        fill = L1.Level1Fill(layers=1, twist=twist)
        self.assertTrue(fill.dual_valid, fill.dual_reason)
        self.assertEqual(fill.dim, 2)
        self.assertEqual(L1.match_gate(fill.emergent_gate()), expected)
        realized = [r["gate"] for r in L1.level1_battery(fill)
                    if r["realizable"]]
        self.assertEqual(realized, [expected])

    def test_gamma_twist_realizes_its_three_cycle(self):
        self._twist_case(L1._GAMMA, "3-cycle (0312)")

    def test_gamma_squared_twist_realizes_the_other_three_cycle(self):
        self._twist_case(L1._compose(L1._GAMMA, L1._GAMMA), "3-cycle (0231)")


class InteriorTopologyRigidityTest(unittest.TestCase):
    """4: interior room exists only at three layers, and gated cuts/growth
    leave the transport at the identity."""

    def test_thin_fills_have_no_interior_tets(self):
        self.assertEqual(len(list(_CYL.es.interiorTopCells())), 0)
        two = L1.Level1Fill(layers=2)
        self.assertEqual(len(list(two.es.interiorTopCells())), 0)

    def test_three_layer_fill_has_interior_room(self):
        thick = L1.Level1Fill(layers=3)
        self.assertGreaterEqual(len(list(thick.es.interiorTopCells())), 3)

    def test_gated_cuts_preserve_the_identity_transport(self):
        fill = L1.Level1Fill(layers=3)
        sites = sorted(tuple(sorted(int(v) for v in c))
                       for c in fill.es.interiorTopCells())
        cut = 0
        for cell in sites[:2]:
            if fill.es.removeInteriorCell(list(cell)):
                ok, _why = fill.es.dualComplexValid()
                if not ok:
                    fill.es.restoreLastRemoval()
                    continue
                cut += 1
        fill.read_spectral()
        self.assertGreaterEqual(cut, 1)
        self.assertTrue(fill.dual_valid, fill.dual_reason)
        self.assertEqual(fill.dim, 2)
        self.assertEqual(L1.match_gate(fill.emergent_gate()), "Identity")

    def test_gated_growth_preserves_the_identity_transport(self):
        fill = L1.Level1Fill(layers=3, grow_vertices=2, grow_seed=7)
        self.assertGreaterEqual(fill.grown, 1)
        self.assertTrue(fill.dual_valid, fill.dual_reason)
        self.assertEqual(fill.dim, 2)
        self.assertEqual(L1.match_gate(fill.emergent_gate()), "Identity")


if __name__ == "__main__":
    unittest.main()
