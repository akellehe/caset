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

"""The L_2 register on the hexagon-join S^3
(``examples/cobordism/l2_register_realizability.py``).

The register theorems are dimension-free, and these tests pin the degree-2
instance through the real pipeline (Spacetime construction, degree-2
EigenstateSynthesis surgery, HodgeLaplacian.harmonics(2)):

  1. **Emergence.** Opening the three vertex-disjoint tetrahedra grows b_2 and
     ker L_2 0 -> 2 (the staircase [0, 0, 1, 2]); the boundary-period
     constraint symmetrizes to Sigma = 0 (the all-ones covector).
  2. **The gate set is the conservation law, one degree up.** The 52-gate
     battery splits 13/39 with membership identical to the 2d register, and
     the spectral verdict agrees with the closed-form column-sum criterion on
     every gate.
  3. **H3 at the value level.** The anchor-normalized Gram is the identity
     (the hole triple is a full S_3 orbit of the join's automorphism group),
     Z_spec equals the operator amplitude at machine precision with the
     independent Choi reading agreeing, the value carries over exactly to the
     symmetric (9,9)-join variant, and a generic (8,8) draw deviates by
     exactly its Gram defect.
  4. **Additive growth.** The 3d composed stellar move (cone a tet's four
     faces, remove the parent) adds interior vertices on seeds with interior
     room, preserving ker L_2, the identity anchor, and the uniform metric.
"""

import importlib.util
import os
import sys
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.join(_HERE, "..", "..", "examples", "cobordism",
                        "l2_register_realizability.py")


def _load_example():
    spec = importlib.util.spec_from_file_location("l2_register_realizability",
                                                  _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["l2_register_realizability"] = module
    spec.loader.exec_module(module)
    return module


L2 = _load_example()

# One register, one battery sweep, and one H3 sweep shared by every test below
# (the construction is deterministic; building once keeps the suite fast).
_REG = L2.RegisterL2()
_TRACE = L2.register_emergence()
_ROWS = L2.gate_sweep(_REG)
_H3_ROWS, _INFO = L2.h3_value_sweep(_REG, n_states=4, seed=7)
_REALIZED = [r for r in _H3_ROWS if r["realizable"]]
_FLOORED = [r for r in _H3_ROWS if not r["realizable"]]


class RegisterEmergenceTest(unittest.TestCase):
    """1: surgery grows the degree-2 register out of the closed S^3."""

    def test_hexagon_join_counts(self):
        self.assertEqual(int(_REG.st.getVertexList().size()), 12)
        self.assertEqual(len(_REG.cells), 72)

    def test_ker_l2_emerges_0_to_2(self):
        self.assertEqual([t["kerL2"] for t in _TRACE], [0, 0, 1, 2])

    def test_b2_grows_0_to_2(self):
        self.assertEqual(_TRACE[0]["b2"], 0)
        self.assertEqual(_TRACE[-1]["b2"], 2)

    def test_register_is_two_dimensional_with_full_period_rank(self):
        self.assertEqual(_REG.dim, 2)
        self.assertEqual(_REG.rank, 2)

    def test_charge_constraint_is_the_all_ones_covector(self):
        self.assertTrue(np.allclose(np.abs(_REG.n), 1.0, atol=1e-9))

    def test_identity_floors_until_the_register_is_grown(self):
        anchor = L2.identity_anchor(_REG)
        for row in anchor[:-1]:
            self.assertFalse(row["realizable"])
            self.assertGreater(row["residual"], L2.CERT_FLOOR)
        self.assertTrue(anchor[-1]["realizable"])
        self.assertEqual(anchor[-1]["b2"], 2)


class GateSetIsTheConservationLawTest(unittest.TestCase):
    """2: the battery splits 13/39 exactly as at degree 1."""

    def test_realized_set_matches_the_canonical_thirteen(self):
        realized = sorted(r["gate"] for r in _ROWS if r["realizable"])
        self.assertEqual(realized, sorted(L2.CANONICAL_SET))

    def test_spectral_verdict_matches_the_criterion_on_every_gate(self):
        for row, (_name, U, _fam) in zip(_ROWS, L2.BASE._gates()):
            self.assertEqual(row["realizable"], L2.BASE.conserves_charge(U),
                             msg=row["gate"])

    def test_realized_residuals_are_machine_zero(self):
        for row in _ROWS:
            if row["realizable"]:
                self.assertLess(row["residual"], L2.REALIZE, msg=row["gate"])

    def test_floored_gates_sit_above_the_certificate_floor(self):
        for row in _ROWS:
            if not row["realizable"]:
                self.assertGreater(row["residual"], L2.CERT_FLOOR,
                                   msg=row["gate"])

    def test_every_floored_gate_is_certified_by_a_nonzero_leak(self):
        for row in _ROWS:
            if not row["realizable"]:
                self.assertGreater(row["leak"], 1e-6, msg=row["gate"])


class H3ValueLevelTest(unittest.TestCase):
    """3: the value identities, realized by the implementation at degree 2."""

    def test_metric_is_uniform_at_the_equilateral_area(self):
        self.assertLess(_INFO["metric_uniform_dev"], 1e-12)
        self.assertAlmostEqual(_INFO["metric_mean"], np.sqrt(3.0) / 4.0,
                               places=12)

    def test_generic_input_is_carried(self):
        self.assertLess(_INFO["psi_b_leak"], 1e-9)

    def test_register_gram_is_the_identity(self):
        self.assertLess(_INFO["gram_dev"], 1e-12)

    def test_value_equals_amplitude_on_every_realized_gate(self):
        self.assertEqual(len(_REALIZED), len(L2.CANONICAL_SET))
        for row in _REALIZED:
            self.assertLess(row["max_dev"], 1e-9, msg=row["gate"])

    def test_choi_reading_agrees(self):
        for row in _REALIZED:
            self.assertLess(row["choi_dev"], 1e-9, msg=row["gate"])

    def test_floored_gates_have_no_value(self):
        self.assertEqual(len(_FLOORED), 39)
        for row in _FLOORED:
            self.assertIsNone(row["max_dev"], msg=row["gate"])


class BulkIndependenceTest(unittest.TestCase):
    """3 (continued): exact carry-over on the symmetric variant, exact
    Gram-defect law on a generic draw."""

    @classmethod
    def setUpClass(cls):
        cls.inv = L2.h3_invariance(n_variants=1, n_states=2, seed=11)

    def test_symmetric_join_variant_carries_the_value_exactly(self):
        eq = self.inv["equivariant"]
        self.assertLess(eq["gram_dev"], 1e-9)
        self.assertLess(eq["drift"], 1e-9)

    def test_generic_draw_deviates_by_exactly_its_gram_defect(self):
        self.assertGreaterEqual(len(self.inv["anisotropic"]), 1)
        for v in self.inv["anisotropic"]:
            self.assertLess(v["defect_residual"], 1e-9)


class AdditiveGrowthTest(unittest.TestCase):
    """4: the 3d composed stellar move grows interior vertices and preserves
    the register."""

    @classmethod
    def setUpClass(cls):
        cls.reg = L2.RegisterL2(cells=L2._join_cells(8, 8),
                                class_holes=L2._stride_holes(8, 8, 2),
                                grow_vertices=3, grow_seed=7)

    def test_growth_adds_interior_vertices(self):
        self.assertGreaterEqual(self.reg.grown, 1)
        self.assertGreater(int(self.reg.st.getVertexList().size()), 16)

    def test_register_survives_growth(self):
        self.assertEqual(self.reg.dim, 2)
        self.assertEqual(self.reg.rank, 2)
        res, b2, _leak = L2.post_interaction(self.reg, L2.BASE._gates()[0][1])
        self.assertLess(res, L2.REALIZE)
        self.assertEqual(b2, 2)

    def test_metric_stays_uniform_after_the_re_pin(self):
        hodge = L2.cob.HodgeLaplacian(self.reg.st)
        w2 = np.asarray(hodge.weights(2), dtype=float)
        self.assertLess(float(np.max(np.abs(w2 - w2.mean()))), 1e-12)

    def test_canonical_hexjoin_has_no_interior_room(self):
        # The three canonical holes consume all 12 vertices, so the additive
        # budget is unused on the canonical bulk -- documented behavior.
        reg = L2.RegisterL2(grow_vertices=5, grow_seed=3)
        self.assertEqual(reg.grown, 0)
        self.assertEqual(reg.dim, 2)


if __name__ == "__main__":
    unittest.main()
