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

"""The composed growth move-set: additions as well as surgical cuts.

`RealizabilityOracle.GrowthMode.SURGERY_AND_CONE` composes the two existing
moves per growth step — the best IMPROVING interior-top-cell removal (the
SURGERY step), with the additive cone (`growInterior`) as the fallback when no
cut improves. `max_cones` budgets the ADDITIVE commits only (the added
vertices — the resource the examples' `--max-additional-vertices` flag caps);
cuts are bounded by the improving-only rule and the finite interior-cell set.

Covered here:
  1. The mode realizes where cone growth realizes (the 1×3 solid-triangle
     witness needs one added vertex), within the additive budget.
  2. The mode realizes where surgery realizes (the superposed two-boundary
     meridian on the disk seed realizes via a CUT that opens the annulus
     handle), even at additive budget 0 — cuts do not consume the budget.
  3. The additive budget is honored: at budget 0 with no improving cut, no
     vertex is ever added.
  4. The staged-synthesis `Register` accepts additive growth (seeded stellar
     subdivisions) and stays a genuine register: the identity anchor holds and
     the realizable set is still exactly the charge-conservation criterion.
"""

import importlib.util
import os
import unittest

import numpy as np

import tessera

cob = tessera.cobordism
MODE = cob.RealizabilityOracle.GrowthMode

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.join(_HERE, "..", "..", "examples", "cobordism",
                        "spectral_gate_realizability.py")


def _load_example():
    spec = importlib.util.spec_from_file_location("spectral_gate_realizability",
                                                  _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---- the 1×3 cone witness (the realizability_report growth case) ---------- #
def _solid_triangle():
    sig = tessera.Signature(2, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED,
                           tessera.SolidSimplex(2))
    st.build()
    for e in st.getEdgeList().toVector():
        e.setSquaredLength(1.0)
        e.setPhase(0.0)
    return st


def _bend(U, dA, dB):
    cj = tessera.quantum.ChoiJamiolkowski
    flat = [complex(z) for z in np.asarray(U, dtype=complex).reshape(-1)]
    return [complex(z) for z in cj.vectorize(flat, dA, dB)]


_U13 = [[1.0 + 0j, 0.3 + 0.5j, -0.8 + 0.2j]]


class ComposedModeIsExposedTest(unittest.TestCase):
    def test_enum_value_exists(self):
        self.assertTrue(hasattr(MODE, "SURGERY_AND_CONE"))
        self.assertNotEqual(MODE.SURGERY_AND_CONE, MODE.SURGERY)
        self.assertNotEqual(MODE.SURGERY_AND_CONE, MODE.CONE)


class RealizesWhereConeRealizesTest(unittest.TestCase):
    """1 + 3: the 1×3 witness floors at the seed and realizes once the additive
    fallback may add a vertex; at budget 0 (no improving cut on this seed) the
    composed mode adds nothing and floors."""

    def test_one_added_vertex_realizes_the_1x3_witness(self):
        v = cob.RealizabilityOracle(_solid_triangle()).decide(
            _bend(_U13, 1, 3), 1, 3, epsilon=1e-10, restarts=80, max_cones=20,
            seed=0, growth_mode=MODE.SURGERY_AND_CONE)
        self.assertTrue(v.realizable)
        self.assertLess(v.residual, 1e-10)
        self.assertGreaterEqual(v.interior_vertex_count, 1)   # an ADDITION happened
        self.assertLessEqual(v.interior_vertex_count, 20)     # within the budget

    def test_budget_zero_adds_no_vertex(self):
        v = cob.RealizabilityOracle(_solid_triangle()).decide(
            _bend(_U13, 1, 3), 1, 3, epsilon=1e-10, restarts=40, max_cones=0,
            seed=0, growth_mode=MODE.SURGERY_AND_CONE)
        self.assertEqual(v.interior_vertex_count, 0)          # budget honored
        self.assertFalse(v.realizable)                        # floors at the seed


class RealizesWhereSurgeryRealizesTest(unittest.TestCase):
    """2: the superposed two-boundary meridian realizes from the disk seed via a
    CUT (the handle opens, b_1 0 -> 1) under the composed mode — with the
    additive budget at 0, proving cuts do not consume it."""

    @classmethod
    def setUpClass(cls):
        emergent = os.path.join(_HERE, "test_emergent_bulk_python.py")
        spec = importlib.util.spec_from_file_location("emergent_bulk_fixture",
                                                      emergent)
        cls.fx = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.fx)

    def test_meridian_realizes_via_a_cut_at_additive_budget_zero(self):
        st = self.fx._disk()
        matched, _vals = self.fx._meridian_target(flip=False)
        self.assertEqual(self.fx._b1(st), 0)                  # the disk seed
        v = cob.RealizabilityOracle(st).decideHarmonic(
            matched, epsilon=self.fx.DEEP_EPS, restarts=self.fx.RESTARTS,
            max_cones=0, seed=1, growth_mode=MODE.SURGERY_AND_CONE,
            connectivity_candidates=8, harmonic=True)
        self.assertEqual(self.fx._b1(st), 1)                  # the CUT opened b_1
        self.assertGreaterEqual(v.surgery_removals, 1)
        self.assertLess(v.residual, self.fx.REALIZE)          # carried (fixture bar)
        self.assertEqual(v.interior_vertex_count, 0)          # additive budget intact


class RegisterAdditiveGrowthTest(unittest.TestCase):
    """4: the staged-synthesis Register grows additively (seeded stellar
    subdivisions) and stays genuine — identity anchor + the criterion set."""

    @classmethod
    def setUpClass(cls):
        cls.GATE = _load_example()
        cls.reg = cls.GATE.Register(grow_vertices=3, grow_seed=7)

    def test_growth_added_vertices_and_kept_the_register(self):
        base = self.GATE.Register()
        self.assertEqual(self.reg.grown, 3)
        self.assertEqual(int(self.reg.st.getVertexList().size()),
                         int(base.st.getVertexList().size()) + 3)
        self.assertEqual(self.reg.dim, 2)                     # ker L_1 unchanged

    def test_identity_anchor_holds_on_the_grown_register(self):
        res, _b1, leak = self.GATE.post_interaction(self.reg,
                                                    self.GATE._gates()[0][1])
        self.assertLess(res, self.GATE.REALIZE)
        self.assertLess(leak, 1e-9)

    def test_realizable_set_is_still_the_criterion(self):
        for name, U, _fam in self.GATE._gates():
            res, _b1, _leak = self.GATE.post_interaction(self.reg, U)
            self.assertEqual(bool(res < self.GATE.REALIZE),
                             self.GATE.conserves_charge(U), msg=name)

    def test_grown_register_keeps_the_unit_cochain_metric(self):
        """attachInteriorVertex's new edges inherit createSimplexTracked's
        time rule (timelike l^2 < 0 on a time difference) rather than
        Simplex::cone's causal vertex placement; _stellar_grow re-pins the
        bulk uniform so the register's unit metric holds BY CONSTRUCTION.
        Every edge must be spacelike at exactly 1.0 with zero phase, and the
        k=1 Hodge weights exactly unit."""
        w = np.asarray(cob.HodgeLaplacian(self.reg.st).weights(1), dtype=float)
        self.assertEqual(float(np.max(np.abs(w - 1.0))), 0.0)
        for e in self.reg.st.getEdgeList().toVector():
            self.assertEqual(e.getSquaredLength(), 1.0)
            self.assertEqual(e.getPhase(), 0.0)


if __name__ == "__main__":
    unittest.main()
