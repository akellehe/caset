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

"""H3 at the VALUE level, on the spectral data alone (the --h3 leg of
``examples/cobordism/spectral_gate_realizability.py``).

The staged spectral synthesis proves a gate's post-interaction state is CARRIED
(residual -> 0); these tests pin the value equation itself, with no DW input:

  1. **The register Gram is the identity.** The period map V -> ker L_1 of the
     surgery-grown icosahedral register is a scaled isometry (G = I after the T1
     anchor fixes the one scale). By Schur's lemma that is exactly the
     S_3-equivariance of the carried register: V is the irreducible S_3 standard
     rep, so any invariant inner product on it is proportional to the flat one.
  2. **Z_spec = <psi_A|U|psi_B> on every realized gate.** The Hodge pairing of the
     carried harmonic representatives equals the flat register amplitude at machine
     precision, over the V-generic input and random carried psi_A — and the
     Choi/operator reading (quantum::ChoiJamiolkowski.transitionAmplitude on the
     C^4 holonomy embedding) agrees independently.
  3. **A floored gate has no spectral value.** Its post-interaction periods leak out
     of V (|Sigma| != 0), so no carried representative exists — the value-level
     obstruction certificate.
  4. **Bulk independence, with its mechanism.** The symmetry-preserving
     re-triangulation (one geodesic subdivision, holes on the central child of each
     original hole face) carries the value exactly; a generic vertex-disjoint hole
     draw deviates from the amplitude by EXACTLY its register Gram defect,
     a^dag (G - I) b. The value-level H3 is the charge-conservation criterion PLUS
     the isometric (equivariant) register chart.
"""

import importlib.util
import os
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.join(_HERE, "..", "..", "examples", "cobordism",
                        "spectral_gate_realizability.py")


def _load_example():
    spec = importlib.util.spec_from_file_location("spectral_gate_realizability",
                                                  _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GATE = _load_example()

# One register + one H3 sweep shared by every test below (the construction is
# deterministic; building it once keeps the suite fast).
_REG = GATE.Register()
_ROWS, _INFO = GATE.h3_value_sweep(_REG, n_states=6, seed=7)
_REALIZED = [r for r in _ROWS if r["realizable"]]
_FLOORED = [r for r in _ROWS if not r["realizable"]]


class RegisterChartIsIsometricTest(unittest.TestCase):
    """1: the carried register's period chart is a scaled isometry (Gram = I)."""

    def test_register_bulk_has_the_unit_cochain_metric(self):
        self.assertLess(_INFO["unit_metric_dev"], 1e-12)

    def test_generic_input_is_carried(self):
        self.assertLess(_INFO["psi_b_leak"], 1e-9)

    def test_register_gram_is_the_identity(self):
        self.assertLess(_INFO["gram_dev"], 1e-12)


class ValueEqualsAmplitudeTest(unittest.TestCase):
    """2: Z_spec = <psi_A|U|psi_B> for every gate the construction realizes."""

    def test_realized_set_is_the_charge_conservation_criterion(self):
        self.assertEqual([r["gate"] for r in _REALIZED],
                         list(GATE.CANONICAL_SET))

    def test_t1_anchor_identity_reproduces_the_inner_product(self):
        identity = _REALIZED[0]
        self.assertEqual(identity["gate"], "Identity")
        self.assertLess(identity["max_dev"], 1e-12)

    def test_value_equals_amplitude_on_every_realized_gate(self):
        for r in _REALIZED:
            self.assertLess(r["max_dev"], 1e-12, msg=r["gate"])

    def test_choi_operator_reading_agrees(self):
        for r in _REALIZED:
            self.assertLess(r["choi_dev"], 1e-12, msg=r["gate"])


class FlooredGatesHaveNoValueTest(unittest.TestCase):
    """3: a floored gate's post-state leaks out of V — no spectral value exists."""

    def test_every_floored_gate_leaks(self):
        self.assertEqual(len(_FLOORED), len(GATE._gates()) - len(GATE.CANONICAL_SET))
        for r in _FLOORED:
            self.assertGreater(r["leak"], 1e-6, msg=r["gate"])
            self.assertIsNone(r["max_dev"], msg=r["gate"])


class BulkIndependenceTest(unittest.TestCase):
    """4: the value carries over exactly to the symmetry-preserving
    re-triangulation; a generic draw deviates by exactly its Gram defect."""

    @classmethod
    def setUpClass(cls):
        cls.inv = GATE.h3_invariance(n_variants=1, n_states=3, seed=7)

    def test_equivariant_retriangulation_carries_the_value_exactly(self):
        eq = self.inv["equivariant"]
        self.assertLess(eq["gram_dev"], 1e-12)
        self.assertLess(eq["drift"], 1e-12)

    def test_generic_draw_deviation_is_exactly_the_gram_defect(self):
        self.assertGreaterEqual(len(self.inv["anisotropic"]), 1)
        for v in self.inv["anisotropic"]:
            # a non-trivial control: the chart is genuinely anisotropic...
            self.assertGreater(v["gram_dev"], 1e-3)
            # ...and the deviation is predicted by a^dag (G - I) b to machine zero.
            self.assertLess(v["defect_residual"], 1e-12)


if __name__ == "__main__":
    unittest.main()
