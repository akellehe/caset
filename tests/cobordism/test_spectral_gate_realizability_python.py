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

"""Spectral gate realizability via STAGED spectral synthesis (S^2 / torus register).

Independent re-derivation of the claims the example
(``examples/cobordism/spectral_gate_realizability.py``) makes, plus a self-verify that
the committed example exits 0. The construction synthesizes the boundary (rather than
pinning it bit-exact) and decides realizability by the CONTINUOUS spectral method -- the
Hodge Laplacian's ker L_1 read by eigendecomposition, not a Levenberg-Marquardt fill:

  1. **The register is an S^2/torus output of surgery.** A triangulated S^2
     (icosahedron) carries the three Z_2 holonomy classes {[a],[b],[a+b]} as three
     vertex-disjoint boundary 1-cycles. The closed sphere has ker L_1 = 0; the boundary-
     fixed surgery ``removeInteriorCell`` opens the three holes, growing b_1 0 -> 2 and
     ker L_1 0 -> 2 -- the 2-dim S_3 standard representation (the carried register V).
  2. **Stages 1-2: each boundary state is synthesized, then unioned.** Each register
     state is carried as a harmonic on its minimal complex (the genuine metric Hodge
     residual -> 0), and dW = geo(psi_A) || geo(psi_B) is held as the boundary.
  3. **The identity is the sanity check, decided spectrally, and it passes.** The
     identity post-interaction state floors on every seed with ker L_1 < 2 and realizes
     only once surgery has grown the full register (b_1 = 2) -- surgery is load-bearing.
  4. **Stage 3: the realizable set is the spectral OUTPUT, and it is 8.** Scored by the
     genuine L_1 residual of U|psi_B>, the realizable set is S_3 + H(x)H + sqrt-SWAP --
     ONE more than the pinned fixed-boundary S_3 + H(x)H = 7 (the relaxation the
     synthesized boundary buys), and not {I} (the topology-free result).
"""

import importlib.util
import os
import subprocess
import sys
import unittest

import numpy as np

import tessera

cob = tessera.cobordism

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


# --------------------------------------------------------------------------- #
class RegisterIsAnS2TorusOutputTest(unittest.TestCase):
    """1: the register is an output of surgery on a triangulated S^2 -- the three
    holonomy holes are genuine interior removable cells, and ker L_1 (the register)
    emerges from the spectrum as surgery opens them."""

    def test_icosahedron_is_a_closed_s2_with_three_interior_holonomy_holes(self):
        st = GATE._surface(GATE._ICO)
        self.assertEqual(GATE._betti(st), [1, 0, 1])         # closed S^2 (b_2 = 1)
        self.assertEqual(GATE._ker_l1_dim(st), 0)            # no register on the sphere
        interior = {tuple(sorted(c))
                    for c in cob.EigenstateSynthesis(st, 1).interiorTopCells()}
        for hole in GATE._CLASS_HOLES:
            self.assertIn(hole, interior)                    # a genuine removable cell
        # the three holes are vertex-disjoint (a clean 3-class register)
        verts = [v for hole in GATE._CLASS_HOLES for v in hole]
        self.assertEqual(len(set(verts)), len(verts))

    def test_surgery_grows_b1_and_ker_l1_zero_to_two(self):
        trace = GATE.register_emergence()
        self.assertEqual([t["b1"] for t in trace], [0, 0, 1, 2])      # b_1 emerges
        self.assertEqual([t["kerL1"] for t in trace], [0, 0, 1, 2])   # ker L_1 emerges
        # b_1 and ker L_1 (= dim H_1) track exactly -- the spectral register IS homology
        for t in trace:
            self.assertEqual(t["b1"], t["kerL1"])


# --------------------------------------------------------------------------- #
class CarriedRegisterTest(unittest.TestCase):
    """The carried register V = ker L_1 of the surgery-grown S^2, read by the genuine
    eigendecomposition: 2-dimensional, with a symmetrizable Sigma = 0 period constraint."""

    def setUp(self):
        self.reg = GATE.Register()

    def test_register_is_two_dimensional_read_from_the_spectrum(self):
        self.assertEqual(self.reg.dim, 2)                    # the S_3 standard rep
        self.assertEqual(GATE._betti1(self.reg.st), 2)
        self.assertEqual(self.reg.P.shape, (2, 3))           # 2 harmonics, 3 circles
        # the harmonics genuinely come from HodgeLaplacian (the continuous spectrum)
        harm = cob.HodgeLaplacian(self.reg.st).harmonics(1)
        self.assertEqual(len(harm), 2)

    def test_period_constraint_is_a_nullvector_of_the_carried_periods(self):
        # the boundary periods of the carried harmonics satisfy n . p = 0 (the signed
        # circle-period sum); n is read off the spectrum, not imposed
        self.assertTrue(np.allclose(self.reg.P @ self.reg.n, 0, atol=1e-9))
        # symmetrized by the induced-orientation signs to the symmetric Sigma = 0
        self.assertTrue(np.allclose(np.abs(self.reg.n), 1.0))

    def test_a_carried_state_has_zero_residual_a_leaking_one_floors(self):
        # a Sigma = 0 (consistent-orientation) period vector is carried -> residual ~ 0;
        # a Sigma != 0 vector leaks out of ker L_1 -> residual floors
        carried = self.reg.sign * np.array([1.0, -1.0, 0.0])       # Sigma = 0
        leaking = self.reg.sign * np.array([1.0, 1.0, 1.0])        # Sigma = 3
        self.assertLess(self.reg.spectral_residual(carried), GATE.REALIZE)
        self.assertGreater(self.reg.spectral_residual(leaking), GATE.CERT_FLOOR)


# --------------------------------------------------------------------------- #
class StagedSynthesisTest(unittest.TestCase):
    """Stages 1-2: each register boundary state is synthesized independently and carried
    as a harmonic; the identity sanity check (stage 3, decided spectrally) passes."""

    def setUp(self):
        self.reg = GATE.Register()

    def test_stage1_synthesizes_each_state_as_a_carried_harmonic(self):
        res_b, nv_b, ne_b = GATE.synthesize_state(self.reg, self.reg.sign * GATE._CP_IN)
        self.assertLess(res_b, GATE.REALIZE)                 # carried on its own complex
        self.assertEqual(nv_b, 12)                           # the icosahedron's vertices
        self.assertEqual(ne_b, 30)                           # |C_1| of the grown bulk

    def test_identity_floors_before_surgery_realizes_after(self):
        anchor = GATE.identity_anchor(self.reg)
        # floors on every under-grown seed (ker L_1 < 2): surgery is load-bearing
        for row in anchor[:-1]:
            self.assertLess(row["kerL1"], 2)
            self.assertFalse(row["realizable"])
            self.assertGreater(row["residual"], GATE.CERT_FLOOR)
        # realizes only once surgery opens the full register (b_1 = 2, ker L_1 = 2)
        self.assertEqual(anchor[-1]["b1"], 2)
        self.assertEqual(anchor[-1]["kerL1"], 2)
        self.assertTrue(anchor[-1]["realizable"])
        self.assertLess(anchor[-1]["residual"], GATE.REALIZE)


# --------------------------------------------------------------------------- #
class RealizableSetIsTheSpectralOutputTest(unittest.TestCase):
    """Stage 3: the realizable set is the spectral OUTPUT -- S_3 + H(x)H + sqrt-SWAP = 8,
    one more than the pinned fixed-boundary S_3 + H(x)H = 7, and not the topology-free
    {I}."""

    def setUp(self):
        self.reg = GATE.Register()
        self.rows = GATE.gate_sweep(self.reg)

    def test_realizable_set_is_s3_plus_hxh_plus_sqrtswap(self):
        realized = [r["gate"] for r in self.rows if r["realizable"]]
        self.assertEqual(realized,
                         ["Identity", "SWAP", "CNOT", "reversed-CNOT",
                          "3-cycle (0231)", "3-cycle (0312)", "H(x)H", "sqrt-SWAP"])
        self.assertEqual(len(realized), 8)                   # one more than 7

    def test_the_six_s3_controls_all_realize(self):
        s3 = [r for r in self.rows if r["family"] == "S3 control"]
        self.assertEqual(len(s3), 6)
        for r in s3:
            self.assertTrue(r["realizable"])
            self.assertLess(r["residual"], GATE.REALIZE)     # machine-zero (the spectrum)

    def test_sqrtswap_is_the_extra_gate_the_synthesized_boundary_buys(self):
        by = {r["gate"]: r for r in self.rows}
        # sqrt-SWAP realizes here though the integer-monodromy fixed-boundary run floored
        # it: a non-integer register automorphism admissible once the boundary is grown
        self.assertTrue(by["sqrt-SWAP"]["realizable"])
        self.assertLess(by["sqrt-SWAP"]["residual"], GATE.REALIZE)
        # H(x)H (the 7th, shared with the fixed-boundary run) also realizes
        self.assertTrue(by["H(x)H"]["realizable"])

    def test_every_floored_gate_is_a_certified_obstruction(self):
        floored = [r for r in self.rows if not r["realizable"]]
        self.assertTrue(floored)
        for r in floored:
            self.assertGreater(r["residual"], GATE.CERT_FLOOR)   # leaks out of ker L_1
            self.assertGreater(r["leak"], 1e-6)                  # Sigma(U|psi_B>) != 0


# --------------------------------------------------------------------------- #
class ExampleSelfVerifiesTest(unittest.TestCase):
    """The committed example runs end-to-end and exits 0 (its own assertions, including
    the spectral realize/floor contrast and the 8-gate realizable set)."""

    def test_example_exits_zero(self):
        self.assertTrue(os.path.exists(_EXAMPLE))
        result = subprocess.run(
            [sys.executable, _EXAMPLE, "--no-write"],
            capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0,
                         msg=f"example exited {result.returncode}\n"
                             f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}")


if __name__ == "__main__":
    unittest.main()
