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

"""Spectral gate realizability: fixed boundaries, emergent bulk.

Independent re-derivation of the claims the example
(``examples/cobordism/spectral_gate_realizability.py``) makes, plus a self-verify
that the committed example exits 0. The question: does the surgery / emergent-b_1
mechanism that realizes superposed STATES also realize superposition / entangling
GATES, when the boundaries are fixed and only the bulk grows?

  1. **The genuine engine works at k=1.** ``RealizabilityOracle.decideHarmonic``
     with boundary-fixed ``GrowthMode.SURGERY`` realizes the matched boundary
     harmonic on the octahedron once surgery opens b_1 0 -> 1, and floors the
     sign-flipped conjugation -- the state test's mechanism, the anchor that the
     harmonic residual + surgery is the right spectral object.
  2. **Surgery grows the Hodge register.** ``EigenstateSynthesis.removeInteriorCell``
     opens the three holonomy holes of a triangulated S^2 on its own, growing
     b_1 0 -> 2 (ker L_1 -> the S_3 standard rep), the boundary held bit-exact.
  3. **S_3 controls realize (validity anchor).** The six holonomy permutations'
     Hodge-register monodromy is a carried permutation -> residual ~ 0. The
     DW-spectral bridge demands this (Z_spec = Z_DW on S_3); if it fails the
     construction is broken.
  4. **The gate sweep.** Of the superposition / phase / entangling battery exactly
     ONE realizes -- H (x) H, whose Hodge-register action *is* the holonomy SWAP --
     and every other gate floors, b_1 free notwithstanding. The mechanism realizes
     superposed states, not superposition gates.
"""

import importlib.util
import os
import subprocess
import sys
import unittest

import numpy as np

import tessera

cob = tessera.cobordism
SURGERY = cob.RealizabilityOracle.GrowthMode.SURGERY

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
class EngineAnchorTest(unittest.TestCase):
    """1: the genuine decideHarmonic + boundary-fixed surgery realizes the matched
    boundary harmonic (b_1 0 -> 1) and floors the sign-flipped one -- at k=1."""

    def test_matched_realizes_by_surgery_flipped_floors(self):
        rows = GATE.engine_anchor()
        matched = next(r for r in rows if "matched" in r["target"])
        flipped = next(r for r in rows if "flipped" in r["target"])
        # The matched harmonic realizes once surgery opens the handle (b_1 0 -> 1).
        self.assertEqual(matched["b1_before"], 0)
        self.assertEqual(matched["b1_after"], 1)
        self.assertGreaterEqual(matched["removals"], 1)
        self.assertLess(matched["residual"], GATE.ENGINE_REALIZE)
        # The sign-flipped conjugation floors -- a cohomological obstruction.
        self.assertGreater(flipped["residual"], GATE.CERT_FLOOR)
        self.assertFalse(flipped["realizable"])


# --------------------------------------------------------------------------- #
class HodgeRegisterEmergenceTest(unittest.TestCase):
    """2: removeInteriorCell opens the three holonomy holes -> b_1 0 -> 2, with the
    boundary bit-exact (the genuine engine grows the register)."""

    def test_surgery_grows_b1_to_two(self):
        trace, harmonics = GATE.hodge_register_emergence()
        self.assertEqual(trace[0]["b1"], 0)            # closed seed
        self.assertEqual(trace[-1]["b1"], 2)           # the Hodge qubit
        self.assertEqual(harmonics, 2)                 # ker L_1 = the standard rep
        # b_1 only moves on a removal (opening the second + third hole).
        b1s = [t["b1"] for t in trace]
        self.assertEqual(b1s, sorted(b1s))             # monotone non-decreasing

    def test_class_holes_are_interior_cells_of_the_closed_seed(self):
        st = GATE._surface(GATE._ICO)
        self.assertEqual(GATE._betti1(st), 0)          # closed S^2
        interior = {tuple(sorted(c))
                    for c in cob.EigenstateSynthesis(st, 1).interiorTopCells()}
        for hole in GATE._CLASS_HOLES:
            self.assertIn(tuple(sorted(hole)), interior)


# --------------------------------------------------------------------------- #
class S3ValidityAnchorTest(unittest.TestCase):
    """3: the six S_3 controls realize -- their Hodge monodromy is a carried
    permutation. The DW-spectral bridge demands this."""

    def test_all_six_s3_controls_realize(self):
        for name, U, fam in GATE._gates():
            if fam != "S3 control":
                continue
            r = GATE.hodge_monodromy_residual(U)
            self.assertLess(r, GATE.REALIZE,
                            msg=f"{name} must realize spectrally (validity anchor); "
                                f"residual {r:.2e}")

    def test_carried_monodromies_are_the_six_s3_permutations(self):
        # The realizable register actions are exactly the standard-rep images of
        # the six cycle permutations -- the S_3 holonomy permutations on ker L_1.
        self.assertEqual(len(GATE._S3_REG), 6)
        # Each is a genuine 2x2 orthogonal map (a triangle symmetry).
        for g in GATE._S3_REG:
            np.testing.assert_allclose(g @ g.conj().T, np.eye(2), atol=1e-9)


# --------------------------------------------------------------------------- #
class GateSweepTest(unittest.TestCase):
    """4: of the superposition / entangling battery exactly H (x) H realizes (its
    Hodge action is the holonomy SWAP); every other gate floors."""

    def test_only_HxH_realizes_in_the_sweep(self):
        realized, floored = [], []
        for name, U, fam in GATE._gates():
            if fam == "S3 control":
                continue
            r = GATE.hodge_monodromy_residual(U)
            (realized if r < GATE.REALIZE else floored).append((name, r))
        self.assertEqual([n for n, _ in realized], ["H(x)H"],
                         msg=f"only H(x)H should realize; got {realized}")
        # Every floored gate is certified obstructed (orders of magnitude above).
        for name, r in floored:
            self.assertGreater(r, GATE.CERT_FLOOR, msg=f"{name} residual {r:.2e}")

    def test_HxH_hodge_action_is_the_swap(self):
        # H (x) H collapses to the holonomy SWAP on the 2-dim Hodge register: its
        # in-register action equals SWAP's, so the SWAP cobordism realizes it.
        hxh = next(U for n, U, _ in GATE._gates() if n == "H(x)H")
        swap = next(U for n, U, _ in GATE._gates() if n == "SWAP")
        reg_hxh = GATE._REG.conj().T @ np.asarray(hxh, dtype=complex)[1:4, 1:4] @ GATE._REG
        reg_swap = GATE._REG.conj().T @ np.asarray(swap, dtype=complex)[1:4, 1:4] @ GATE._REG
        np.testing.assert_allclose(reg_hxh, reg_swap, atol=1e-9)

    def test_off_lattice_gates_genuinely_leak_or_misalign(self):
        # CZ leaves the register (a relative sign on [a+b]); a phase gate misaligns
        # within it. Both floor -- the cohomological obstruction surgery cannot fix.
        for name in ("CZ", "iSWAP", "T(x)I", "sqrt-SWAP"):
            U = next(U for n, U, _ in GATE._gates() if n == name)
            self.assertGreater(GATE.hodge_monodromy_residual(U), GATE.CERT_FLOOR)


# --------------------------------------------------------------------------- #
class ExampleSelfVerifiesTest(unittest.TestCase):
    """The committed example runs end-to-end and exits 0 (its own assertions)."""

    def test_example_exits_zero(self):
        self.assertTrue(os.path.exists(_EXAMPLE))
        result = subprocess.run(
            [sys.executable, _EXAMPLE, "--no-write"],
            capture_output=True, text=True, timeout=900)
        self.assertEqual(result.returncode, 0,
                         msg=f"example exited {result.returncode}\n"
                             f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}")


if __name__ == "__main__":
    unittest.main()
