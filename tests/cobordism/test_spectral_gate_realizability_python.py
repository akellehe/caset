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

"""Spectral gate realizability: topology-less, identity-anchored, emergent bulk.

Independent re-derivation of the claims the example
(``examples/cobordism/spectral_gate_realizability.py``) makes, plus a self-verify that
the committed example exits 0. The construction assumes NO topology (the seed is a
contractible blob grown from a single triangle, betti [1,0,0]) and NO S_3 (the register
``ker L_1`` and the realizable set are read off the genuinely-grown bulk):

  1. **The seed is topology-free.** Both blobs have ``bettiNumbers() == [1, 0, 0]``
     (connected, contractible) and a small 3-edge fixed boundary; their buried cells are
     genuine ``EigenstateSynthesis.interiorTopCells()`` (all-interior removable cells).
  2. **The identity is the only sanity check, and it passes.** On the single-circle
     blob the matched boundary harmonic FLOORS on the b_1 = 0 seed and REALIZES once the
     boundary-fixed surgery search opens b_1 0 -> 1 -- emergent topology carries it.
  3. **b_1 and the register are outputs.** ``removeInteriorCell`` grows b_1 0 -> 3 on the
     four-register blob, and the carried ``ker L_1`` is a 3-dimensional V in C^4 with an
     emergent constraint n.p = 0 read off the bulk's harmonics (derived, not imposed).
  4. **The realizable set is an output, and it is not S_3.** Scored by register
     preservation, the cohomological set is {Identity, SWAP, H(x)H, sqrt-SWAP}; the
     genuine engine realizes only {Identity}. Either way CNOT, reversed-CNOT, and the
     two 3-cycles -- all torus-S_3 members -- FLOOR.
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
class TopologyFreeConstructionTest(unittest.TestCase):
    """1: both blobs are contractible (betti [1,0,0] -- no assumed topology), with a
    small 3-edge fixed boundary and genuine all-interior removable cells."""

    def test_single_blob_is_contractible_with_one_removable_cell(self):
        faces, cell, cyc = GATE._single_blob()
        st = GATE._surface(faces)
        self.assertEqual(GATE._betti(st), [1, 0, 0])     # contractible blob
        self.assertEqual(len(cyc), 3)                    # small 3-edge boundary
        interior = {tuple(sorted(c))
                    for c in cob.EigenstateSynthesis(st, 1).interiorTopCells()}
        self.assertEqual(interior, {tuple(sorted(cell))})  # exactly the buried cell

    def test_four_register_blob_is_contractible_with_three_disjoint_cells(self):
        faces, cells, circles = GATE._four_register_blob()
        st = GATE._surface(faces)
        self.assertEqual(GATE._betti(st), [1, 0, 0])     # contractible blob
        self.assertEqual(len(circles), 4)                # four register circles
        # the three buried register cells are vertex-disjoint and all-interior
        verts = [v for c in cells for v in c]
        self.assertEqual(len(set(verts)), len(verts))    # vertex-disjoint
        interior = {tuple(sorted(c))
                    for c in cob.EigenstateSynthesis(st, 1).interiorTopCells()}
        for c in cells:
            self.assertIn(tuple(sorted(c)), interior)
        # the fixed boundary is the small 3-edge outer triangle
        self.assertEqual(sorted(cob.EigenstateSynthesis(st, 1).boundaryEdges()),
                         [(0, 1), (0, 2), (1, 2)])

    def test_blob_is_grown_from_a_single_triangle_by_pachner_coning(self):
        # The seed is a single triangle; each 1->3 cone adds one vertex and nets +2
        # triangles -- no named topology object, just createVertex / createSimplex.
        faces, _cell, _cyc = GATE._single_blob()
        verts = sorted({v for f in faces for v in f})
        self.assertEqual(verts, [0, 1, 2, 3, 4, 5])      # one triangle + 3 coned apices
        self.assertEqual(len(faces), 7)                  # 1 + 3 cones * (+2 each)
        # the construction is a genuine simplicial complex (every face is a 3-clique)
        for f in faces:
            self.assertEqual(len(set(f)), 3)


# --------------------------------------------------------------------------- #
class IdentityAnchorTest(unittest.TestCase):
    """2: THE sanity check. The identity floors at b_1=0 and realizes at b_1=1 -- the
    emergent hole carries it (the falsifiable core, no topology assumed). The surgery
    move and the fixed-bulk fit are deterministic; the surgery search is robust via
    seed-retry."""

    def test_identity_floors_on_seed_realizes_when_b1_emerges(self):
        seed_row, open_row, search_row = GATE.single_blob_anchor()
        # (1) floors on the b_1=0 seed (no surgery) -- the disk cannot carry it.
        self.assertEqual(seed_row["b1_after"], 0)
        self.assertFalse(seed_row["realizable"])
        self.assertGreater(seed_row["residual"], GATE.CERT_FLOOR)
        # (2) realizes on the opened b_1=1 bulk (the deterministic removeInteriorCell).
        self.assertEqual(open_row["b1_after"], 1)
        self.assertTrue(open_row["realizable"])
        self.assertLess(open_row["residual"], GATE.REALIZE)
        # (3) the surgery search opens b_1 0 -> 1 on its own and realizes.
        self.assertEqual(search_row["b1_before"], 0)
        self.assertEqual(search_row["b1_after"], 1)
        self.assertGreaterEqual(search_row["removals"], 1)
        self.assertLess(search_row["residual"], GATE.REALIZE)


# --------------------------------------------------------------------------- #
class EmergentRegisterTest(unittest.TestCase):
    """3: surgery grows b_1 0 -> 3 on its own, and the carried register V = ker L_1 is a
    3-dimensional output with an emergent (derived) homological constraint."""

    def test_surgery_grows_b1_zero_to_three(self):
        trace, periods, rank, normal = GATE.register_emergence()
        self.assertEqual([t["b1"] for t in trace], [0, 1, 2, 3])  # monotone, emergent
        self.assertEqual(rank, 3)                                 # V is 3-dim in C^4
        self.assertEqual(periods.shape, (3, 4))                   # 3 harmonics, 4 circles
        # the constraint is a genuine nullvector of the carried periods (n.p = 0)
        self.assertTrue(np.allclose(periods @ normal, 0, atol=1e-6))

    def test_register_projector_fixes_v_and_has_rank_three(self):
        _trace, periods, rank, _normal = GATE.register_emergence()
        proj = GATE._register_projector(periods, rank)
        np.testing.assert_allclose(proj @ proj, proj, atol=1e-9)   # idempotent
        self.assertAlmostEqual(np.trace(proj).real, 3.0, places=6)  # rank 3


# --------------------------------------------------------------------------- #
class RealizableSetIsAnOutputTest(unittest.TestCase):
    """4: the realizable set is an OUTPUT, and it is not the torus S_3."""

    def setUp(self):
        _trace, self.periods, self.rank, _normal = GATE.register_emergence()
        self.proj = GATE._register_projector(self.periods, self.rank)
        self.sweep = GATE.cohomological_sweep(self.proj)

    def test_cohomological_realizable_set(self):
        realized = [r["gate"] for r in self.sweep if r["preserves_register"]]
        self.assertEqual(realized, ["Identity", "SWAP", "H(x)H", "sqrt-SWAP"])

    def test_torus_s3_controls_floor_so_the_set_is_not_s3(self):
        leak = {r["gate"]: r["leakage"] for r in self.sweep}
        # the torus-S_3 members beyond the identity all LEAVE the emergent register
        for gate in ("CNOT", "reversed-CNOT", "3-cycle (0231)", "3-cycle (0312)"):
            self.assertGreater(leak[gate], GATE.CERT_FLOOR,
                               msg=f"{gate} would be in S_3 but floors here")
        # the identity always preserves the register (the sanity check, cohomologically)
        self.assertLess(leak["Identity"], GATE.LEAK_TOL)


# --------------------------------------------------------------------------- #
class ExampleSelfVerifiesTest(unittest.TestCase):
    """The committed example runs end-to-end and exits 0 (its own assertions, including
    the genuine-engine realize/floor contrast on the four-register bulk)."""

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
