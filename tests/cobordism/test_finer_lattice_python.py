# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Finer geodesic lattice for the W_ABC junction (#404) — tunable granularity.

The junction base is a frequency-N geodesic icosahedron; the granularity is tunable
via `TripartiteRegisterTopology.set_frequency(N)`. These tests pin, through the
production C++ path and the convergence example:

  * N=2 (the default) is backward-compatible with #398 (the four A4-orbit windows are
    exactly the #398 oracle), and every N gives a valid b1=11 manifold;
  * the lattice actually refines (more vertices) with N;
  * refining shrinks the intertwining residual and drives the singlet overlap -> 1;
  * frequency < 2 is rejected.
"""

import importlib.util
import pathlib
import unittest

import tessera

cob = tessera.cobordism

_EXAMPLE = (pathlib.Path(__file__).resolve().parents[2]
            / "examples" / "cobordism" / "finer_lattice_convergence.py")
_spec = importlib.util.spec_from_file_location("finer_lattice_convergence", _EXAMPLE)
_fl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fl)

_ORACLE = [[(2, 13, 14), (8, 31, 32), (10, 22, 36)],
           [(1, 23, 24), (4, 19, 34), (7, 30, 39)],
           [(0, 16, 18), (6, 26, 27), (9, 33, 41)],
           [(3, 15, 29), (5, 20, 21), (11, 37, 38)]]
_ORACLE = [[tuple(sorted(h)) for h in w] for w in _ORACLE]


class FinerLatticeTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.o2 = _fl.measure(2)   # the #398 base
        cls.o3 = _fl.measure(3)   # one refinement

    def test_n2_is_backward_compatible_with_398(self):
        # N=2 (default) reproduces the #398 oracle windows exactly.
        self.assertTrue(all(sorted(self.o2["windows"][i]) == sorted(_ORACLE[i])
                            for i in range(4)))

    def test_every_frequency_is_a_valid_b1_11_manifold(self):
        for o in (self.o2, self.o3):
            self.assertEqual(o["betti"][1], 11)
            self.assertTrue(o["dual_valid"])

    def test_finer_lattice_has_more_vertices(self):
        self.assertGreater(self.o3["verts"], self.o2["verts"])

    def test_refining_shrinks_the_intertwining_residual(self):
        # The residual is a fixed-resolution artifact; it decreases with N.
        self.assertLess(self.o3["intertwine"], self.o2["intertwine"])

    def test_singlet_overlap_is_near_one_and_improves(self):
        self.assertGreater(self.o2["overlap_min"], 0.999)
        self.assertGreaterEqual(self.o3["overlap_min"], self.o2["overlap_min"] - 1e-6)

    def test_frequency_below_two_is_rejected(self):
        with self.assertRaises(Exception):
            cob.TripartiteRegisterTopology().set_frequency(1)


if __name__ == "__main__":
    unittest.main()
