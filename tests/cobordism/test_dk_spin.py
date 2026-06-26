# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Spin-½ readout from the Dirac–Kähler spinor sector (#477).

A fast structural check: the Kähler–Atiyah fiber of a d=4 mesh is `16 = 4 spinor × 4 taste`,
the gammas close as a Clifford algebra, and every spatial rotation generator
`Σ_ij = ¼[γ_i, γ_j]` has eigenvalues exactly `±½` — the spin-½ signature. No convergence
needed: this is a structural property of the DK construction.
"""
import importlib.util
import os
import sys
import unittest

_EX = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "cobordism")


def _load(name):
    sys.path.insert(0, _EX)
    spec = importlib.util.spec_from_file_location(name, os.path.join(_EX, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class DiracKahlerSpinTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eo = _load("emergent_optimizer")
        cls.sp = _load("dk_spin_readout")

    def test_spin_half_from_clifford_rotation_generators(self):
        host = self.eo.build_closed_s4(n_refine=12, seed=0)
        rep = self.sp.spin_report(host)

        # the fiber is 16 = 4 Dirac spinor × 4 taste
        self.assertEqual(rep["gamma_dim"], 16)
        self.assertEqual(rep["taste_multiplicity"], 4)

        # the gammas close as a Clifford algebra (so the Σ_ij eigenvalues are meaningful)
        self.assertLess(rep["clifford_residual"], 1e-9)

        # spin-½: every spatial rotation generator Σ_ij has eigenvalues exactly ±½ —
        # and nothing else (no 0 → not scalar, no ±1 → not spin-1)
        for _plane, mags in rep["spin_eigenvalue_magnitudes"].items():
            self.assertEqual(len(mags), 1)
            self.assertAlmostEqual(mags[0], 0.5, places=6)
        self.assertTrue(rep["spin_half"])


if __name__ == "__main__":
    unittest.main()
