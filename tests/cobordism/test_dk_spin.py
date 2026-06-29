# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Structural spin-½ of the Dirac-Kahler fiber (#415/#477) — the revived C++ DiracKahler.

The Kahler-Atiyah fiber of a d=4 mesh is 16 = 4 spinor x 4 taste; the gammas close as a
Clifford algebra ({gamma^a, gamma^b} = 2 eta^ab); every spatial rotation generator
Sigma_ij = 1/4 [gamma_i, gamma_j] has eigenvalues exactly +/- 1/2 (the spin-1/2 signature —
not 0/scalar, not +/-1/vector); and (d + delta)^2 reproduces the Hodge Laplacian.

Structural: no convergence needed. This tests the C++ `DiracKahler` class directly; the retired
Python `spin_report` / `emergent_optimizer` examples are deliberately NOT brought back.
"""
import unittest

import numpy as np

import tessera as T

cob = T.cobordism


def _host():
    """Any valid 4D complex — the Clifford framework is the fixed 4D structure, so two
    pentatopes sharing a facet suffice."""
    return T.Spacetime.fromCells(4, [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], 1.0, 0.0)


class DiracKahlerSpinTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dk = cob.DiracKahler(_host())

    def test_fiber_is_16_with_taste_4(self):
        self.assertEqual(self.dk.frameworkDimension(), 4)
        self.assertEqual(self.dk.gammaDimension(), 16)     # 2^4
        self.assertEqual(self.dk.multiplicity(), 4)        # 16 = 4 spinor x 4 taste

    def test_gammas_close_as_clifford(self):
        self.assertLess(self.dk.cliffordResidual(), 1e-9)

    def test_spatial_rotation_generators_are_spin_half(self):
        n = self.dk.gammaDimension()
        g = [np.array(m, dtype=complex).reshape(n, n) for m in self.dk.gammas()]
        for i in range(1, 4):                               # spatial planes (1,2,3)
            for j in range(i + 1, 4):
                sigma = 0.25 * (g[i] @ g[j] - g[j] @ g[i])  # Sigma_ij = 1/4 [g_i, g_j]
                mags = sorted({round(abs(e), 6) for e in np.linalg.eigvals(sigma)})
                self.assertEqual(mags, [0.5])               # only +/- 1/2

    def test_square_reproduces_hodge_laplacian(self):
        self.assertLess(self.dk.laplacianResidual(), 1e-6)  # (d + delta)^2 = L


if __name__ == "__main__":
    unittest.main()
