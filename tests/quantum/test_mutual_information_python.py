"""Python acceptance tests for the C++ MutualInformation utility class.

Skips cleanly when tessera was built without TESSERA_QUANTUM=1.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

try:
    from tessera.quantum import MutualInformation
    HAVE_QUANTUM = True
except ImportError:
    HAVE_QUANTUM = False


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestVonNeumannEntropy(unittest.TestCase):
    """Pure tests on the von Neumann entropy helper."""

    def test_maximally_mixed_qubit(self) -> None:
        rho = 0.5 * np.eye(2, dtype=complex)
        self.assertAlmostEqual(
            MutualInformation.vonNeumannEntropy(rho), math.log(2), places=12)

    def test_pure_qubit(self) -> None:
        rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
        self.assertAlmostEqual(
            MutualInformation.vonNeumannEntropy(rho), 0.0, places=12)

    def test_bell_state_marginal(self) -> None:
        """Bell |Φ+⟩ on two qubits has marginal ρ_A = I/2; S(ρ_A) = ln 2."""
        bell = (np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2)).reshape(4, 1)
        rho_AB = bell @ bell.conj().T
        # Partial trace over B: reshape to (2,2,2,2) then sum over B.
        rho_tensor = rho_AB.reshape(2, 2, 2, 2)
        rho_A = np.trace(rho_tensor, axis1=1, axis2=3)
        self.assertAlmostEqual(
            MutualInformation.vonNeumannEntropy(rho_A), math.log(2), places=12)

    def test_zero_eigenvalues_handled(self) -> None:
        """Eigenvalues below tol shouldn't trip the log(0) branch."""
        rho = np.diag([1.0 - 1e-15, 1e-15, 0.0, 0.0]).astype(complex)
        # Tiny eigenvalues contribute ~0; total entropy is finite.
        S = MutualInformation.vonNeumannEntropy(rho)
        self.assertGreaterEqual(S, 0.0)
        self.assertLess(S, math.log(4))


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestEdgeLength(unittest.TestCase):
    """ℓ = -log(I) with infinity floor."""

    def test_unit_mi_gives_zero_length(self) -> None:
        self.assertAlmostEqual(MutualInformation.edgeLength(1.0), 0.0)

    def test_small_mi_gives_large_length(self) -> None:
        self.assertAlmostEqual(
            MutualInformation.edgeLength(0.1), -math.log(0.1), places=12)

    def test_below_cutoff_returns_infinity(self) -> None:
        self.assertTrue(math.isinf(MutualInformation.edgeLength(1e-15)))
        self.assertTrue(MutualInformation.edgeLength(1e-15) > 0)

    def test_cutoff_threshold(self) -> None:
        eps = 1e-6
        # Just below cutoff → inf
        self.assertTrue(math.isinf(MutualInformation.edgeLength(eps / 2, eps)))
        # Just above cutoff → finite
        self.assertFalse(math.isinf(MutualInformation.edgeLength(eps * 2, eps)))


if __name__ == "__main__":
    unittest.main()
