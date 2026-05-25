"""Tests for the QuantumSimplex factory and KI bindings."""

from __future__ import annotations

import math
import unittest

import numpy as np
import pytest


try:
    from tessera import (
        Foliation,
        Metric,
        Signature,
        SignatureType,
        Spacetime,
        SpacetimeType,
    )
    from tessera.quantum import (
        QuantumSimplex,
        QuantumSimplexPosition as P,
        koashiImotoDecompose,
        mutualInformation,
        partialTraceA,
        partialTraceB,
    )
    _has_quantum = True
except ImportError:
    _has_quantum = False


pytestmark = pytest.mark.skipif(
    not _has_quantum,
    reason="tessera.quantum.QuantumSimplex not built (needs TESSERA_QUANTUM=1)",
)


def fresh_spacetime():
    metric = Metric(True, Signature(4, SignatureType.Euclidean))
    return Spacetime(metric, SpacetimeType.REGGE, 1.0, 1.0,
                     Foliation.NONE, None)


def bell_phi_plus():
    """ρ_AB = |Φ+⟩⟨Φ+| in (A ⊗ B) basis."""
    rho = np.zeros((4, 4), dtype=complex)
    rho[0, 0] = 0.5
    rho[0, 3] = 0.5
    rho[3, 0] = 0.5
    rho[3, 3] = 0.5
    return rho


def product_state(rhoA, rhoB):
    """ρ_A ⊗ ρ_B in (A ⊗ B) ordering."""
    dA = rhoA.shape[0]
    dB = rhoB.shape[0]
    out = np.zeros((dA * dB, dA * dB), dtype=complex)
    for i in range(dA):
        for j in range(dA):
            for a in range(dB):
                for b in range(dB):
                    out[i * dB + a, j * dB + b] = rhoA[i, j] * rhoB[a, b]
    return out


class TestPartialTraceAndMI(unittest.TestCase):
    """Standalone KI helpers behave on hand-calculable inputs."""

    def test_partial_trace_of_bell(self):
        rho = bell_phi_plus()
        rho_A = partialTraceB(rho, 2, 2)
        rho_B = partialTraceA(rho, 2, 2)
        half_I = 0.5 * np.eye(2)
        np.testing.assert_allclose(rho_A, half_I, atol=1e-12)
        np.testing.assert_allclose(rho_B, half_I, atol=1e-12)

    def test_mutual_information_bell(self):
        rho = bell_phi_plus()
        I = mutualInformation(rho, 2, 2)
        self.assertAlmostEqual(I, 2 * math.log(2.0), places=10)

    def test_mutual_information_product_is_zero(self):
        rhoA = np.diag([0.6, 0.4]).astype(complex)
        rhoB = np.diag([0.3, 0.7]).astype(complex)
        rho = product_state(rhoA, rhoB)
        I = mutualInformation(rho, 2, 2)
        self.assertAlmostEqual(I, 0.0, places=10)


class TestKoashiImotoDecompose(unittest.TestCase):
    def test_bell_decomposition(self):
        rho = bell_phi_plus()
        r = koashiImotoDecompose(rho, 2, 2)
        # Bell pair: 1 block, K_A = K_B = 2, r_A = r_B = 1.
        self.assertEqual(len(r.blocks), 1)
        blk = r.blocks[0]
        self.assertEqual(blk.dimLeftA, 2)
        self.assertEqual(blk.dimLeftB, 2)
        self.assertEqual(blk.dimRightA, 1)
        self.assertEqual(blk.dimRightB, 1)
        self.assertAlmostEqual(blk.weight, 1.0, places=10)

    def test_product_decomposition_is_trivial_block(self):
        rhoA = np.diag([0.6, 0.4]).astype(complex)
        rhoB = np.diag([0.3, 0.7]).astype(complex)
        rho = product_state(rhoA, rhoB)
        r = koashiImotoDecompose(rho, 2, 2)
        # Product: 1 block, K_A = K_B = 1, r_A = r_B = 2.
        self.assertEqual(len(r.blocks), 1)
        blk = r.blocks[0]
        self.assertEqual(blk.dimLeftA, 1)
        self.assertEqual(blk.dimLeftB, 1)
        self.assertEqual(blk.dimRightA, 2)
        self.assertEqual(blk.dimRightB, 2)


class TestQuantumSimplexFromBell(unittest.TestCase):
    def setUp(self):
        self.spacetime = fresh_spacetime()
        self.rho = bell_phi_plus()
        self.i_max = 2.0 * math.log(2.0)
        self.qs = QuantumSimplex.fromKIInteraction(
            self.spacetime, self.rho, 2, 2, self.i_max)

    def test_five_vertices_added(self):
        self.assertEqual(self.spacetime.getVertexCount(), 5)

    def test_marginals_are_half_I(self):
        half_I = 0.5 * np.eye(2)
        np.testing.assert_allclose(self.qs.stateAt(P.A), half_I, atol=1e-12)
        np.testing.assert_allclose(self.qs.stateAt(P.B), half_I, atol=1e-12)

    def test_sigma_is_bell_state(self):
        # For Bell input KI has a 4-dim core equal to |Φ+⟩⟨Φ+|.
        sigma = self.qs.stateAt(P.Sigma)
        self.assertEqual(sigma.shape, (4, 4))
        self.assertAlmostEqual(sigma[0, 0].real, 0.5, places=10)
        self.assertAlmostEqual(sigma[0, 3].real, 0.5, places=10)
        self.assertAlmostEqual(sigma[3, 0].real, 0.5, places=10)
        self.assertAlmostEqual(sigma[3, 3].real, 0.5, places=10)

    def test_tails_are_trivial(self):
        # Bell is maximally entangled — both tails collapse to 1-dim.
        self.assertEqual(self.qs.stateAt(P.APrime).shape, (1, 1))
        self.assertEqual(self.qs.stateAt(P.BPrime).shape, (1, 1))

    def test_mi_ab_matches_bell(self):
        self.assertAlmostEqual(
            self.qs.mutualInfoFor(P.A, P.B),
            2.0 * math.log(2.0),
            places=10)

    def test_d_vr_ab_is_zero(self):
        # Bell at MI = iMax → d_VR = -log(1) = 0.
        self.assertAlmostEqual(
            self.qs.vanRaamsdonkDistanceFor(P.A, P.B),
            0.0,
            places=10)

    def test_other_edges_have_near_zero_mi(self):
        # Cross edges (A, Σ), (A, A'), (A', B'), etc. are product joints
        # in this factory (only the input (A, B) carries inherited MI),
        # so their MI is analytically zero. Numerically the eigensolver
        # in mutualInformation produces noise at machine epsilon
        # (~1e-16) which propagates through -log(I/iMax) to a large
        # finite d_VR rather than exactly +∞. Test accepts either
        # behaviour: MI < 1e-10 AND (d_VR is inf OR d_VR > 30).
        zero_pairs = [
            (P.A, P.Sigma), (P.B, P.Sigma), (P.A, P.APrime),
            (P.B, P.BPrime), (P.A, P.BPrime), (P.B, P.APrime),
            (P.APrime, P.BPrime), (P.APrime, P.Sigma), (P.BPrime, P.Sigma),
        ]
        for p, q in zero_pairs:
            mi = self.qs.mutualInfoFor(p, q)
            dvr = self.qs.vanRaamsdonkDistanceFor(p, q)
            self.assertLess(mi, 1e-10,
                            msg=f"expected ≈ zero MI for ({p}, {q}); got {mi}")
            self.assertTrue(
                math.isinf(dvr) or dvr > 30.0,
                msg=f"expected very large d_VR for ({p}, {q}); got {dvr}")


class TestQuantumSimplexInvalidInputs(unittest.TestCase):
    def test_rejects_zero_imax(self):
        st = fresh_spacetime()
        with self.assertRaises(Exception):
            QuantumSimplex.fromKIInteraction(st, bell_phi_plus(), 2, 2, 0.0)

    def test_rejects_dim_mismatch(self):
        st = fresh_spacetime()
        rho_4x4 = bell_phi_plus()
        with self.assertRaises(Exception):
            # rho is 4x4 but dimA*dimB=6 — should fail.
            QuantumSimplex.fromKIInteraction(st, rho_4x4, 2, 3, 1.0)


if __name__ == "__main__":
    unittest.main()
