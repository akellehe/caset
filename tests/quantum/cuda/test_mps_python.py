"""Cross-validation tests for tessera.quantum.cuda.MPS.

Each test runs on the CPU torch backend and (if available) on
``cuda:0``. The MPS state lives entirely in C++; only torch::Tensor
values cross the binding boundary.
"""
from __future__ import annotations

import math
import unittest

import torch

try:
    from tessera.quantum.cuda import MPS, MutualInformation
    HAVE_BACKEND = True
except ImportError:
    HAVE_BACKEND = False


def devices():
    if not HAVE_BACKEND:
        return []
    ds = [torch.device("cpu")]
    if torch.cuda.is_available():
        ds.append(torch.device("cuda:0"))
    return ds


@unittest.skipUnless(HAVE_BACKEND, "tessera.quantum.cuda not built")
class TestProductState(unittest.TestCase):
    """Product states have zero single-site entropy and zero MI."""

    def test_product_state_norm_and_rdm(self):
        for dev in devices():
            with self.subTest(device=str(dev)):
                psi = MPS.computational_basis([0, 1, 0, 1, 0], device=dev)
                self.assertAlmostEqual(psi.norm_squared(), 1.0, places=10)
                for i in range(5):
                    rho = psi.one_site_reduced_density(i)
                    self.assertEqual(tuple(rho.shape), (2, 2))
                    S = MutualInformation.von_neumann_entropy(rho)
                    self.assertAlmostEqual(S, 0.0, places=10)

    def test_product_state_pair_mi_zero(self):
        for dev in devices():
            with self.subTest(device=str(dev)):
                psi = MPS.computational_basis([0, 1, 0, 1], device=dev)
                for i in range(4):
                    for j in range(i + 1, 4):
                        I_ij = MutualInformation.site_site(psi, i, j)
                        self.assertAlmostEqual(
                            I_ij, 0.0, places=10,
                            msg=f"product MI(i={i},j={j}) = {I_ij}")


@unittest.skipUnless(HAVE_BACKEND, "tessera.quantum.cuda not built")
class TestBellPair(unittest.TestCase):
    """|Phi+> on 2 sites: rho_i = I/2, entropy = ln 2, MI = 2 ln 2."""

    def test_bell_marginal_is_maximally_mixed(self):
        for dev in devices():
            with self.subTest(device=str(dev)):
                psi = MPS.bell_chain(1, device=dev)
                self.assertAlmostEqual(psi.norm_squared(), 1.0, places=10)
                expected = 0.5 * torch.eye(2, device=dev,
                                            dtype=torch.complex128)
                for i in (0, 1):
                    rho = psi.one_site_reduced_density(i)
                    self.assertTrue(
                        torch.allclose(rho, expected, atol=1e-10),
                        msg=f"rho_{i} != I/2 (got {rho.cpu()})")

    def test_bell_pair_mi(self):
        for dev in devices():
            with self.subTest(device=str(dev)):
                psi = MPS.bell_chain(1, device=dev)
                I_01 = MutualInformation.site_site(psi, 0, 1)
                self.assertAlmostEqual(I_01, 2.0 * math.log(2.0),
                                         places=10)

    def test_bell_chain_two_pairs(self):
        """|Phi+>(2): pairs (0,1) and (2,3) carry MI 2 ln 2; all others
        have zero MI."""
        for dev in devices():
            with self.subTest(device=str(dev)):
                psi = MPS.bell_chain(2, device=dev)
                self.assertAlmostEqual(psi.norm_squared(), 1.0, places=10)
                two_ln2 = 2.0 * math.log(2.0)
                expected = {(0, 1): two_ln2, (2, 3): two_ln2}
                for i in range(4):
                    for j in range(i + 1, 4):
                        I_ij = MutualInformation.site_site(psi, i, j)
                        exp = expected.get((i, j), 0.0)
                        self.assertAlmostEqual(
                            I_ij, exp, places=10,
                            msg=f"MI(i={i},j={j}) = {I_ij}, want {exp}")


@unittest.skipUnless(HAVE_BACKEND, "tessera.quantum.cuda not built")
class TestGHZState(unittest.TestCase):

    def test_ghz_marginals(self):
        for dev in devices():
            with self.subTest(device=str(dev)):
                psi = MPS.ghz(5, device=dev)
                self.assertAlmostEqual(psi.norm_squared(), 1.0, places=10)
                ln2 = math.log(2.0)
                for i in range(5):
                    rho = psi.one_site_reduced_density(i)
                    S = MutualInformation.von_neumann_entropy(rho)
                    self.assertAlmostEqual(
                        S, ln2, places=10,
                        msg=f"site {i}: entropy = {S}, want ln 2")

    def test_ghz_pair_mi(self):
        for dev in devices():
            with self.subTest(device=str(dev)):
                psi = MPS.ghz(5, device=dev)
                two_ln2 = 2.0 * math.log(2.0)
                for i in range(5):
                    for j in range(i + 1, 5):
                        I_ij = MutualInformation.site_site(psi, i, j)
                        self.assertAlmostEqual(
                            I_ij, two_ln2, places=10,
                            msg=f"MI(i={i},j={j}) = {I_ij}, want 2 ln 2")


@unittest.skipUnless(HAVE_BACKEND, "tessera.quantum.cuda not built")
class TestCanonicalForm(unittest.TestCase):

    def test_random_mps_canonical_form(self):
        for dev in devices():
            with self.subTest(device=str(dev)):
                N = 6
                psi = MPS.random(N, 2, 8, device=dev, seed=42)
                for oc in (0, 2, 5):
                    psi.canonicalize(oc)
                    for k in range(0, oc):
                        self.assertTrue(
                            psi.is_left_canonical(k),
                            msg=f"site {k} not left-canonical "
                                 f"(oc={oc})")
                    for k in range(oc + 1, N):
                        self.assertTrue(
                            psi.is_right_canonical(k),
                            msg=f"site {k} not right-canonical "
                                 f"(oc={oc})")

    def test_norm_preserved_under_canonicalization(self):
        for dev in devices():
            with self.subTest(device=str(dev)):
                psi = MPS.random(6, 2, 6, device=dev, seed=7)
                n0 = psi.norm_squared()
                psi.canonicalize(3)
                n1 = psi.norm_squared()
                psi.canonicalize(0)
                n2 = psi.norm_squared()
                self.assertAlmostEqual(n0, 1.0, places=10)
                self.assertAlmostEqual(n1, 1.0, places=10)
                self.assertAlmostEqual(n2, 1.0, places=10)


def _brute_force_rdm_from_dense(state: torch.Tensor, N: int, d: int,
                                  i: int, j: int) -> torch.Tensor:
    """ρ_{ij} via a single einsum on the dense state vector.

    Interior site indices share a letter between ket and bra so they
    auto-trace; sites i and j keep distinct letters so they remain
    open. Output row index = d * ket_i + ket_j, col index = d * bra_i +
    bra_j — matches the MPS basis convention.
    """
    assert i < j
    psi = state.reshape([d] * N)
    ket_letters, bra_letters = [], []
    next_char = ord('a')
    for k in range(N):
        if k == i or k == j:
            ket_letters.append(chr(next_char)); next_char += 1
            bra_letters.append(chr(next_char)); next_char += 1
        else:
            shared = chr(next_char); next_char += 1
            ket_letters.append(shared)
            bra_letters.append(shared)
    ket_str = "".join(ket_letters)
    bra_str = "".join(bra_letters)
    out_str = (ket_letters[i] + ket_letters[j]
                + bra_letters[i] + bra_letters[j])
    rho4 = torch.einsum(f"{ket_str},{bra_str}->{out_str}",
                          psi, psi.conj())
    return rho4.reshape(d * d, d * d)


@unittest.skipUnless(HAVE_BACKEND, "tessera.quantum.cuda not built")
class TestTransferSweepCorrectness(unittest.TestCase):
    """The transfer-matrix two-site RDM must agree with a brute-force
    dense contraction. This is the regression check that the algorithm
    matches the C++ ITensor fix."""

    def test_random_mps_matches_dense(self):
        for dev in devices():
            with self.subTest(device=str(dev)):
                N = 6
                d = 2
                psi = MPS.random(N, d, max_chi=8, device=dev, seed=123)
                state = psi.dense_state_vector()
                for i in range(N):
                    for j in range(i + 1, N):
                        rho_sweep = psi.two_site_reduced_density(i, j)
                        rho_dense = _brute_force_rdm_from_dense(
                            state, N, d, i, j)
                        diff = (rho_sweep - rho_dense).norm().item()
                        self.assertLess(
                            diff, 1e-9,
                            msg=f"mismatch at (i={i}, j={j}): "
                                 f"diff norm {diff:.2e}")

    def test_random_mps_mi_non_negative(self):
        for dev in devices():
            with self.subTest(device=str(dev)):
                psi = MPS.random(5, 2, max_chi=6, device=dev, seed=11)
                mi = MutualInformation.all_pairs(psi)
                self.assertGreaterEqual(mi.min().item(), -1e-9)


@unittest.skipUnless(HAVE_BACKEND, "tessera.quantum.cuda not built")
class TestVonNeumannEntropy(unittest.TestCase):

    def test_maximally_mixed_qubit(self):
        for dev in devices():
            with self.subTest(device=str(dev)):
                rho = 0.5 * torch.eye(2, device=dev,
                                       dtype=torch.complex128)
                S = MutualInformation.von_neumann_entropy(rho)
                self.assertAlmostEqual(S, math.log(2.0), places=12)

    def test_pure_state_entropy_zero(self):
        for dev in devices():
            with self.subTest(device=str(dev)):
                rho = torch.zeros((2, 2), device=dev,
                                    dtype=torch.complex128)
                rho[0, 0] = 1.0
                S = MutualInformation.von_neumann_entropy(rho)
                self.assertAlmostEqual(S, 0.0, places=12)

    def test_edge_length(self):
        self.assertAlmostEqual(MutualInformation.edge_length(1.0), 0.0)
        self.assertAlmostEqual(MutualInformation.edge_length(0.1),
                                -math.log(0.1), places=10)
        self.assertTrue(math.isinf(MutualInformation.edge_length(1e-15)))


if __name__ == "__main__":
    unittest.main()
