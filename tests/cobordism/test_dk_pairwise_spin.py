# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Tests for the pairwise C_ij composite-spin readout (#514, part of #410).

`⟨J²⟩` is a two-body operator, so the pairwise readout from the three reduced 2-qubit states
reproduces the full-operator instrument exactly — ¾ (proton), 7/4 (product), 15/4 (Δ) — and the
connected correlator `C_ij` is exactly the part the per-hole Bloch read discards. The joint
extraction then adds rotationally-invariant correlation against two proven obstructions. Pure
NumPy; no `tessera` build required (the instrument `_j2_direct` is self-contained).
"""
import importlib.util
import os
import sys
import unittest

import numpy as np

_EX = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "cobordism")


def _load(name):
    sys.path.insert(0, _EX)
    spec = importlib.util.spec_from_file_location(name, os.path.join(_EX, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pw = _load("dk_pairwise_spin")
_UP = np.array([1, 0], complex)
_DN = np.array([0, 1], complex)


def _kr(*a):
    out = a[0]
    for x in a[1:]:
        out = np.kron(out, x)
    return out


_PROTON = 2 * _kr(_UP, _UP, _DN) - _kr(_UP, _DN, _UP) - _kr(_DN, _UP, _UP)
_PRODUCT = _kr(_UP, _UP, _DN)
_DELTA = _kr(_UP, _UP, _UP)


class ReferenceInstrument(unittest.TestCase):
    """The self-contained full-operator J² returns the textbook values."""

    def test_direct_j2_values(self):
        self.assertAlmostEqual(pw._j2_direct(_PROTON), 0.75, places=9)
        self.assertAlmostEqual(pw._j2_direct(_PRODUCT), 1.75, places=9)
        self.assertAlmostEqual(pw._j2_direct(_DELTA), 3.75, places=9)


class PairwiseJ2Gate(unittest.TestCase):
    """The pairwise readout returns the textbook values on the hand-fed states."""

    def test_proton_is_three_quarters(self):
        self.assertAlmostEqual(pw.pairwise_j2_from_state(_PROTON), 0.75, places=9)

    def test_product_uud_is_seven_quarters(self):
        self.assertAlmostEqual(pw.pairwise_j2_from_state(_PRODUCT), 1.75, places=9)

    def test_delta_is_fifteen_quarters(self):
        self.assertAlmostEqual(pw.pairwise_j2_from_state(_DELTA), 3.75, places=9)


class ExactnessIdentity(unittest.TestCase):
    """J² is two-body, so the pairwise reformulation equals the full operator on ANY state."""

    def test_pairwise_equals_direct_on_random_states(self):
        rng = np.random.default_rng(514)
        for _ in range(25):
            psi = rng.normal(size=8) + 1j * rng.normal(size=8)
            self.assertAlmostEqual(pw.pairwise_j2_from_state(psi), pw._j2_direct(psi),
                                   places=9)


class ConnectedCorrelator(unittest.TestCase):
    """C_ij is exactly the entanglement content the per-hole Bloch read discards."""

    def test_product_has_zero_connected_correlator(self):
        d = pw.j2_decomposition(_PRODUCT)
        for c in d["C_ij"].values():
            self.assertAlmostEqual(c, 0.0, places=9)
        self.assertAlmostEqual(d["j2_connected"], 0.0, places=9)
        self.assertAlmostEqual(d["j2"], d["j2_disconnected"], places=9)

    def test_proton_bloch_read_floors_at_nine_quarters(self):
        d = pw.j2_decomposition(_PROTON)
        self.assertAlmostEqual(d["j2"], 0.75, places=9)
        self.assertAlmostEqual(d["j2_disconnected"], 2.25, places=9)
        self.assertAlmostEqual(d["j2_connected"], -1.5, places=9)
        self.assertGreater(d["j2_disconnected"], 0.75 + 0.2)
        self.assertTrue(any(abs(c) > 1e-6 for c in d["C_ij"].values()))

    def test_delta_is_pure_disconnected(self):
        d = pw.j2_decomposition(_DELTA)
        for c in d["C_ij"].values():
            self.assertAlmostEqual(c, 0.0, places=9)
        self.assertAlmostEqual(d["j2"], 3.75, places=9)


class ReducedStateSanity(unittest.TestCase):
    """The partial traces are bona fide density matrices."""

    def test_reduced_states_are_unit_trace_hermitian(self):
        rng = np.random.default_rng(7)
        psi = rng.normal(size=8) + 1j * rng.normal(size=8)
        pairs, singles = pw.reduced_states(psi)
        for r in list(pairs.values()) + list(singles.values()):
            self.assertAlmostEqual(float(np.trace(r).real), 1.0, places=9)
            self.assertLess(float(np.linalg.norm(r - r.conj().T)), 1e-9)


class JointExtractionObstructions(unittest.TestCase):
    """The two facts that constrain any joint two-hole read off the field."""

    def test_isotropic_heisenberg_preserves_j2(self):
        rng = np.random.default_rng(2)
        u01 = np.kron(pw.isotropic_heisenberg(0.7), np.eye(2))
        for _ in range(10):
            psi = rng.normal(size=8) + 1j * rng.normal(size=8)
            self.assertAlmostEqual(pw.pairwise_j2_from_state(psi),
                                   pw.pairwise_j2_from_state(u01 @ psi), places=9)

    def test_classical_field_outer_read_factorizes(self):
        rng = np.random.default_rng(3)
        u = rng.normal(size=(5, 2)); v = rng.normal(size=(5, 2)); f = rng.normal(size=5)
        amp = sum(f[a] * f[b] * np.kron(u[a], v[b]) for a in range(5) for b in range(5))
        self.assertEqual(np.linalg.matrix_rank(amp.reshape(2, 2), tol=1e-9), 1)


class WernerJointState(unittest.TestCase):
    """The field-sourced Werner read: a valid, covariant, nonzero-C_ij joint state."""

    def test_correlated_pair_spin_values(self):
        self.assertAlmostEqual(pw.spin_correlator(pw.correlated_pair(1.0)), -0.75, places=9)
        self.assertAlmostEqual(pw.spin_correlator(pw.correlated_pair(0.0)), 0.25, places=9)
        self.assertAlmostEqual(pw.spin_correlator(pw.correlated_pair(0.75)), -0.5, places=9)

    def test_product_limit_recovers_the_floor(self):
        u, d = np.array([1, 0], complex), np.array([0, 1], complex)
        pairs = {(0, 1): pw.werner_pair(u, u, 0.0, 0.5),
                 (0, 2): pw.werner_pair(u, d, 0.0, 0.5),
                 (1, 2): pw.werner_pair(u, d, 0.0, 0.5)}
        dec = pw.decomposition_from_pairs(pairs)
        for c in dec["C_ij"].values():
            self.assertAlmostEqual(c, 0.0, places=9)
        self.assertAlmostEqual(dec["j2"], dec["j2_disconnected"], places=9)

    def test_nonzero_lambda_makes_Cij_nonzero(self):
        u, d = np.array([1, 0], complex), np.array([0, 1], complex)
        dec = pw.decomposition_from_pairs({
            (0, 1): pw.werner_pair(u, u, 0.5, 0.75),
            (0, 2): pw.werner_pair(u, d, 0.5, 0.75),
            (1, 2): pw.werner_pair(u, d, 0.5, 0.75)})
        self.assertTrue(any(abs(c) > 1e-6 for c in dec["C_ij"].values()))

    def test_werner_is_valid_density_matrix(self):
        rng = np.random.default_rng(11)
        for _ in range(8):
            qi = rng.normal(size=2) + 1j * rng.normal(size=2); qi /= np.linalg.norm(qi)
            qj = rng.normal(size=2) + 1j * rng.normal(size=2); qj /= np.linalg.norm(qj)
            rho = pw.werner_pair(qi, qj, 0.6, 0.7)
            self.assertAlmostEqual(float(np.trace(rho).real), 1.0, places=9)
            self.assertLess(float(np.linalg.norm(rho - rho.conj().T)), 1e-9)
            self.assertGreater(float(np.linalg.eigvalsh(rho).min()), -1e-9)

    def test_rotational_covariance_of_Cij(self):
        rng = np.random.default_rng(5)
        qi, qj = np.array([1, 0], complex), np.array([0, 1], complex)
        q, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))

        def cc(r):
            return pw.connected_correlator(r, pw._ptrace_second(r), pw._ptrace_first(r))

        self.assertAlmostEqual(cc(pw.werner_pair(qi, qj, 0.6, 0.7)),
                               cc(pw.werner_pair(q @ qi, q @ qj, 0.6, 0.7)), places=9)


class EmergentColorRead(unittest.TestCase):
    """The surviving-APIs color-phase read (pure numpy): measured periods → nonzero C_ij."""

    def test_ideal_singlet_periods_give_120deg_and_nonzero_Cij(self):
        import cmath
        import math
        w = cmath.exp(2j * math.pi / 3)
        dec = pw.emergent_color_pairwise([1, w, w * w], residual=0.0)
        for v in dec["phases"].values():                       # 120-degree inter-hole phases
            self.assertAlmostEqual(abs(v), 2 * math.pi / 3, places=6)
        self.assertAlmostEqual(dec["j2_disconnected"], 2.25, places=6)   # the 9/4 baseline
        for c in dec["C_ij"].values():                         # purely color-phase-sourced
            self.assertAlmostEqual(c, -0.5, places=6)          # 120 deg -> <S.S>_corr = -1/2

    def test_aligned_periods_are_triplet_like(self):
        dec = pw.emergent_color_pairwise([1, 1, 1], residual=0.0)
        for v in dec["phases"].values():
            self.assertAlmostEqual(v, 0.0, places=6)
        for c in dec["C_ij"].values():
            self.assertAlmostEqual(c, 0.25, places=6)          # 0 phase -> triplet (+1/4)

    def test_high_residual_recovers_product(self):
        dec = pw.emergent_color_pairwise([1, 1, 1], residual=50.0)   # lam -> 0
        for c in dec["C_ij"].values():
            self.assertAlmostEqual(c, 0.0, places=6)
        self.assertAlmostEqual(dec["j2"], dec["j2_disconnected"], places=6)


if __name__ == "__main__":
    unittest.main()
