# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Tests for the pairwise C_ij composite-spin readout (#514, part of #410).

The decisive, fast gate: `⟨J²⟩` is a two-body operator, so the pairwise readout from the three
reduced 2-qubit states reproduces the validated instrument *exactly* — ¾ (proton), 7/4
(product), 15/4 (Δ) — and the connected correlator `C_ij` is exactly the part the per-hole
Bloch read discards (the proton's per-hole read floors at 9/4). Pure NumPy; no `tessera` build
needed. The final cross-check confirms the pure-numpy reference equals the canonical
`dk_joint_spin.j2_three_qubit`, and is skipped if `tessera` is not importable.
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


# The same hand-fed clean states as test_dk_joint_spin.py.
_PROTON = 2 * _kr(_UP, _UP, _DN) - _kr(_UP, _DN, _UP) - _kr(_DN, _UP, _UP)
_PRODUCT = _kr(_UP, _UP, _DN)
_DELTA = _kr(_UP, _UP, _UP)


class PairwiseJ2Gate(unittest.TestCase):
    """The readout returns the textbook values on the hand-fed states."""

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
        # a product is fully disconnected: full J² == the Bloch-read value
        self.assertAlmostEqual(d["j2"], d["j2_disconnected"], places=9)

    def test_proton_bloch_read_floors_at_nine_quarters(self):
        d = pw.j2_decomposition(_PROTON)
        self.assertAlmostEqual(d["j2"], 0.75, places=9)                 # truth keeps C_ij
        self.assertAlmostEqual(d["j2_disconnected"], 2.25, places=9)    # per-hole floor = 9/4
        self.assertAlmostEqual(d["j2_connected"], -1.5, places=9)       # discarded entanglement
        # the floor sits strictly above the proton ¾ — the whole point
        self.assertGreater(d["j2_disconnected"], 0.75 + 0.2)
        # and at least one pair is genuinely connected
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


class CrossCheckInstrument(unittest.TestCase):
    """When tessera is built, the pure-numpy reference == the canonical instrument."""

    def test_reference_matches_dk_joint_spin(self):
        try:
            dj = _load("dk_joint_spin")
        except Exception as exc:                       # tessera not importable in this env
            self.skipTest(f"tessera not importable: {exc}")
        for psi in (_PROTON, _PRODUCT, _DELTA):
            self.assertAlmostEqual(pw._j2_direct(psi), dj.j2_three_qubit(psi), places=9)
            self.assertAlmostEqual(pw.pairwise_j2_from_state(psi), dj.j2_three_qubit(psi),
                                   places=9)


if __name__ == "__main__":
    unittest.main()
