# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""C_ij connected-correlator chamber readout (#564) — instrument regression + gates.

Three layers:

* **Instrument (fast, load-bearing, pure NumPy).** `J² = 9/4 + 2·Σ ⟨S_i·S_j⟩` read through
  the joint two-hole reduced states must return EXACTLY ¾ / 7/4 / 15/4 on the hand-fed clean
  proton / product / Δ states (spin_readout.tex §3), to 1e-12; the pairwise (two-body)
  reformulation equals the full operator on arbitrary states; and the connected `C_ij`
  vanishes identically on every product state (the K-orbit floor) while carrying the whole
  −3/2 entangling shift on the proton eigenstate.

* **The two routes' identities on a fixture.** The vertical route's C_ij is zero to machine
  precision (a classical cochain's bilinear factorizes), the horizontal transport channel's
  C_ij is zero identically (K-type frame alignment cannot entangle), and neither route's J²
  reaches the proton ¾ — the quantitative gaps are the experiment's report
  (`examples/cobordism/cij_chamber_readout.py`, PR table), not a test assertion.

* **GAUGE / RELABEL gates.** Both routes' J² are invariant under a random per-cell SO(4)
  rotation of the embedding and under a vertex-id relabeling (the joint field is pinned in
  the orientation-canonical ε-signed convention, which is what makes the multi-hole carried
  representative a label-free object).
"""
import importlib.util
import os
import sys
import unittest

import numpy as np
import pytest

_EX = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "cobordism")


def _load_example(name):
    sys.path.insert(0, _EX)
    spec = importlib.util.spec_from_file_location(name, os.path.join(_EX, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cij = _load_example("cij_chamber_readout")

_PAIR_KEYS = [(0, 1), (0, 2), (1, 2)]


class InstrumentTest(unittest.TestCase):
    """The chamber-coordinate J² instrument on hand-fed clean states — exact to 1e-12."""

    def test_clean_states_exact(self):
        # proton eigenstate 2|uud>-|udu>-|duu> -> 3/4; product |uud> -> 7/4; Delta -> 15/4
        want = {"proton 2|uud>-|udu>-|duu>": 0.75,
                "product |uud>": 1.75,
                "Delta |uuu>": 3.75}
        for name, psi in cij.clean_states().items():
            d = cij.j2_decomposition(psi)
            self.assertLessEqual(abs(d["j2"] - want[name]), 1e-12, name)
            # and the pairwise (two-body) read equals the full-operator reference exactly
            self.assertLessEqual(abs(d["j2"] - cij.j2_direct(psi)), 1e-12, name)

    def test_proton_connected_content(self):
        # The entangling shift lives entirely in C_ij: the per-hole Bloch floor is exactly
        # 9/4 and the connected part carries the whole -3/2 down to 3/4.
        d = cij.j2_decomposition(cij.clean_states()["proton 2|uud>-|udu>-|duu>"])
        self.assertLessEqual(abs(d["j2_disconnected"] - 2.25), 1e-12)
        self.assertLessEqual(abs(d["j2_connected"] + 1.5), 1e-12)
        self.assertLessEqual(abs(sum(d["C_ij"].values()) + 0.75), 1e-12)

    def test_pairwise_equals_direct_operator_on_random_states(self):
        # J^2 has no three-body term, so the three rho_ij determine it exactly on ANY state.
        rng = np.random.default_rng(564)
        for _ in range(10):
            psi = rng.normal(size=8) + 1j * rng.normal(size=8)
            self.assertLessEqual(
                abs(cij.j2_decomposition(psi)["j2"] - cij.j2_direct(psi)), 1e-12)

    def test_cij_vanishes_on_every_product_state(self):
        # C_ij = 0 identically on the K-orbit (products, including under arbitrary local
        # unitaries): the connected correlator is exactly what local frames cannot supply.
        rng = np.random.default_rng(7)

        def haar2():
            a = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
            q, r = np.linalg.qr(a)
            return q @ np.diag(np.diag(r) / np.abs(np.diag(r)))

        for _ in range(10):
            qs = [rng.normal(size=2) + 1j * rng.normal(size=2) for _ in range(3)]
            qs = [haar2() @ (q / np.linalg.norm(q)) for q in qs]   # a random K-orbit point
            d = cij.j2_decomposition(cij.kron(*qs))
            for p in _PAIR_KEYS:
                self.assertLessEqual(abs(d["C_ij"][p]), 1e-12)
            self.assertLessEqual(abs(d["j2_connected"]), 1e-12)

    def test_delta_and_product_reach_j2_from_marginals_alone(self):
        # Both non-proton clean states are products: their J² is entirely disconnected.
        for name in ("product |uud>", "Delta |uuu>"):
            d = cij.j2_decomposition(cij.clean_states()[name])
            self.assertLessEqual(abs(d["j2"] - d["j2_disconnected"]), 1e-12)


class TwoWaysFixtureTest(unittest.TestCase):
    """The C_ij-two-ways identities on the synthetic b₃=3 fixture (the escape-hatch test)."""

    @classmethod
    def setUpClass(cls):
        cls.cells, cls.edges, cls.st, cls.holes = cij.load_fixture("synthetic_b3_3.json")
        cls.out = cij.two_ways(cls.st, cls.holes)

    def test_vertical_cij_is_zero_machine_precision(self):
        # The reconstructed joint two-hole reduced states of a classical cochain factorize
        # (rank-1 bilinear => rho_ij = rho_i (x) rho_j), so the vertical C_ij vanishes.
        vert = self.out["vertical"]
        self.assertIsNotNone(vert)
        for p in _PAIR_KEYS:
            self.assertLessEqual(abs(vert["C_ij"][p]), 1e-12)

    def test_horizontal_transport_cij_is_zero(self):
        # K-type frame transport cannot entangle: the transport channel's connected part
        # is identically zero (Claim 1 of cartan_weyl_gluon.tex).
        horiz = self.out["horizontal"]
        self.assertIsNotNone(horiz)
        for p in _PAIR_KEYS:
            self.assertEqual(horiz["C_ij"][p], 0.0)

    def test_neither_route_reaches_three_quarters(self):
        # The decisive escape-hatch identity on this fixture: no route lands on the
        # entangled proton 3/4 (the gaps are reported by the example / PR table).
        self.assertFalse(self.out["reaches_proton"]["vertical"])
        self.assertFalse(self.out["reaches_proton"]["horizontal"])
        self.assertFalse(self.out["reaches_proton"]["horizontal_color"])

    def test_reads_are_sane(self):
        # Finite, and the vertical J² sits in the three-spin-1/2 product band [3/2, 15/4]
        # (a separable 3-qubit pure state cannot leave it).
        v, h = self.out["vertical"], self.out["horizontal"]
        self.assertTrue(np.isfinite(v["j2"]) and np.isfinite(h["j2"]))
        self.assertGreaterEqual(v["j2"], 1.5 - 1e-9)
        self.assertLessEqual(v["j2"], 3.75 + 1e-9)
        # the carried register certifies the pin (the emergence certificate)
        self.assertLessEqual(h["carry_residual"], 1e-9)


class GatesTest(unittest.TestCase):
    """GAUGE + RELABEL invariance of both routes' J² on one fixture — machine precision."""

    @classmethod
    def setUpClass(cls):
        cls.cells, cls.edges, cls.st, cls.holes = cij.load_fixture("synthetic_b3_3.json")

    def test_gauge_gate(self):
        for read in (cij.vertical_read, cij.horizontal_read):
            self.assertLessEqual(cij.gauge_gate(read, self.st, self.holes), 1e-8,
                                 read.__name__)

    def test_relabel_gate(self):
        for read in (cij.vertical_read, cij.horizontal_read):
            self.assertLessEqual(
                cij.relabel_gate(read, self.cells, self.edges, self.holes), 1e-8,
                read.__name__)


class EmergentReadTest(unittest.TestCase):
    """The emergent read on the converged fixture: joint vs independent-product reads."""

    @pytest.mark.slow
    def test_joint_vs_product_read(self):
        _cells, _edges, st, holes = cij.load_fixture("converged_b3_3.json")
        em = cij.emergent_read(st, holes)
        self.assertIsNotNone(em)
        j, p = em["joint"], em["product"]
        for rep in (j, p):
            self.assertTrue(np.isfinite(rep["j2"]))
            # separable reconstructions live in the three-spin-1/2 product band
            self.assertGreaterEqual(rep["j2"], 1.5 - 1e-9)
            self.assertLessEqual(rep["j2"], 3.75 + 1e-9)
            # and their C_ij vanish (the classical bilinear factorizes)
            for pair in _PAIR_KEYS:
                self.assertLessEqual(abs(rep["C_ij"][pair]), 1e-12)
        # whether the joint read moved below the product read is REPORTED (PR table), never
        # asserted — the honest outcome is the experiment's result, not a test invariant.


if __name__ == "__main__":
    unittest.main()
