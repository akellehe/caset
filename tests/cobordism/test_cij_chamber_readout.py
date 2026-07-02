# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""C_ij connected-correlator chamber readout (#564) — instrument regression + gates.

Four layers:

* **Instrument (fast, load-bearing, pure NumPy).** `J² = 9/4 + 2·Σ ⟨S_i·S_j⟩` read through
  the joint two-hole reduced states must return EXACTLY ¾ / 7/4 / 15/4 on the hand-fed clean
  proton / product / Δ states (spin_readout.tex §3), to 1e-12; the pairwise (two-body)
  reformulation equals the full operator on arbitrary states; and the connected `C_ij`
  vanishes identically on every product state (the K-orbit floor) while carrying the whole
  −3/2 entangling shift on the proton eigenstate.

* **Route identities on a fixture.** The vertical route's C_ij is zero to machine precision
  — the STRUCTURAL consequence of a separable reconstruction (a classical cochain's
  bilinear pair read factorizes), asserted as the route's documented scope, not as a field
  measurement. The horizontal route reports no C_ij at all (K-type transport measures
  none), and its transports are genuine SO(3) rotations whose angles reconstruct its J²
  exactly. Neither route's J² reaches the proton ¾ on the fixture — the ¾-exclusion; the
  quantitative gaps are the experiment's report (PR table), not a test assertion.

* **Register validation.** `register_holes` raises on a deficit (a holeless closed S⁴)
  and warns on a surplus (`synthetic_b3_3` stores four holes) instead of silently slicing.

* **GAUGE / RELABEL gates.** EVERY reported numeric channel of both routes — per-pair
  angles and correlators included, not just J² — is invariant under a random per-cell
  SO(4) rotation of the embedding (threaded through `MeshContext(gauge=...)`) and under a
  vertex relabeling + cell-order shuffle with the holes re-derived on the relabeled
  complex. The joint field is pinned in the orientation-canonical ε-signed convention
  (`ChainComplex.endSignCovector`), which is what makes the multi-hole carried
  representative a label-free object.
"""
import importlib.util
import math
import os
import sys
import unittest
import warnings

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


class RegisterSelectionTest(unittest.TestCase):
    """`register_holes` validates the hole count instead of silently slicing."""

    def test_deficit_raises(self):
        # A closed S⁴ has no removed top cells — no register — and must raise, not slice.
        import tessera as T
        st = T.Spacetime(T.Metric(True, T.Signature(4, T.Lorentzian)), T.CDT, 1.0, 1.0,
                         T.PREFERRED, T.SimplexBoundarySphere(4))
        st.build()
        for e in st.getEdgeList().toVector():
            e.setSquaredLength(1.0)
        with self.assertRaises(ValueError):
            cij.register_holes(st)

    def test_surplus_warns_and_is_explicit(self):
        # synthetic_b3_3 stores FOUR emergent holes (b₃ = 4 − 1 = 3): the register read is
        # a sub-register, and the selection must say so.
        with warnings.catch_warnings(record=True) as wlog:
            warnings.simplefilter("always")
            _cells, _edges, st, holes = cij.load_fixture("synthetic_b3_3.json")
        self.assertEqual(len(holes), 3)
        self.assertTrue(any("register selection" in str(w.message) for w in wlog))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            selected, dropped = cij.register_holes(st)
        self.assertEqual(selected, holes)
        self.assertEqual(len(dropped), 1)


class TwoRoutesFixtureTest(unittest.TestCase):
    """The two routes' identities on the synthetic b₃=3 fixture."""

    @classmethod
    def setUpClass(cls):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.cells, cls.edges, cls.st, cls.holes = cij.load_fixture(
                "synthetic_b3_3.json")
            cls.out = cij.two_ways(cls.st, cls.holes)

    def test_vertical_cij_is_the_structural_zero(self):
        # The separable reconstruction's C_ij vanishes at machine precision — the
        # documented STRUCTURAL scope of the route (a classical cochain's bilinear pair
        # read factorizes: rank-1 => rho_ij = rho_i (x) rho_j), not a field measurement.
        vert = self.out["vertical"]
        self.assertIsNotNone(vert)
        for p in _PAIR_KEYS:
            self.assertLessEqual(abs(vert["C_ij"][p]), 1e-12)

    def test_horizontal_reports_no_cij(self):
        # The honest scoping is part of the contract: the horizontal route measures
        # Wilson-line angles only and must NOT report a C_ij (K-type transport measures
        # none); its color block is labeled a reference, carrying only the pinned-input
        # phases and the carry residual.
        horiz = self.out["horizontal"]
        self.assertIsNotNone(horiz)
        self.assertNotIn("C_ij", horiz)
        self.assertEqual(set(horiz["color_reference"]), {"phase_deg", "carry_residual"})

    def test_transport_channel_is_a_genuine_rotation_measurement(self):
        # The Wilson lines are genuine Spin(4) transports: unitary, covering an SO(4)
        # vector rotation (orthogonal, det +1 — break facet_transport/rotation_to_spin
        # and this fails). The reported angles are nontrivial, the route's J² is exactly
        # its own angle reconstruction 9/4 + (1/2)·Σ cos θ, and the reported per-pair
        # axial_mixing equals its definition (max|MᵀM − I| of the diagonal-spin
        # projection — the honest measure of how far each projection is from SO(3)).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ctx = cij.MeshContext(self.st, self.holes)
        horiz = self.out["horizontal"]
        recon = 2.25
        for p in _PAIR_KEYS:
            W = ctx.line(*p)
            self.assertLessEqual(np.max(np.abs(W.conj().T @ W - np.eye(4))), 1e-9)
            R4 = cij.transport_so4(W)
            self.assertLessEqual(np.max(np.abs(R4.T @ R4 - np.eye(4))), 1e-9)
            self.assertLessEqual(abs(np.linalg.det(R4) - 1.0), 1e-9)
            theta = math.radians(horiz["theta_deg"][p])
            self.assertGreater(theta, 0.0)
            self.assertLessEqual(theta, math.pi + 1e-12)
            recon += 0.5 * math.cos(theta)
            self.assertLessEqual(abs(horiz["axial_mixing"][p] - cij.axial_mixing(W)),
                                 1e-12)
        self.assertLessEqual(abs(horiz["j2"] - recon), 1e-9)

    def test_neither_route_reaches_three_quarters(self):
        # The 3/4-exclusion on this fixture: no route lands on the entangled proton value
        # (the quantitative gaps are reported by the example / PR table).
        self.assertFalse(self.out["reaches_proton"]["vertical"])
        self.assertFalse(self.out["reaches_proton"]["horizontal"])

    def test_reads_are_sane(self):
        # Finite, and the vertical J² sits in the three-spin-1/2 product band [3/2, 15/4]
        # (a separable 3-qubit pure state cannot leave it).
        v, h = self.out["vertical"], self.out["horizontal"]
        self.assertTrue(np.isfinite(v["j2"]) and np.isfinite(h["j2"]))
        self.assertGreaterEqual(v["j2"], 1.5 - 1e-9)
        self.assertLessEqual(v["j2"], 3.75 + 1e-9)
        # the carried register certifies the pin (the emergence certificate)
        self.assertLessEqual(h["color_reference"]["carry_residual"], 1e-9)


class GatesTest(unittest.TestCase):
    """GAUGE + RELABEL over EVERY reported channel of both routes, on one fixture."""

    @classmethod
    def setUpClass(cls):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.cells, cls.edges, cls.st, cls.holes = cij.load_fixture(
                "synthetic_b3_3.json")

    def test_gauge_gate(self):
        # Random per-cell SO(4) rotations threaded through MeshContext(gauge=...): every
        # reported channel (per-pair values included) must be invariant.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for read in (cij.vertical_read, cij.horizontal_read):
                self.assertLessEqual(cij.gauge_gate(read, self.st, self.holes), 1e-8,
                                     read.__name__)

    def test_relabel_gate(self):
        # Vertex relabeling + cell-order shuffle, holes re-derived on the relabeled
        # complex: every reported channel must be invariant.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for read in (cij.vertical_read, cij.horizontal_read):
                self.assertLessEqual(
                    cij.relabel_gate(read, self.cells, self.edges, self.holes), 1e-8,
                    read.__name__)

    def test_report_delta_flags_a_perturbed_channel(self):
        # The gate metric itself must see every channel: perturb one per-pair leaf and
        # one nested color leaf and require report_delta to report exactly that size.
        import copy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = cij.horizontal_read(self.st, self.holes)
        mod = copy.deepcopy(base)
        mod["theta_deg"][(0, 2)] += 1e-3
        self.assertGreaterEqual(cij.report_delta(base, mod), 1e-3 - 1e-12)
        mod = copy.deepcopy(base)
        mod["color_reference"]["carry_residual"] += 5e-4
        self.assertGreaterEqual(cij.report_delta(base, mod), 5e-4 - 1e-12)


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
            # and their C_ij vanish (the structural separable-by-construction zero)
            for pair in _PAIR_KEYS:
                self.assertLessEqual(abs(rep["C_ij"][pair]), 1e-12)
        # whether the joint read moved below the product read is REPORTED (PR table),
        # never asserted — the honest outcome is the experiment's result, not a test
        # invariant.


if __name__ == "__main__":
    unittest.main()
