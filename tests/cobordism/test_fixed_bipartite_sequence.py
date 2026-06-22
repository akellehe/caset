"""Experiment B: the fixed bipartite sequence (pinned intermediates) (#438).

The SAME connected, tube-connected (#378, never welded) event cobordism as
Experiment A (`EmergentEventTopology`, #434) -- reused VERBATIM by the subclass
`FixedBipartiteSequenceTopology` -- but with the intermediate result window R
ADDITIONALLY pinned to the known bipartite sequence: the colored 3bar diquark
(the #416-twisted antisymmetric anti-triplet), pinned in the bulk. The endpoints
stay color-emergent. This module pins the ticket's falsifiable criteria with
PRE-REGISTERED thresholds, all read OFF the CONVERGED relaxed geometry through the
shipped C++ (never a preliminary checkpoint; ||grad S||^2 is extensive in temporal
volume, so the verdict is read at convergence at minimal depth nL=2).

Criteria (ticket #438), on top of the epic invariants
(`test_epic410_invariants.py` G1-G5 stays green):
  1. realizability of the known path -- ||grad S||^2 below the per-depth carriable
     floor at convergence (< 100 at nL=2; B = 49, A was 71);
  2. the colored 3bar diquark is HOSTED -- the connected-bulk JOINT carry (the three
     quark inputs -> the colored diquark) is comparable to A's connected-bulk
     0.3-0.5, NOT a free-quark-like floor (the single-window measure is degenerate);
  3. A-vs-B -- A's EMERGENT intermediate (singlet-dominated, weak 3bar) vs B's
     IMPOSED strong 3bar, overlapped in a common frame (endSignCovector signing);
  4. final singlets ~1, charge CPT 0, flux protected, validity, determinism (= A).

Where the geometry does NOT meet a pre-registered floor, the threshold is NEVER
loosened: the criterion is marked xfail with a documented reason (the honest
negative), exactly as #434/#435 handle their negatives.
"""

import os
import sys
import unittest

import numpy as np

import pytest

import tessera

cob = tessera.cobordism

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "examples", "cobordism"))
import emergent_intermediates as A  # noqa: E402
import fixed_bipartite_sequence as B  # noqa: E402

_RELAX = 80                # enough to reach the plateau (values stable 40..300)
_LAYERS = 2                # minimal temporal depth (slices 0,1,2; middle = 1)

# --- PRE-REGISTERED thresholds (fixed BEFORE the run; never loosened) ---
_GRADS2_FLOOR = 100.0      # carriable ||grad S||^2 at nL=2 (per-depth, extensive)
_HOSTED_CEILING = 1.0      # diquark HOSTED if the 3-quarks->diquark carry < 1.0
                           # (same order as A's connected-bulk 0.3-0.5; a floored,
                           # free-quark-like residual would be >> 1 -- cf. the
                           # whole-path 19 / the strained thousands of #382)
_STRONG_3BAR = 0.30        # the IMPOSED diquark is a STRONG 3bar: sigma > 0.30
                           # (vs A's weak emergent ~0.10)
_SINGLET_FLOOR = 0.95      # final proton/anti-proton color-singlet overlap
_CHARGE_CANCEL = 1e-3      # |Q_proton + Q_antiproton| (CPT total charge 0)
_CHARGE_TAU = 0.05         # min Lorentzian emergent |Q_e| (degenerate Q == 0 FAILS)
_FLUX_FLOOR = 1e-6         # full closed-surface flux (topological protection)
_AVB_AGREE = 0.90          # A-vs-B overlap >= 0.90 => the geometry WANTS the path


@pytest.mark.slow
class FixedBipartiteSequenceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mp, cls.tp = B.build_event_B(n_layers=_LAYERS, lorentzian=True,
                                         u_turn=False, max_iters=_RELAX)
        cls.op = B.measure_B(cls.mp, cls.tp)
        cls.ma, cls.ta = B.build_event_B(n_layers=_LAYERS, lorentzian=True,
                                         u_turn=True, max_iters=_RELAX)
        cls.oa = B.measure_B(cls.ma, cls.ta)
        # the all-spacelike (Riemannian) control: the degenerate E == 0 case.
        cls.m0, cls.t0 = B.build_event_B(n_layers=_LAYERS, lorentzian=False,
                                         u_turn=False, max_iters=_RELAX)
        cls.o0 = B.measure_B(cls.m0, cls.t0)

    # --- 7: validity / topology / no welds (structural, must hold) ---
    def test_7_validity_topology_no_welds(self):
        self.assertTrue(self.op["dual_valid"])
        self.assertTrue(self.oa["dual_valid"])
        # b1 == 11: same as #434 -- the subclass adds NO holes/registers, only pins.
        self.assertEqual(self.op["b1"], 11)
        self.assertEqual(self.oa["b1"], 11)

    # --- 7: determinism (relaxation, not a sampler) ---
    def test_7_deterministic(self):
        m2, t2 = B.build_event_B(n_layers=_LAYERS, lorentzian=True, u_turn=False,
                                 max_iters=_RELAX)
        o2 = B.measure_B(m2, t2)
        self.assertAlmostEqual(o2["gradS2"], self.op["gradS2"], places=6)
        self.assertAlmostEqual(o2["hosting_rU"], self.op["hosting_rU"], places=6)

    # --- structure reused VERBATIM: pin OFF reproduces Experiment A exactly ---
    def test_structure_reused_verbatim(self):
        mB0, _t = B.build_event_B(n_layers=_LAYERS, lorentzian=True, u_turn=False,
                                  max_iters=_RELAX, pin_intermediate=False)
        mA, _tA = A.build_event(n_layers=_LAYERS, lorentzian=True, u_turn=False,
                                max_iters=_RELAX)
        # With the intermediate pin OFF, B's subclass is A bit-for-bit.
        self.assertAlmostEqual(mB0.stats.stat_action_residual,
                               mA.stats.stat_action_residual, places=6)
        self.assertAlmostEqual(mB0.stats.state_residual,
                               mA.stats.state_residual, places=6)

    # --- 1: realizability of the known bipartite path (CONVERGED, per-depth floor) ---
    def test_1_known_path_realizable(self):
        # THE realizability positive: at minimal depth the CONVERGED ||grad S||^2 is
        # below the pre-registered per-depth carriable floor (B ~ 49 < 100 at nL=2,
        # even below A's 71) -- the imposed bipartite path keeps the conformal
        # runaway regulated. Read only at convergence (values are stable 40..300).
        self.assertLess(self.op["gradS2"], _GRADS2_FLOOR)

    # --- 2: the colored 3bar diquark is HOSTED (the decisive check) ---
    def test_2_colored_diquark_is_hosted(self):
        # The connected-bulk JOINT carry (three quark inputs -> the colored 3bar
        # diquark, 12 holes -- the SAME count as A's 3-quarks->singlet whole path) is
        # comparable to A's connected-bulk 0.3-0.5, NOT a free-quark-like floor: the
        # bulk HOSTS the strong colored diquark. (The single-window diquark_rU is
        # degenerate ~0 and non-discriminating -- see connected_bulk_rU.)
        self.assertLess(self.op["hosting_rU"], _HOSTED_CEILING)

    def test_2_imposed_diquark_is_strong_3bar(self):
        # The pin DID impose a STRONG 3bar: its colored content sigma is well above
        # A's weak emergent ~0.10 -- the sharpened #438 question's premise.
        self.assertGreater(self.op["diquark_sigma"], _STRONG_3BAR)

    # --- 3: A-vs-B comparison (emergent vs imposed, common frame) ---
    # XFAIL (the documented #438 finding, per "never loosen the pre-registered
    # threshold"): A's EMERGENT intermediate is singlet-dominated (a WEAK 3bar,
    # sigma ~ 0.10); B's IMPOSED intermediate is a STRONG 3bar (sigma ~ 0.58). In a
    # common (endSignCovector) frame their normalized overlap is FAR below the
    # pre-registered agreement threshold 0.90 -- so the geometry does NOT, left to
    # itself, want the textbook strong-3bar bipartite path: A's emergent path
    # DIVERGES from the imposed one. That divergence is the answer to question 3 (a
    # real result), not a code failure; it is quantified by test_3_quantify_divergence.
    @unittest.expectedFailure
    def test_3_geometry_wants_bipartite_path(self):
        overlap, _ai, _bi = B.compare_A_vs_B(n_layers=_LAYERS, max_iters=_RELAX)
        self.assertGreaterEqual(overlap, _AVB_AGREE)

    def test_3_quantify_divergence(self):
        # Quantify how A's emergent path diverges from B's imposed one: A is
        # singlet-dominated and only weakly colored; B is strongly colored. Both read
        # in the SAME induced-orientation frame.
        overlap, ai, bi = B.compare_A_vs_B(n_layers=_LAYERS, max_iters=_RELAX)
        self.assertTrue(0.0 <= overlap <= 1.0)
        self.assertLess(A.color_sigma(ai), 0.30)      # A emergent: weak 3bar
        self.assertGreater(A.color_sigma(bi), _STRONG_3BAR)  # B imposed: strong 3bar

    # --- 4: final singlets (the pinned top slice is the proton/anti-proton) ---
    def test_4_final_singlets(self):
        self.assertGreaterEqual(self.op["top_singlet"], _SINGLET_FLOOR)
        self.assertGreaterEqual(self.oa["top_singlet"], _SINGLET_FLOOR)

    # --- 4: charge / Stokes conservation (same as A) ---
    def test_4_charge_flux_protected(self):
        self.assertLessEqual(abs(self.op["Q_f"]), _FLUX_FLOOR)
        self.assertLessEqual(abs(self.oa["Q_f"]), _FLUX_FLOOR)

    def test_4_total_charge_cpt(self):
        self.assertLessEqual(abs(self.op["Q_e"] + self.oa["Q_e"]), _CHARGE_CANCEL)

    def test_4_charge_emergent_nondegenerate(self):
        # The emergent Gauss-law charge is non-degenerate with the colored diquark
        # pinned (the Lorentzian electric sector survives), while the all-spacelike
        # control stays degenerate (|Q_e| ~ 0). Charge is emergent, never a register.
        self.assertGreater(abs(self.op["Q_e"]), _CHARGE_TAU)
        self.assertLessEqual(abs(self.o0["Q_e"]), 1e-9)

    # --- 4: photons (emergent null edges; reported, not over-claimed) ---
    def test_4_photons_reported(self):
        self.assertGreaterEqual(self.op["n_null"], 0)


if __name__ == "__main__":
    unittest.main()
