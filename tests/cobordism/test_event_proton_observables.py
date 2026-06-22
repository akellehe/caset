"""Is the Experiment-B event a proton? Falsifiable, final-t, proton-localized (#449).

Asserts the fixed-bipartite-sequence event (#438/#445) produces a proton at its
final time slice, with observables read CORRECTLY for a temporal cobordism:
  * **final-t only** -- the color singlet is read off window R at the TOP slice, not
    the whole worldtube; the charge is the conserved worldtube holonomy (checked
    slice-independent);
  * **proton, not proton+antiproton** -- a single build is ONE baryon
    (`window_count == 4`); the antiproton is a SEPARATE build and the CPT ratio is
    measured ACROSS the two.

Pre-registered thresholds (fixed BEFORE the run; never loosened). The metric
rest mass/radius are NOT asserted: the final-t slice is a fixed Dirichlet boundary
(frozen seed l^2 = 1), so it carries no relaxed metric information -- the test pins
that fact instead (final-t radius == 1.0).
"""

import os
import sys
import unittest

import pytest

import tessera

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "examples", "cobordism"))
import event_proton_observables as E  # noqa: E402

_RELAX = 80
_LAYERS = 2

# --- PRE-REGISTERED thresholds (fixed BEFORE the run; never loosened) ---
_SINGLET_FLOOR = 0.95     # proton/anti-proton color-singlet overlap at final-t
_SIGMA_CEIL = 0.05        # color charge sigma (confinement => 0)
_CPT_TOL = 0.05           # |Q_p/Q_pbar - (-1)| (proton:antiproton ratio = -1)
_CPT_TOTAL = 1e-3         # |Q_p + Q_pbar| (CPT total charge 0)
_FLUX_FLOOR = 1e-6        # full closed-surface flux (topological protection)
_SURF_DEV = 1e-3          # max |ratio+1| over bottom/top/whole-tube surfaces
_FROZEN = 1e-9            # final-t slice radius == 1.0 (Dirichlet-frozen boundary)


@pytest.mark.slow
class EventProtonObservablesTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.o = E.measure_proton(n_layers=_LAYERS, max_iters=_RELAX)

    # --- not proton+antiproton: a single build is ONE baryon ---
    def test_single_baryon_not_a_pair(self):
        # exactly the four windows A,B,C,R: an antiproton would need its own windows,
        # so no single reading sees both the proton and the antiproton.
        self.assertEqual(self.o["window_count"], 4)

    # --- color at FINAL-t (window R at the top slice only) ---
    def test_final_t_color_singlet(self):
        self.assertGreaterEqual(self.o["singlet"], _SINGLET_FLOOR)
        self.assertLessEqual(self.o["sigma"], _SIGMA_CEIL)

    def test_anti_proton_singlet(self):
        self.assertGreaterEqual(self.o["anti_singlet"], _SINGLET_FLOOR)

    # --- electric charge: Gauss-law holonomy, CPT-conjugate to the antiproton ---
    def test_charge_flux_protected(self):
        self.assertLessEqual(self.o["Q_f"], _FLUX_FLOOR)

    def test_cpt_charge_ratio_is_minus_one(self):
        # measured ACROSS the two SEPARATE builds: Q_p / Q_pbar = -1 exactly.
        self.assertLessEqual(abs(self.o["cpt_ratio"] - (-1.0)), _CPT_TOL)
        self.assertLessEqual(self.o["cpt_total"], _CPT_TOTAL)

    def test_cpt_ratio_surface_robust(self):
        # the absolute Q_e magnitude is surface-dependent, but the proton:antiproton
        # ratio is -1 for EVERY Gauss surface (bottom / top / whole worldtube) -- the
        # genuine, normalization-free CPT statement, not a single surface's artifact.
        self.assertLessEqual(self.o["cpt_surface_dev"], _SURF_DEV)

    # --- the final-t slice is a frozen Dirichlet boundary (documented limitation) ---
    def test_final_t_slice_is_frozen_boundary(self):
        # the proton's final-t slice is pinned (the endpoints are the boundary), so
        # its spatial edges keep the uniform seed l^2 = 1 -> radius == 1.0 exactly.
        # This is WHY the metric rest mass/radius are not asserted as the proton's.
        r, n = self.o["final_t_radius"]
        self.assertGreater(n, 0)
        self.assertLessEqual(abs(r - 1.0), _FROZEN)

    # --- the verdict: the top-slice object is a proton ---
    def test_verdict_is_proton(self):
        o = self.o
        self.assertTrue(
            o["window_count"] == 4 and o["singlet"] >= _SINGLET_FLOOR
            and o["sigma"] <= _SIGMA_CEIL
            and abs(o["cpt_ratio"] + 1.0) <= _CPT_TOL
            and o["cpt_total"] <= _CPT_TOTAL)


if __name__ == "__main__":
    unittest.main()
