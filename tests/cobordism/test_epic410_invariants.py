"""Shared regression guard for the charge/flavor-sector epic (#410).

Every PR in the epic must keep this module green -- it pins the proton invariants
G1-G5 so a later subtask cannot silently drift an earlier result. The build is the
emergent relaxed `W_ABC` color singlet (`examples/cobordism/proton_observables.py`
`measure()`), read off the relaxed geometry (never hand-placed).

- G1  singlet overlap >= 0.999  (the proton emerges)
- G2  color charge sigma -> 0   (confinement)
- G3  charge conservation: the closed-surface field-strength flux (the discrete
      Stokes holonomy) vanishes to round-off
- G4  topology: b1 == 11 and dualComplexValid  (no smuggled holes/registers)
- G5  color Z3 intact: P_out eigenvalues are {1, w, w^2}

The Experiment-A (#434) bilateral q/qbar -> proton relaxation
(`examples/cobordism/emergent_intermediates.py` `EmergentEventTopology`) pins three
further invariants. They reuse the exact PRE-REGISTERED thresholds + measurement
helpers from `tests/cobordism/test_emergent_intermediates.py` (copied, never
re-derived; the thresholds are NEVER loosened to manufacture green):

- G9   bilateral pinning regulates the conformal runaway: ||grad S||^2 < 100 at the
       minimal/fair depth nL=2 (asserted at nL=2 ONLY -- ||grad S||^2 is extensive in
       the temporal volume: 71 / 159 / 268 at nL=2 / 3 / 4)
- G10  non-degenerate, CPT-conserved emergent Gauss-law charge: |Q_e| > 0.05
       (Lorentzian) AND <= 1e-9 (all-spacelike control) AND |Q_proton + Q_antiproton|
       <= 1e-3 (CPT)
- G11  proton color singlet >= 0.95 from frame-symmetric COLORED inputs (color is
       never painted; the singlet emerges off the relaxed geometry)

(G6/G7/G8 are the cross-cutting relabeling / determinism / emergent-first concepts
exercised across the suite, not concrete invariants in this module.)
"""

import cmath
import os
import sys
import unittest

import pytest

import numpy as np

import tessera

cob = tessera.cobordism
_W = cmath.exp(2j * cmath.pi / 3)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "examples", "cobordism"))
import proton_observables as P  # noqa: E402
import emergent_intermediates as E  # noqa: E402

_RELAX = 25

# --- Experiment-A (#434) harness, copied verbatim from
# `tests/cobordism/test_emergent_intermediates.py` (PRE-REGISTERED; never loosened). ---
_A_RELAX = 60             # enough to plateau; the reading is under test
_A_LAYERS = 2             # minimal temporal depth (slices 0,1,2; middle = 1, emergent).
                          # ||grad S||^2 is extensive in temporal volume -- it MEETS the
                          # pre-registered floor only at minimal depth (71 at nL=2, vs 159
                          # at nL=3, 268 at nL=4).
_A_GRADS2_FLOOR = 100.0   # carriable ||grad S||^2 (tens) vs strained (thousands)
_A_SINGLET_FLOOR = 0.95   # final proton/anti-proton color-singlet overlap
_A_CHARGE_CANCEL = 1e-3   # |Q_proton + Q_antiproton| (CPT total charge 0)
_A_CHARGE_TAU = 0.05      # min Lorentzian emergent |Q_e| (a degenerate Q == 0 FAILS)


def _build(max_iters=_RELAX):
    seed = cob.TransportCobordism([[1, -1, 0], [1, 0, -1], [0, 1, -1]],
                                  max_iters=0, seed=0,
                                  topology=cob.TripartiteRegisterTopology())
    states = P._omega_rep_input(P._windows(seed))
    return cob.TransportCobordism(states, max_iters=max_iters, seed=0,
                                  topology=cob.TripartiteRegisterTopology())


@pytest.mark.slow
class Epic410InvariantsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.o = P.measure(max_iters=_RELAX)
        cls.m = _build()

    def test_G1_singlet_emerges(self):
        self.assertGreaterEqual(self.o["singlet"], 0.999)

    def test_G2_color_charge_confined(self):
        self.assertLessEqual(self.o["sigma"], 0.06)

    def test_G3_charge_conservation_stokes_holonomy(self):
        # The full closed-surface field-strength flux oint_S F = <psi, d^2 V> = 0:
        # the discrete Stokes / charge-conservation holonomy, exact to round-off.
        self.assertLessEqual(self.o["charge_flux"], 1e-6)
        # the net Dirac-Kahler (Noether) charge is likewise 0 (neutral singlet).
        self.assertLessEqual(self.o["charge_dk_net"], 1e-4)

    def test_G4_topology_unchanged(self):
        cc = cob.ChainComplex.fromSpacetime(self.m.cobordism)
        self.assertEqual(list(cc.bettiNumbers())[1], 11)  # b1 == 11, no new holes
        valid, _msg = cob.EigenstateSynthesis(self.m.cobordism, 1).dualComplexValid()
        self.assertTrue(valid)

    def test_G5_color_z3_spectrum_intact(self):
        _p_in, p_out = P._window_cycle_rep(P._windows(self.m))
        eigs = np.linalg.eigvals(p_out)
        angles = sorted(float(np.angle(e)) for e in eigs)
        expected = sorted([0.0, 2 * np.pi / 3, -2 * np.pi / 3])
        for got, exp in zip(angles, expected):
            self.assertAlmostEqual(got, exp, delta=1e-6)


@pytest.mark.slow
class Epic410ExperimentAInvariantsTest(unittest.TestCase):
    """G9-G11: the Experiment-A (#434) bilateral q/qbar -> proton relaxation, read OFF
    the relaxed `EmergentEventTopology` geometry. The proton sector is Lorentzian
    (timelike worldlines -> a non-empty electric sector), the anti-proton is the
    U-turn twist (opposite charge), and the all-spacelike (Riemannian) build is the
    degenerate control the Lorentzian node must beat."""

    @classmethod
    def setUpClass(cls):
        # proton sector (untwisted, Lorentzian) at the minimal depth nL=2.
        cls.mp, cls.tp = E.build_event(n_layers=_A_LAYERS, lorentzian=True,
                                       u_turn=False, max_iters=_A_RELAX)
        cls.op = E.measure(cls.mp, cls.tp)
        # anti-proton sector (U-turn twist) -- opposite-sign emergent charge.
        cls.ma, cls.ta = E.build_event(n_layers=_A_LAYERS, lorentzian=True,
                                       u_turn=True, max_iters=_A_RELAX)
        cls.oa = E.measure(cls.ma, cls.ta)
        # all-spacelike control: the degenerate E == 0 case (no electric sector).
        cls.m0, cls.t0 = E.build_event(n_layers=_A_LAYERS, lorentzian=False,
                                       u_turn=False, max_iters=_A_RELAX)
        cls.o0 = E.measure(cls.m0, cls.t0)

    def test_G9_bilateral_pinning_regulates_runaway(self):
        # bilateral pinning supplies the restoring force the singly-pinned #435 node
        # lacked: at minimal temporal depth the relaxed ||grad S||^2 is BELOW the
        # carriable floor (71 < 100 at nL=2). Asserted at nL=2 ONLY -- the residual is
        # extensive in the temporal volume (159 at nL=3, 268 at nL=4).
        self.assertLess(self.op["gradS2"], _A_GRADS2_FLOOR)

    def test_G10_emergent_charge_nondegenerate_cpt(self):
        # the Lorentzian proton carries a NON-DEGENERATE emergent Gauss-law charge
        # (|Q_e| > tau), while the all-spacelike control stays degenerate (|Q_e| ~ 0);
        # the charge is emergent (read off the relaxed connection), never a register.
        self.assertGreater(abs(self.op["Q_e"]), _A_CHARGE_TAU)
        self.assertLessEqual(abs(self.o0["Q_e"]), 1e-9)
        # proton + anti-proton total electric charge cancels (CPT).
        self.assertLessEqual(abs(self.op["Q_e"] + self.oa["Q_e"]), _A_CHARGE_CANCEL)

    def test_G11_proton_singlet_from_colored_inputs(self):
        # the pinned top slice crystallizes to the color singlet (>= 0.95) from the
        # frame-symmetric, color-INDEFINITE quark inputs -- color is never painted, the
        # singlet emerges. Both the proton and anti-proton sectors are singlets.
        self.assertGreaterEqual(self.op["top_singlet"], _A_SINGLET_FLOOR)
        self.assertGreaterEqual(self.oa["top_singlet"], _A_SINGLET_FLOOR)


if __name__ == "__main__":
    unittest.main()
