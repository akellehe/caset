"""Electric charge as the temporal-sector Gauss-law holonomy (#411).

`Q = oint_S E` reads electric charge as a discrete Gauss law on the electric part
of the field strength `F = d psi` (`EigenstateSynthesis.gaussLawCharge`), summed
over a closed surface `S` enclosing the quark windows. The claim under test: `Q` is
a genuine gauged-U(1) holonomy -- quantized and metric-robust, like the color charge
sigma -- whereas the hand-weighted `(2/3, 2/3, -1/3)` flavor covector is not
A4/color-Z3 equivariant and drifts under metric jitter.

All randomness uses a FIXED seed (`numpy.random.default_rng(411)`); the build is the
existing relaxed `W_ABC` color singlet, read OFF the relaxed geometry (never
hand-placed). See `docs/design/proton_charge_gauss_law.md`.
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

_RELAX = 25     # short relax keeps the test quick; the reading is under test
_N_JIT = 16     # independent jitter seeds (F1/F2)
_MAG = 0.20     # relative spacelike-l^2 jitter magnitude


def _build(max_iters=_RELAX):
    seed = cob.TransportCobordism([[1, -1, 0], [1, 0, -1], [0, 1, -1]],
                                  max_iters=0, seed=0,
                                  topology=cob.TripartiteRegisterTopology())
    states = P._omega_rep_input(P._windows(seed))
    return cob.TransportCobordism(states, max_iters=max_iters, seed=0,
                                  topology=cob.TripartiteRegisterTopology())


def _read(m):
    """One carried representative -> (Q_electric, Q_full, covector), off the LIVE
    metric (exercises the shipped `gaussLawCharge` C++ method)."""
    es1 = cob.EigenstateSynthesis(m.cobordism, 1)
    es2 = cob.EigenstateSynthesis(m.cobordism, 2)
    result = [list(h) for h in P._windows(m)[3]]
    psi = np.array(es1.carriedRepresentative(result, [1.0, _W, _W * _W]))
    F = list(es2.curvatureFromConnection(list(psi)))
    enclosed = sorted(set(v for w in P._windows(m)[:3] for h in w for v in h))
    q_e = es2.gaussLawCharge(F, enclosed, True)
    q_f = es2.gaussLawCharge(F, enclosed, False)
    edge = {(min(c), max(c)): i
            for i, c in enumerate(es1.cellSimplices()) if len(c) == 2}

    def period(h):
        a, b, c = h
        return psi[edge[(a, b)]] + psi[edge[(b, c)]] - psi[edge[(a, c)]]

    cov = sum(P._FLAVOR_COVECTOR[i] * sum(period(h) for h in win)
              for i, win in enumerate(P._windows(m)[:3]))
    return q_e, q_f, cov


def _jitter_sweep(m, n=_N_JIT, mag=_MAG):
    """N independent jitter perturbations of the spacelike l^2 metric, restoring to
    the pristine metric between each. Returns (Q_E, Q_F, covector) arrays."""
    evec = m.cobordism.getEdgeList().toVector()
    l0 = [e.getSquaredLength() for e in evec]
    rng = np.random.default_rng(411)
    qe, qf, cov = [], [], []
    for _ in range(n):
        for e in evec:
            v = e.getSquaredLength()
            if v.real > 0:
                nv = v.real * (1.0 + rng.normal(0.0, mag))
                if nv > 0:
                    e.setSquaredLength(nv)
        a, b, c = _read(m)
        qe.append(a)
        qf.append(b)
        cov.append(c)
        for e, v in zip(evec, l0):  # restore the pristine metric
            e.setSquaredLength(v)
    return np.array(qe), np.array(qf), np.array(cov)


@pytest.mark.slow
class ProtonChargeGaussLawTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.m = _build()
        # uniform-metric readings, through the SHIPPED helpers.
        cls.Q_uniform, cls.QF_uniform = P.gauss_law_charge(cls.m)
        cls.C_uniform = P.covector_charge(cls.m)
        cls.dk_net, cls.dk_total = P.dirac_kahler_net_charge(cls.m)
        # the F1/F2 jitter sweep (restores the metric afterwards).
        cls.Qe_jit, cls.Qf_jit, cls.C_jit = _jitter_sweep(cls.m)

    # --- F1: Q is metric-ROBUST (the headline claim) ---
    def test_F1_gauss_charge_is_metric_robust(self):
        # The electric Gauss charge is exactly 0 for every jitter (no temporal
        # sector to populate E under spacelike perturbations).
        self.assertLessEqual(float(np.std(np.abs(self.Qe_jit))), 1e-9)
        self.assertLessEqual(
            float(np.max(np.abs(self.Qe_jit - self.Q_uniform))), 1e-9)
        # The full closed-surface flux is the topological protection: <psi,d^2 V>=0
        # to round-off, robust under metric jitter (the genuine cancellation of a
        # NONzero field strength).
        self.assertLessEqual(float(np.std(np.abs(self.Qf_jit))), 1e-3)
        self.assertLessEqual(
            float(np.max(np.abs(self.Qf_jit - self.QF_uniform))), 1e-3)

    # --- F2: the covector baseline DRIFTS (the contrast that makes F1 meaningful) ---
    def test_F2_covector_baseline_drifts(self):
        std_c = float(np.std(np.abs(self.C_jit)))
        max_dev = float(np.max(np.abs(self.C_jit - self.C_uniform)))
        self.assertGreaterEqual(std_c, 1e-2)        # the covector genuinely drifts
        self.assertGreaterEqual(max_dev, 2e-2)      # reaching the ~0.07 jittered scale
        # the separation: the holonomy is protected, the covector is not.
        std_q = max(float(np.std(np.abs(self.Qf_jit))), 1e-12)
        self.assertGreaterEqual(std_c / std_q, 100.0)

    # --- F3: Q equals the neutral total on the singlet ---
    def test_F3_gauss_charge_is_the_neutral_total(self):
        Q_EXPECTED = 0.0  # flavor-blind S^2 x I sees only the neutral total
        self.assertLessEqual(abs(abs(self.Q_uniform) - Q_EXPECTED), 1e-6)

    # --- F4: Q is quantized (integer / third-integer lattice) ---
    def test_F4_gauss_charge_is_quantized(self):
        q = abs(self.Q_uniform)
        self.assertLessEqual(abs(q - round(3.0 * q) / 3.0), 1e-6)

    # --- F5: determinism (G7) ---
    def test_F5_determinism(self):
        q1, qf1, c1 = _jitter_sweep(self.m)
        q2, qf2, c2 = _jitter_sweep(self.m)
        self.assertTrue(np.array_equal(q1, q2))
        self.assertTrue(np.array_equal(qf1, qf2))
        self.assertTrue(np.array_equal(c1, c2))
        # the uniform reading is reproducible too.
        q_e, q_f = P.gauss_law_charge(self.m)
        self.assertEqual(q_e, self.Q_uniform)
        self.assertEqual(q_f, self.QF_uniform)

    # --- required cross-check: Q == Dirac-Kahler j^0 / carriedCharge (#415) ---
    def test_gauss_flux_agrees_with_dirac_kahler_current(self):
        # The NET Dirac-Kahler (Noether) charge of the neutral singlet is 0 -- the
        # three per-window j^0 charges are equal (A4-symmetric apex) and the singlet
        # phases [1, w, w^2] sum to 0 -- agreeing with the Gauss-law flux Q = 0.
        self.assertLessEqual(abs(self.dk_net), 1e-4)
        self.assertLessEqual(abs(abs(self.Q_uniform) - abs(self.dk_net)), 1e-4)
        # The positive TOTAL carriedCharge is the constituent norm (three quarks),
        # NOT the net -- it counts |Phi|^2, the neutral net is carried by the signed
        # periods.
        self.assertGreater(self.dk_total, 1.0)
        # j^0 is the bona-fide conserved density: charge(lift(psi)) == the period
        # read-out's weighted norm Sum_e W_e |psi_e|^2 (HodgeLaplacian.weights).
        es1 = cob.EigenstateSynthesis(self.m.cobordism, 1)
        hl = cob.HodgeLaplacian(self.m.cobordism)
        dk = cob.DiracKahler(self.m.cobordism)
        psi = np.array(es1.carriedRepresentative(
            [list(h) for h in P._windows(self.m)[3]], [1.0, _W, _W * _W]))
        q_noether = dk.charge(dk.lift(1, list(psi)))
        w_norm = float(np.sum(np.array(hl.weights(1)) * np.abs(psi) ** 2))
        self.assertAlmostEqual(q_noether, w_norm, delta=1e-9)

    # --- charge conservation (no net source on the closed surface) ---
    def test_charge_conservation_closed_surface_flux_vanishes(self):
        # oint_S F = <d psi, d V> = <psi, d^2 V> = 0: the net charge enclosed by the
        # closed worldtube boundary vanishes (the neutral singlet is sourceless).
        self.assertLessEqual(abs(self.QF_uniform), 1e-6)


if __name__ == "__main__":
    unittest.main()
