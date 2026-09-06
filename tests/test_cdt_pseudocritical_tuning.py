# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Tests for the pseudo-critical tuning of the cosmological coupling k4.

The pseudo-critical coupling is the value of k4 at which the four-volume neither
grows nor shrinks under the bare action.  Below it the volume grows, and since
the volume-fixing term constrains N41 alone, the (4,1) sector then settles at a
fixed multiple of its target rather than at the target (#965).

References:
  [RU]  Ambjorn, Jurkiewicz, Loll, "Reconstructing the Universe",
        Phys. Rev. D 72 (2005), arXiv:hep-th/0505154v2
"""

import unittest

import tessera


class TestPseudoCriticalTuning(unittest.TestCase):
    K0 = 2.2
    DELTA = 0.6
    SEED = 20260905
    BUILD = 1600
    TARGET = 2000

    #: The closed form tune() starts its search from: the coupling that zeroes
    #: the Regge action change of one (2,2d) add move, entropy not accounted for.
    CLOSED_FORM_K4 = (K0 + 6.0 * DELTA) / 6.0 - 2.0 * DELTA

    def _simulation(self, k4, epsilon=None, target=None, seed=SEED):
        target = self.TARGET if target is None else target
        epsilon = (1.0 / target) if epsilon is None else epsilon
        sig = tessera.Signature(4, tessera.Lorentzian)
        st = tessera.Spacetime(tessera.Metric(True, sig), tessera.CDT,
                               1.0, 1.0, tessera.PREFERRED, tessera.Toroid())
        st.setSeed(seed)
        st.build(self.BUILD)
        cdt = tessera.CDTSimulation(st, self.K0, k4, self.DELTA, epsilon, target)
        cdt.setSeed(seed)
        return st, cdt

    @staticmethod
    def _volume(st):
        return st.getN41() + st.getN32()

    def _drift_per_sweep(self, st, cdt, sweeps):
        """Mean relative change in the four-volume per sweep."""
        before = self._volume(st)
        cdt.sweep(sweeps)
        return (self._volume(st) - before) / (before * sweeps)

    def test_tuned_coupling_is_above_the_closed_form_estimate(self):
        """The closed form ignores the entropy of the available triangulations,
        so it lies below the coupling at which the volume stops growing.  It
        returns -0.233 here; the measured value is near +0.44."""
        _, cdt = self._simulation(k4=0.5)
        cdt.tune()
        self.assertGreater(cdt.getK4(), self.CLOSED_FORM_K4 + 0.4)

    def test_four_volume_is_stationary_at_the_tuned_coupling(self):
        """With no volume-fixing term k4 alone sets the volume, so at the tuned
        coupling the four-volume neither grows nor collapses."""
        st, cdt = self._simulation(k4=0.5, epsilon=0.0, target=1)
        cdt.tune()
        self.assertLess(abs(self._drift_per_sweep(st, cdt, 200)), 3.0e-3)

    def test_a_sub_critical_coupling_grows_the_volume(self):
        """The tuned value means something only if a coupling below it behaves
        differently: 0.3 lower, the same complex inflates an order of magnitude
        faster."""
        _, tuned = self._simulation(k4=0.5, epsilon=0.0, target=1)
        tuned.tune()
        st, cdt = self._simulation(k4=tuned.getK4() - 0.3, epsilon=0.0, target=1)
        self.assertGreater(self._drift_per_sweep(st, cdt, 100), 4.0e-3)

    def test_volume_fixing_holds_the_four_one_sector_at_its_target(self):
        """Below the pseudo-critical coupling the volume-fixing term cannot hold
        N41 at the target; it settles where the restoring force balances the
        supercritical drive, measured at 1.55x the target under the closed form.
        At the tuned coupling N41 tracks the target itself."""
        st, cdt = self._simulation(k4=0.5)
        cdt.tune()
        cdt.sweep(800)
        self.assertLess(abs(st.getN41() - self.TARGET), 0.25 * self.TARGET)

    def test_tuning_keeps_the_configuration_it_was_given(self):
        """Measurements above critical shrink the complex and measurements below
        it inflate one.  Tuning must hand back a configuration the caller can
        still run."""
        st, cdt = self._simulation(k4=0.5)
        before = self._volume(st)
        cdt.tune()
        self.assertGreaterEqual(self._volume(st), before // 2)
        self.assertLessEqual(self._volume(st), 2 * before)

    def test_tuning_is_reproducible_under_a_fixed_seed(self):
        """Both the spacetime and the Markov chain are seeded, so two runs of
        the search return the same coupling."""
        _, first = self._simulation(k4=0.5)
        first.tune()
        _, second = self._simulation(k4=0.5)
        second.tune()
        self.assertEqual(first.getK4(), second.getK4())


if __name__ == "__main__":
    unittest.main()
