"""Flavor without a register: the documented NEGATIVE result (#414, epic #410).

The proton (`uud`, `Q = +1`) and the neutron (`udd`, `Q = 0`) differ only in flavor.
This ticket asks whether the `u`/`d` label can be read off the EXISTING `W_ABC`
geometry as a split of the register it already carries --- spatio-temporally (timelike
vs spacelike) or via the Dirac-Kahler taste multiplicity --- WITHOUT a parallel hole
register. The answer is NO, and these tests PIN the obstruction so the suite can never
silently regress to a false-positive "flavor found".

The precise cause is structural (NOT dimensional --- the dimensional reach is #429's):
the three quark windows are one A4 orbit (the window-cycling `g` is a transitive 3-cycle
on {A, B, C}) and the transport intertwines color Z3 exactly (`M P_in = P_out M`). Any
per-window measurable invariant under base-vertex relabeling --- the #412 G6 property the
flavor audit (F4) REQUIRES --- is therefore CONSTANT on the orbit, so its u-vs-d margin is
identically 0. Relabeling-invariance (F4) and a real discriminator (F3, margin >= 0.1) are
mutually exclusive on the symmetric register. Both candidate measurables collapse:
  (i)  the spatio-temporal split: the relaxed geometry has 0 timelike edges, so the
       electric (timelike-leg) sector is empty --- there is nothing to split;
  (ii) the Dirac-Kahler charge: equal across windows AND positive-definite (a norm),
       so it can neither separate the windows nor sign u(+2/3) vs d(-1/3).

Fixed seed `numpy.random.default_rng(410414)` (G7); the build is the relaxed `W_ABC`
color singlet read OFF the relaxed geometry (G8). See `docs/design/proton_flavor_split.md`.
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
import proton_flavor_split as FS  # noqa: E402

_RELAX = 25
_SEED = 410414

# The F3 discriminator threshold from the ticket: a real u/d split must differ by
# >= 0.1. The negative result is that every candidate margin sits orders below this.
_F3_MARGIN = 0.1


@pytest.mark.slow
class ProtonFlavorNegativeResultTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(_SEED)
        cls.m = FS.build(_RELAX)
        cls.tl, cls.sp, cls.nu = FS.causal_census(cls.m)
        cls.period_fracs = FS.candidate_i_period_split(cls.m)
        cls.field = FS.candidate_i_field_split(cls.m)
        cls.dk, cls.mult = FS.candidate_ii_dk_charge(cls.m)
        cls.labels, cls.dk_margin = FS.label_vector(cls.m)

    # --- N1: candidate (i) --- the electric (timelike) sector is structurally empty ---
    def test_N1_electric_sector_is_empty(self):
        # The relaxed symmetric junction is Riemannian: no timelike edges, hence no
        # electric (timelike-leg) plaquettes -- candidate (i) has nothing to split.
        self.assertEqual(self.tl, 0)
        for _efrac, q_e, n_e in self.field:
            self.assertEqual(n_e, 0)              # no electric plaquettes
            self.assertLessEqual(q_e, 1e-9)       # Q_electric = oint_S E = 0 per window

    # --- N2: candidate (i) period split COLLAPSES (window-independent) ---
    def test_N2_period_causal_split_collapses(self):
        margin = max(self.period_fracs) - min(self.period_fracs)
        for f in self.period_fracs:
            self.assertLessEqual(f, 1e-9)         # every window period is 100% spacelike
        self.assertLessEqual(margin, 1e-9)
        self.assertLess(margin, _F3_MARGIN)       # cannot meet the F3 discriminator

    # --- N3: candidate (i') field E/B split COLLAPSES ---
    def test_N3_field_strength_split_collapses(self):
        efracs = [ef for ef, _, _ in self.field]
        margin = max(efracs) - min(efracs)
        for ef in efracs:
            self.assertLessEqual(ef, 1e-9)        # entirely magnetic, no electric part
        self.assertLessEqual(margin, 1e-9)
        self.assertLess(margin, _F3_MARGIN)

    # --- N4: candidate (ii) DK charge is A4-EQUAL and POSITIVE-DEFINITE ---
    def test_N4_dirac_kahler_charge_equal_and_positive(self):
        qs = [q for q, _ in self.dk]
        margin = max(qs) - min(qs)
        self.assertLessEqual(margin, 1e-4)        # equal across windows (A4-symmetric)
        for q, dens_min in self.dk:
            self.assertGreater(q, 0.0)            # a norm <Phi,Phi>_W, strictly positive
            self.assertGreaterEqual(dens_min, -1e-12)  # density j^0 = W|Phi|^2 >= 0
        # the taste multiplicity is a FIXED framework constant of 4 lattice tastes,
        # never a 2-valued isospin doublet, and independent of the window.
        self.assertEqual(self.mult, 4)

    # --- N5: THE DISCRIMINATOR COLLAPSES --- uud and udd are indistinguishable ---
    def test_N5_discriminator_collapses(self):
        # The would-be per-window u/d label vector is the constant (zero) vector: it is
        # neither uud nor udd, so the two assignments cannot be separated. This directly
        # negates the ticket's F3 ("the discriminator is real, margin >= 0.1").
        self.assertEqual([abs(x) for x in self.labels], [0.0, 0.0, 0.0])
        self.assertLess(self.dk_margin, _F3_MARGIN)

    # --- N6: STRUCTURAL CAUSE --- equivariance forces the collapse (F4 vs F3) ---
    def test_N6_relabeling_invariance_forces_collapse(self):
        # The window-cycling g is a TRANSITIVE 3-cycle on {A, B, C}: P_out has the cyclic
        # Z3 spectrum {1, w, w^2}, so g permutes the three windows in a single orbit.
        p_in, p_out = P._window_cycle_rep(P._windows(self.m))
        angles = sorted(float(np.angle(e)) for e in np.linalg.eigvals(p_out))
        for got, exp in zip(angles, sorted([0.0, 2 * np.pi / 3, -2 * np.pi / 3])):
            self.assertAlmostEqual(got, exp, delta=1e-6)   # transitive 3-cycle (exact)
        # The transport intertwines color Z3 on the symmetric apex interior (#413):
        # M P_in = P_out M -- machine-zero (4.5e-14) on the exact uniform metric, and
        # ~1e-5 on this lightly-relaxed (RELAX=25) build, still orders below the prism's
        # 4.26e-2. Any relabeling-invariant per-window measurable (F4/G6) commutes with g,
        # so it is CONSTANT on the {A,B,C} orbit up to the intertwining residual.
        M = P._transport(self.m)
        residual = np.linalg.norm(M @ p_in - p_out @ M) / (np.linalg.norm(M) + 1e-30)
        self.assertLessEqual(residual, 1e-4)               # symmetric-apex scale, not prism
        # the smoking gun: the per-window discriminator margin is BOUNDED BY (the same
        # order as) the intertwining residual -- the collapse is forced by the equivariance,
        # not a coincidence. F4 therefore implies margin ~0, mutually exclusive with F3
        # (margin >= 0.1): no symmetry-respecting split of the register carries flavor.
        self.assertLessEqual(self.dk_margin, 10.0 * residual + 1e-9)

    # --- N7: the success TARGET is unreachable on the symmetric register (xfail) ---
    @unittest.expectedFailure
    def test_N7_flavor_charge_target_unreached(self):
        """DOCUMENTED OBSTRUCTION (expected to fail): the ticket's success target is
        Q(uud) = +1 and Q(udd) = 0 from DISTINCT per-window u/d labels. On the
        A4-symmetric register the per-window label vector is constant (margin
        ~1e-6 << 0.1), so a uud labeling and a udd labeling are indistinguishable --
        they yield the SAME (neutral) charge. The assertion below demands a real
        discriminator and a +1 proton charge; it cannot be met without a
        symmetry-BREAKING isospin structure (new construction), so it xfails by design.
        """
        qs = np.array([q for q, _ in self.dk])
        # a real discriminator would separate one window from the other two by >= 0.1...
        self.assertGreaterEqual(float(qs.max() - qs.min()), _F3_MARGIN)
        # ...and the proton (uud) would carry total charge +1.
        q_uud = 2 * FS._Q_UP + FS._Q_DOWN  # = +1 IF the labels were assignable; they are not
        self.assertAlmostEqual(q_uud, 1.0, delta=1e-9)
        # the conjunction (real margin AND a faithful +1 read OFF the geometry) is what
        # fails: the margin is ~1e-6, so the first assertion above already xfails.


if __name__ == "__main__":
    unittest.main()
