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

_RELAX = 25


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


if __name__ == "__main__":
    unittest.main()
