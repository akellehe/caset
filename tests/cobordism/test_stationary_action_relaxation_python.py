# MIT License
# Copyright (c) 2025 Andrew Kelleher
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""The per-edge stationary-action relaxation's invariants
(``examples/cobordism/stationary_action_relaxation.py``).

The objective is Φ = ‖∇S‖² + Γ·r_U — the discrete Einstein equation δS = 0 for
the FULL COMPLEX Lorentzian action, with the realizability residual r_U as the
SINGLE matter source (no Dirichlet energy). These guard the things that are easy
to break silently:

  * the matter term is exactly Γ·``residualForPeriods`` — no Dirichlet source
    sneaks back in;
  * the action and its stationarity keep the imaginary part (never reduced to
    Re S) — the Lorentzian physics that pins the overall scale;
  * the analytic gradient (FD-Hessian of the exact ``actionGradientExact`` plus
    the low-rank r_U perturbation) matches a finite difference of Φ;
  * relaxing actually descends ‖∇S‖² toward δS = 0 while keeping the register.
"""
import importlib.util
import os
import sys
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.join(_HERE, "..", "..", "examples", "cobordism",
                        "stationary_action_relaxation.py")


def _load_example():
    spec = importlib.util.spec_from_file_location("stationary_action_relaxation",
                                                  _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["stationary_action_relaxation"] = module
    spec.loader.exec_module(module)
    return module


SAR = _load_example()
_RX = SAR.StationaryActionRelaxer(gamma=1e3)    # one shared level-0 relaxer


class StationaryActionRelaxationTest(unittest.TestCase):
    def setUp(self):
        _RX.set_var(_RX.x0)            # reset the geometry before each test

    def test_objective_no_dirichlet_full_complex_action(self):
        """Φ = ‖∇S‖² + Γ·r_U, the matter term is r_U ALONE (no Dirichlet), and
        the action is complex (Lorentzian) with a nonzero imaginary stationarity."""
        Phi, grad, statres, rU, S = _RX.objective(_RX.x0)
        self.assertEqual(len(grad), len(_RX.VAR))
        self.assertGreater(statres, 0.0)                 # ‖∇S‖² > 0 away from δS=0
        self.assertLess(rU, 1e-3)                        # the register is realized
        # Φ is exactly ‖∇S‖² + Γ·r_U ...
        self.assertAlmostEqual(Phi, statres + _RX.gamma * rU, places=6)
        # ... and the matter source is residualForPeriods alone — no Dirichlet term
        # (the relaxer carries no energy() functional at all).
        self.assertFalse(hasattr(_RX, "energy"))
        rU_src = _RX.es.residualForPeriods(_RX.holes, _RX.target_c)
        self.assertAlmostEqual(Phi - statres, _RX.gamma * rU_src, places=6)
        # the action is Lorentzian (complex), and ∂Im S contributes to ‖∇S‖² —
        # the imaginary part is kept, never reduced to Re S.
        rs = SAR.tessera.ReggeSolver(_RX.st, SAR.tessera.MatterConfiguration())
        dS = rs.actionGradientExact()
        g = np.array([complex(dS[_RX.EIDX[k]]) for k in _RX.VAR])
        self.assertGreater(abs(S.imag), 1e-6)
        self.assertGreater(float(np.linalg.norm(g.imag)), 1e-6)

    def test_analytic_gradient_matches_finite_difference(self):
        """∇Φ (analytic: FD-Hessian of the exact action gradient + low-rank dr_U)
        matches a finite difference of Φ. Loose tol — the Hessian-vector term in
        the analytic gradient is itself a finite difference."""
        self.assertLess(_RX.check_gradient(n=2, h=1e-6), 1e-3)

    def test_relaxation_descends_stationarity_residual(self):
        """A few L-BFGS-B steps decrease ‖∇S‖² (toward the discrete Einstein
        equation δS = 0) while keeping the register realized and Lorentzian."""
        _, _, sr0, _, _ = _RX.objective(_RX.x0)
        res = _RX.relax(maxiter=6)
        _, _, srf, rUf, Sf = _RX.objective(res.x)
        self.assertLess(srf, sr0)                        # ‖∇S‖² descended
        self.assertLess(rUf, 1e-3)                       # realizability maintained
        self.assertGreater(abs(Sf.imag), 1e-6)           # still Lorentzian


if __name__ == "__main__":
    unittest.main()
