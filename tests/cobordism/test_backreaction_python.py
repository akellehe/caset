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

"""Backreaction at linear order
(``examples/cobordism/backreaction_kappa_flow.py``).

The carried state's stress-energy sources the fill metric through the discrete
linearized Einstein equation H*delta_l = -kappa*T, H the Regge Hessian (the
graviton kinetic operator) at the unit metric and T_e = w_e|h(e)|^2 the
Hellmann-Feynman stress-energy density. These tests pin the structure:

  1. **The Euclidean action is unbounded below** (edge shrink lowers
     S_Regge) -- which is why the work is at linear order, not a nonlinear
     minimum, and why the metric guard matters.
  2. **The graviton operator and the source.** H is full rank on the
     boundary-pinned interior edges; the stress-energy T is concentrated on
     the worldtube edges and lies in range(H) (a genuine source, not gauge).
  3. **Charge curves the fill, near the charge.** The sourced deflection
     delta_l = -kappa*H^+ T solves the Einstein equation to machine
     precision, is exactly O(kappa), and is concentrated on the worldtubes.
  4. **The observable is protected, the representative responds.**
     Realizability is invariant along the flow (periods are topological);
     the field energy responds at second order (the backreaction does work);
     the anchor-normalized transport value is robust.
  5. **The metric guard** fires on a near-collapsed edge the topological
     (#275) dual gate passes.
"""

import importlib.util
import os
import sys
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.join(_HERE, "..", "..", "examples", "cobordism",
                        "backreaction_kappa_flow.py")


def _load_example():
    spec = importlib.util.spec_from_file_location("backreaction_kappa_flow",
                                                  _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["backreaction_kappa_flow"] = module
    spec.loader.exec_module(module)
    return module


BR = _load_example()

# Shared fixtures: one flow, one graviton operator, one source (FD is the
# cost; build once).
_FLOW = BR.BackreactionFlow()
_H = _FLOW.regge_hessian(_FLOW.iw0)
_FLOW.set_metric(_FLOW.iw0)
_T = _FLOW.stress_energy()
_HPINV = np.linalg.pinv(_H, rcond=1e-6)
_WT = _FLOW.worldtube.astype(bool)


def _deflect(kappa):
    dl = -kappa * (_HPINV @ _T)
    valid = _FLOW.set_metric(_FLOW.iw0 + dl)
    return dl, valid


class ConformalModeTest(unittest.TestCase):
    """1: the Euclidean action is unbounded below."""

    def test_edge_shrink_lowers_the_action(self):
        _FLOW.set_metric(_FLOW.iw0)
        s_unit = _FLOW.regge()
        _FLOW.set_metric(_FLOW.iw0 * 0.5)
        s_half = _FLOW.regge()
        _FLOW.set_metric(_FLOW.iw0)
        self.assertLess(s_half, s_unit)


class GravitonAndSourceTest(unittest.TestCase):
    """2: the operator and the stress-energy."""

    def test_hessian_is_full_rank(self):
        evals = np.linalg.eigvalsh(_H)
        rank = int(np.sum(np.abs(evals) > 1e-6 * np.max(np.abs(evals))))
        self.assertEqual(rank, _FLOW.n)

    def test_stress_energy_is_worldtube_concentrated(self):
        wt = float(np.mean(np.abs(_T[_WT])))
        bulk = float(np.mean(np.abs(_T[~_WT])))
        self.assertGreater(wt, 3.0 * bulk)

    def test_stress_energy_is_in_range_of_H(self):
        frac = float(np.linalg.norm((_HPINV @ _H) @ _T)
                     / np.linalg.norm(_T))
        self.assertGreater(frac, 0.5)


class ChargeCurvesTheFillTest(unittest.TestCase):
    """3: the sourced deflection."""

    def test_einstein_equation_holds_to_machine_precision(self):
        kappa = 1.0
        dl, _valid = _deflect(kappa)
        proj = _HPINV @ _H
        resid = float(np.linalg.norm(proj @ (_H @ dl + kappa * _T))
                      / np.linalg.norm(kappa * _T))
        _FLOW.set_metric(_FLOW.iw0)
        self.assertLess(resid, 1e-6)

    def test_deflection_is_linear_in_kappa(self):
        d1, _ = _deflect(0.5)
        d2, _ = _deflect(1.0)
        _FLOW.set_metric(_FLOW.iw0)
        self.assertTrue(np.allclose(d2, 2.0 * d1, rtol=1e-9))

    def test_deflection_is_worldtube_concentrated(self):
        dl, _ = _deflect(1.0)
        _FLOW.set_metric(_FLOW.iw0)
        self.assertGreater(np.linalg.norm(dl[_WT]),
                           np.linalg.norm(dl[~_WT]))


class ObservableProtectedTest(unittest.TestCase):
    """4: realizability invariant, energy responds, value robust."""

    def test_realizability_invariant_along_the_flow(self):
        for kappa in (0.5, 1.0, 2.0):
            _deflect(kappa)
            self.assertLess(_FLOW.identity_residual(), 1e-9)
            self.assertEqual(BR.L1.match_gate(_FLOW.fill.emergent_gate()),
                             "Identity")
        _FLOW.set_metric(_FLOW.iw0)

    def test_field_energy_responds_at_second_order(self):
        _FLOW.set_metric(_FLOW.iw0)
        e0 = _FLOW.energy()
        _deflect(2.0)
        e2 = _FLOW.energy()
        _FLOW.set_metric(_FLOW.iw0)
        self.assertGreater(e2, e0 + 1e-5)   # work done against the geometry

    def test_normalized_value_is_robust(self):
        _FLOW.set_metric(_FLOW.iw0)
        g0 = _FLOW.gram_dev()
        _deflect(2.0)
        g2 = _FLOW.gram_dev()
        _FLOW.set_metric(_FLOW.iw0)
        self.assertLess(abs(g2 - g0), 0.01 * g0)   # protected though l moves O(1)


class MetricGuardTest(unittest.TestCase):
    """5: the metric half of the dual gate."""

    def test_guard_fires_on_a_near_collapsed_edge(self):
        collapse = _FLOW.iw0.copy()
        collapse[0] = 1e-8
        valid = _FLOW.set_metric(collapse)
        _FLOW.set_metric(_FLOW.iw0)
        self.assertFalse(valid)

    def test_topological_gate_passes_what_the_metric_guard_catches(self):
        # the #275 dual-complex gate is metric-blind: it reads "ok" at the
        # collapsed edge the metric guard rejects
        collapse = _FLOW.iw0.copy()
        collapse[0] = 1e-8
        _FLOW.fill.es.setInteriorWeights([float(x) for x in collapse])
        _FLOW.fill.read_spectral()
        ok, _why = _FLOW.fill.es.dualComplexValid()
        _FLOW.set_metric(_FLOW.iw0)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
