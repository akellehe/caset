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

"""The exact analytic gradient of the complex dual (Sorkin) Regge action matches
finite differences to machine precision -- the regression guard for the
hand-derived gradient (Simplex::lorentzianDeficitAngleGradient,
Simplex::dualVolumeGradient, ReggeSolver::actionGradientExact). Run on the merge
cobordism so both real (spacelike) and complex (boost) hinges are exercised."""

import importlib.util
import os
import sys
import unittest

import tessera

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.join(_HERE, "..", "..", "examples", "cobordism",
                        "merge_cobordism.py")


def _load_merge():
    spec = importlib.util.spec_from_file_location("merge_cobordism", _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["merge_cobordism"] = module
    spec.loader.exec_module(module)
    return module


MC = _load_merge()
_FD = 1e-6           # central-difference step in l^2
_TOL = 1e-5          # analytic vs FD agreement


def _central(setter, get, h=_FD):
    o = setter(None)
    setter(o + h); sp = get()
    setter(o - h); sm = get()
    setter(o)
    return (sp - sm) / (2.0 * h)


class ExactActionGradientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = MC.MergeCobordism()
        cls.st = cls.m.st
        cls.st.materializeFacets()
        cls.edges = cls.st.getEdgeList().toVector()
        cls.emap, cls.eidx = {}, {}
        for i, e in enumerate(cls.edges):
            a, b = e.getSource().getId(), e.getTarget().getId()
            key = (min(a, b), max(a, b))
            cls.emap[key] = e
            cls.eidx[key] = i
        cls.hinges = {tuple(sorted(int(v.getId()) for v in s.getVertices())): s
                      for s in cls.st.getSimplices()
                      if len(s.getVertices()) == 2}

    def _action(self):
        return complex(tessera.ReggeSolver(
            self.st, tessera.MatterConfiguration()).dualReggeAction())

    def _set(self, edge):
        def setter(val):
            if val is None:
                return edge.getSquaredLength().real
            edge.setSquaredLength(float(val)); self.st.materializeFacets()
            return None
        return setter

    def test_action_gradient_exact_matches_fd(self):
        rs = tessera.ReggeSolver(self.st, tessera.MatterConfiguration())
        g = [complex(z) for z in rs.actionGradientExact()]
        self.assertEqual(len(g), len(self.edges))
        worst = 0.0
        saw_complex = False
        for key, edge in list(self.emap.items())[:24]:
            fd = _central(self._set(edge), self._action)
            ana = g[self.eidx[key]]
            worst = max(worst, abs(ana - fd))
            saw_complex = saw_complex or abs(fd.imag) > 1e-3
        self.assertLess(worst, _TOL, f"worst |analytic-FD| = {worst:.2e}")
        self.assertTrue(saw_complex, "expected some boost (complex) gradients")

    def test_deficit_gradient_matches_fd_complex_hinge(self):
        # the most-complex-deficit hinge: exercises the boost branch of d(eps)/dl^2
        hk = max(self.hinges, key=lambda k: abs(
            complex(self.hinges[k].lorentzianDeficitAngle()).imag))
        hs = self.hinges[hk]
        grad = hs.lorentzianDeficitAngleGradient()
        worst = 0.0
        for e in list(grad)[:8]:
            fd = _central(self._set(self.emap[e]),
                          lambda: complex(hs.lorentzianDeficitAngle()))
            worst = max(worst, abs(complex(grad[e]) - fd))
        self.assertLess(worst, _TOL)

    def test_dualvolume_gradient_matches_fd(self):
        hk = next(iter(self.hinges))
        hs = self.hinges[hk]
        grad = hs.dualVolumeGradient()
        self.assertGreater(len(grad), 0)
        worst = 0.0
        for e in list(grad)[:8]:
            fd = _central(self._set(self.emap[e]), hs.dualVolume).real
            worst = max(worst, abs(float(grad[e]) - fd))
        self.assertLess(worst, _TOL)

    # --- the Lorentzian action is genuinely complex; never reduce it to Re ---
    # These guard a recurring mistake: building a variational principle / EOM / diagnostic
    # on Re S alone. The imaginary part (boost & spacelike-hinge contributions) is real
    # physics; a stationary-action objective is delta S = 0 for the FULL complex S, i.e.
    # the gradient norm Sum|dS/dl^2|^2 must include |d Im S|^2. Dropping it lets the
    # geometry drift along directions the imaginary part actually constrains (observed:
    # the Re-only objective drifted up in overall scale because d Im S -- which curbs that
    # drift -- was missing). The margins below are comfortably met (~0.6) on the merge;
    # the threshold catches a regression that silently zeroes the imaginary part.

    def test_action_is_materially_complex(self):
        S = self._action()
        self.assertGreater(
            abs(S.imag), 0.1 * abs(S.real),
            f"Im S={S.imag:.3f} is not material vs Re S={S.real:.3f}; the Lorentzian "
            "action is complex and its imaginary part must not be dropped")

    def test_stationarity_residual_requires_imaginary_part(self):
        rs = tessera.ReggeSolver(self.st, tessera.MatterConfiguration())
        g = [complex(z) for z in rs.actionGradientExact()]
        reN = sum(z.real * z.real for z in g)   # ||d Re S||^2
        imN = sum(z.imag * z.imag for z in g)   # ||d Im S||^2
        # A Re-only stationary-action objective would use reN and miss imN entirely.
        self.assertGreater(
            imN, 0.1 * reN,
            f"||d Im S||^2={imN:.2f} is not material vs ||d Re S||^2={reN:.2f}; the "
            "stationary-action objective must use the full complex gradient Sum|dS|^2")


if __name__ == "__main__":
    unittest.main()
