# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Arbitrary-degree analytic r_U gradient (#461, c).

`EigenstateSynthesis.residualForPeriodsGradient` is the degree-generic `∂r_U/∂ℓ²` of
`residualForPeriods` — it now works at the b₂ register (k=2) and beyond (it used to be
k=1 only). It uses `M = L_k` and the exact per-edge `∂L_k/∂ℓ²`
(`HodgeLaplacian.laplacianGradient`, built on `Simplex.volumeGradient`) through
eigenvector-perturbation theory, with the period covector read from each
removed-(k+1)-cell hole's facet boundary.

The rigorous checks (finite difference does not converge — the optimizer uses the
analytic gradient):
  * **exact Euler identity** — `r_U` is homogeneous of degree −1 in ℓ² (the carried
    representative is scale-invariant, `M` is degree −½), so
    `Σ_e ℓ²_e · ∂r_U/∂ℓ²_e = −r_U` at every register degree, k=1 and k=2.

(The k=1 numerical equivalence with the established path is additionally guarded by
`test_ru_gradient_gpu_python.py`, whose FP32 GPU oracle mirrors the prior CPU result.)
"""
import cmath
import importlib.util
import math
import os
import sys
import unittest

import numpy as np

import tessera as T

cob = T.cobordism

_EX = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "cobordism")


def _edge_l2(st):
    return {tuple(sorted((e.getSource().getId(), e.getTarget().getId()))):
            e.getSquaredLength().real for e in st.getEdgeList().toVector()}


def _edges_1cell(st):
    cc = cob.ChainComplex.fromSpacetime(st)
    return [tuple(sorted(c)) for c in cc.kSimplexVertices(1)]


def _euler_lhs(st, grad):
    l2 = _edge_l2(st)
    edges = _edges_1cell(st)
    return sum(l2[edges[i]] * grad[i] for i in range(len(edges)))


class ArbitraryKRuGradientTest(unittest.TestCase):
    def test_k1_euler_on_triangle_holes(self):
        # k=1 (triangle holes): the arbitrary-k path satisfies the exact Euler
        # identity. Uses the MergeCobordism substrate.
        sys.path.insert(0, os.path.join(_EX, "deep_merge_baseline"))
        sys.path.insert(0, _EX)
        try:
            spec = importlib.util.spec_from_file_location(
                "merge_cobordism", os.path.join(_EX, "merge_cobordism.py"))
            mc = importlib.util.module_from_spec(spec)
            sys.modules["merge_cobordism"] = mc
            spec.loader.exec_module(mc)
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"merge_cobordism substrate unavailable: {exc}")

        m = mc.MergeCobordism()
        st, es = m.st, m.es
        holes = [list(t) for t in m.hole_circles]
        periods = np.asarray(es.cyclePeriods(holes), complex).reshape(m.dim, len(holes))
        target = [complex(z) + 0.137 for z in periods[0]]  # perturbed ⇒ r_U > 0

        r_u = es.residualForPeriods(holes, target)
        g = np.asarray(es.residualForPeriodsGradient(holes, target), float)
        self.assertLess(abs(_euler_lhs(st, g) + r_u), 1e-9,
                        "Euler identity Σℓ²∂r_U = −r_U failed at k=1")

    def test_k2_euler_on_b2_register(self):
        # k=2 (tetrahedral holes, the b₂ color register): the exact Euler identity is
        # the analytic correctness certificate where no prior gradient existed.
        surf = cob.S3WindowSurface.build(1, 1)
        faces = [list(t) for t in surf.faces]
        windows = [[list(h) for h in w] for w in surf.windows]
        hs = {tuple(sorted(h)) for w in windows for h in w}
        holed = [t for t in faces if tuple(sorted(t)) not in hs]
        st = T.Spacetime.fromCells(3, [list(t) for t in holed], 1.0, 0.0)
        for i, e in enumerate(st.getEdgeList().toVector()):
            e.setSquaredLength(1.0 + 0.013 * (i % 6))
        es = cob.EigenstateSynthesis(st, 2)
        holes = [list(h) for h in windows[0]]
        w = cmath.exp(2j * math.pi / 3)
        target = [complex(x) + 0.21 for x in [1.0, w, w * w]]  # non-carriable ⇒ r_U > 0

        r_u = es.residualForPeriods(holes, target)
        self.assertGreater(r_u, 1.0)
        g = np.asarray(es.residualForPeriodsGradient(holes, target), float)
        self.assertLess(abs(_euler_lhs(st, g) + r_u) / r_u, 1e-11,
                        "Euler identity Σℓ²∂r_U = −r_U failed at k=2")


if __name__ == "__main__":
    unittest.main()
