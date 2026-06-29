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
import os
import sys
import unittest

import numpy as np

import tessera as T

cob = T.cobordism

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _holed_surface import holed_surface  # noqa: E402


def _refined_s3(n_refine=12):
    # A refined S^3 (Betti [1,0,0,1]) with enough tetrahedra that a disjoint top-cell
    # pair exists — the minimal S^3 has none (every facet pair shares a ridge).
    sig = T.Signature(3, T.Lorentzian)
    st = T.Spacetime(T.Metric(True, sig), T.CDT, 1.0, 1.0, T.PREFERRED,
                     T.SimplexBoundarySphere(3))
    st.build()
    for e in st.getEdgeList().toVector():
        e.setSquaredLength(1.0)
    for seed in range(n_refine):
        mv = T.AddMove(st, seed, False, T.PachnerMode.PreGeometric, False)
        if mv.propose():
            mv.apply()
    for i, e in enumerate(st.getEdgeList().toVector()):
        e.setSquaredLength(1.0 + 0.013 * (i % 6))
    return st


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
        # identity on a holed icosahedron (a b1 register).
        st, es, holes, periods = holed_surface(degree=1)
        # A non-proportional target leaves the carried span (r_U > 0); shifting a
        # single component is enough (a uniform shift of near-equal periods stays
        # in-span).
        target = [complex(z) for z in periods[0]]
        target[0] += 0.5

        r_u = es.residualForPeriods(holes, target)
        self.assertGreater(r_u, 1e-3, "target should be non-realizable (r_U > 0)")
        g = np.asarray(es.residualForPeriodsGradient(holes, target), float)
        self.assertLess(abs(_euler_lhs(st, g) + r_u) / r_u, 1e-9,
                        "Euler identity Σℓ²∂r_U = −r_U failed at k=1")

    def test_k2_euler_on_b2_register(self):
        # k=2 (tetrahedral holes, the b₂ color register): the exact Euler identity is
        # the analytic correctness certificate where no prior gradient existed. The
        # b₂ register is opened from first principles by a disjoint pair of surgical
        # cone-outs on a refined S^3 (raising b₂ by 1); the Euler identity is exact on
        # any such metric 3-complex and does not depend on the register's topology.
        st = _refined_s3()
        cells = sorted(tuple(sorted(v.getId() for v in c.getVertices()))
                       for c in st.getTopSimplices())
        pair = None
        for i, a in enumerate(cells):
            for b in cells[i + 1:]:
                if set(a).isdisjoint(b):
                    pair = (a, b)
                    break
            if pair:
                break
        self.assertIsNotNone(pair, "refined S^3 must contain a disjoint cell pair")
        a, b = pair
        sc = cob.SurgicalCone(st)
        self.assertTrue(sc.coneOut(list(a))[0])   # opens the manifold (b₃ → 0)
        self.assertTrue(sc.coneOut(list(b))[0])   # disjoint ⇒ raises b₂ by 1

        es = cob.EigenstateSynthesis(st, 2)
        holes = [list(a), list(b)]                 # 2 holes, 1 mode ⇒ over-constrained
        target = [complex(1.0), complex(0.3)]      # non-carriable ⇒ r_U > 0

        r_u = es.residualForPeriods(holes, target)
        self.assertGreater(r_u, 1.0)
        g = np.asarray(es.residualForPeriodsGradient(holes, target), float)
        self.assertLess(abs(_euler_lhs(st, g) + r_u) / r_u, 1e-11,
                        "Euler identity Σℓ²∂r_U = −r_U failed at k=2")


if __name__ == "__main__":
    unittest.main()
