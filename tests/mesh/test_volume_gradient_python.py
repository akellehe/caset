# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Exact analytic gradient of a simplex's signed `volume()` w.r.t. edge ℓ² (#461).

`Simplex.volumeGradient` is the per-degree **Hodge inner-product weight** gradient
(the weights `W_k` are signed simplex volumes) — the keystone for an arbitrary-degree
analytic `∂L_k/∂ℓ²`, hence the general-k `r_U` gradient. By Jacobi's formula on the
Gram determinant, `∂V/∂ℓ²_e = (V/2)·tr(G⁻¹ ∂_e G)`, reusing the same
`gramMatrix`/`cofactorMatrix` machinery as the circumcentric `dualVolumeGradient`
(#354). Here it must match a central finite difference of `volume()` to ~machine
precision across every degree (triangle, tetrahedron, pentatope).
"""
import unittest

import tessera as T


def _jittered_s4():
    sig = T.Signature(4, T.Lorentzian)
    st = T.Spacetime(T.Metric(True, sig), T.CDT, 1.0, 1.0, T.PREFERRED,
                     T.SimplexBoundarySphere(4))
    st.build()
    for i, e in enumerate(st.getEdgeList().toVector()):
        e.setSquaredLength(1.0 + 0.017 * (i % 5))
    # materialize sub-simplices down to triangles
    for s in list(st.getTopSimplices()):
        for f in s.getFacets():
            for g in f.getFacets():
                g.getFacets()
    return st


def _key(a, b):
    return (min(a, b), max(a, b))


class VolumeGradientTest(unittest.TestCase):
    def test_matches_finite_difference_every_degree(self):
        st = _jittered_s4()
        em = {_key(e.getSource().getId(), e.getTarget().getId()): e
              for e in st.getEdgeList().toVector()}
        by_size = {}
        for s in st.getSimplices():
            n = len([v for v in s.getVertices()])
            by_size.setdefault(n, []).append(s)
        # 3 = triangle (W_2), 4 = tetrahedron (W_3), 5 = pentatope (top)
        self.assertTrue({3, 4, 5}.issubset(by_size.keys()))

        h = 1e-6
        for size in (3, 4, 5):
            worst, tested = 0.0, 0
            for s in by_size[size]:
                for (a, b), ga in s.volumeGradient().items():
                    e = em.get((a, b))
                    if e is None:
                        continue
                    o = e.getSquaredLength()
                    e.setSquaredLength(o + h)
                    vp = s.volume()
                    e.setSquaredLength(o - h)
                    vm = s.volume()
                    e.setSquaredLength(o)
                    worst = max(worst, abs(ga - (vp - vm) / (2 * h)))
                    tested += 1
            self.assertGreater(tested, 0, f"no edge-derivatives tested at size {size}")
            self.assertLess(worst, 1e-8, f"volumeGradient inexact at size {size}")


if __name__ == "__main__":
    unittest.main()
