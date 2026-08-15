# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Exact analytic gradient of the symmetric metric Hodge Laplacian (#461).

`HodgeLaplacian.laplacianGradient(k, a, b)` is `∂L_k^sym/∂ℓ²_e` at arbitrary degree
— the keystone (with `Simplex.volumeGradient`) for the arbitrary-k `r_U` analytic
gradient. The rigorous check is the **exact Euler homogeneity identity**: `L_k^sym`
is homogeneous of degree `−½` in `ℓ²` (each weight `W_j = |vol|` scales as
`(ℓ²)^{j/2}`, and the `B_k` exponents sum to `−¼`), so

    Σ_e ℓ²_e · ∂L_k/∂ℓ²_e  =  −½ · L_k     (exactly).

This is independent of finite difference (which is roundoff-limited and does not
converge — the optimizer uses the analytic gradient, never FD).
"""
import math
import unittest

import numpy as np

import tessera as T
import cmath

cob = T.cobordism


def _holed_s3(n_refine=12):
    # A refined S^3 (Betti [1,0,0,1]) opened by a surgical cone-out so it carries a
    # nonempty b_2 register; the Euler-homogeneity identity below is exact on any
    # such metric 3-complex (it does not depend on the register's topology).
    sig = T.Signature(3, T.Lorentzian)
    st = T.Spacetime(T.Metric(True, sig), T.CDT, 1.0, 1.0, T.PREFERRED,
                     T.SimplexBoundarySphere(3))
    st.build()
    for e in st.getEdgeList().toVector():
        e.setLength(cmath.sqrt(complex(1.0)))
    for seed in range(n_refine):
        mv = T.AddMove(st, seed, False, T.PachnerMode.PreGeometric, False)
        if mv.propose():
            mv.apply()
    tops = [tuple(sorted(v.getId() for v in c.getVertices()))
            for c in st.getTopSimplices()]
    sc = cob.SurgicalCone(st)
    for t in tops:                      # open one hole (b_3 -> 0, exposes a 2-cycle)
        if sc.coneOut(list(t))[0]:
            break
    for i, e in enumerate(st.getEdgeList().toVector()):
        e.setLength(cmath.sqrt(complex(1.0 + 0.013 * (i % 6))))
    return st


def _Lmat(hl, k):
    L = np.asarray(hl.laplacian(k, True), complex)
    n = int(round(math.sqrt(L.size)))
    return L.reshape(n, n).real


class LaplacianGradientHandCalcTest(unittest.TestCase):
    def test_euler_homogeneity_identity(self):
        # Σ_e ℓ²_e ∂L_k/∂ℓ²_e = −½ L_k, exactly, at every register degree.
        st = _holed_s3()
        hl = cob.HodgeLaplacian(st)
        for k in (1, 2):
            L = _Lmat(hl, k)
            n = L.shape[0]
            self.assertGreater(n, 0)
            acc = np.zeros((n, n))
            for e in st.getEdgeList().toVector():
                a, b = e.getSource().getId(), e.getTarget().getId()
                g = np.asarray(hl.laplacianGradient(k, a, b), float).reshape(n, n)
                acc += (e.getLength() * e.getLength()).real * g
            self.assertLess(np.max(np.abs(acc + 0.5 * L)), 1e-12,
                            f"Euler identity Σℓ²∂L = −½L failed at k={k}")

    def test_symmetric_and_empty_below_k1(self):
        st = _holed_s3()
        hl = cob.HodgeLaplacian(st)
        # ∂L_k inherits L_k's symmetry
        n = _Lmat(hl, 2).shape[0]
        e = st.getEdgeList().toVector()[0]
        g = np.asarray(hl.laplacianGradient(2, e.getSource().getId(),
                                            e.getTarget().getId()), float).reshape(n, n)
        self.assertLess(np.max(np.abs(g - g.T)), 1e-12)
        # k < 1 has no metric Laplacian gradient
        self.assertEqual(hl.laplacianGradient(0, e.getSource().getId(),
                                              e.getTarget().getId()), [])


if __name__ == "__main__":
    unittest.main()
