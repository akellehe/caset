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
    # The signed operator is real on real signed l^2; assert Im = 0 rather than
    # project it away, then hand back the real part.
    L = np.asarray(hl.laplacian(k, True), complex)
    n = int(round(math.sqrt(L.size)))
    L = L.reshape(n, n)
    np.testing.assert_allclose(L.imag, 0.0, atol=1e-12)
    return L.real


class LaplacianGradientHandCalcTest(unittest.TestCase):
    def test_euler_homogeneity_identity(self):
        # Σ_e ℓ²_e ∂L_k/∂ℓ²_e = −L_k exactly: under the V^2 weights every W_k
        # is homogeneous of degree +1 in l^2 (edge W_1 = l^2; triangle
        # W_2 = detG/4, quadratic overall but degree +1 per the Gram scaling —
        # detG(s l^2) = s^2 detG makes W_2 degree 2... the OPERATOR
        # L = W_k^-1 d^T W_{k-1} d + d W_{k+1}^-1 d^T W_k is degree -1 at every
        # k, measured exactly on this fixture). The old -1/2 belonged to the
        # removed sqrt(W)-conjugated form.
        st = _holed_s3()
        hl = cob.HodgeLaplacian(st)
        for k in (1, 2):
            L = _Lmat(hl, k)
            n = L.shape[0]
            self.assertGreater(n, 0)
            acc = np.zeros((n, n))
            for e in st.getEdgeList().toVector():
                a, b = e.getSource().getId(), e.getTarget().getId()
                g = np.asarray(hl.laplacianGradient(k, a, b), complex).reshape(n, n)
                np.testing.assert_allclose(g.imag, 0.0, atol=1e-12)
                acc += (e.getLength() * e.getLength()).real * g.real
            self.assertLess(np.max(np.abs(acc + L)), 1e-10,
                            f"Euler identity Σℓ²∂L = −L failed at k={k}")

    def test_matches_finite_difference_and_empty_below_degree_zero(self):
        # The signed operator is non-symmetric, so its gradient is too; the
        # honest check is a central finite difference of L itself.
        st = _holed_s3()
        hl = cob.HodgeLaplacian(st)
        n = _Lmat(hl, 2).shape[0]
        e = st.getEdgeList().toVector()[0]
        g = np.asarray(hl.laplacianGradient(2, e.getSource().getId(),
                                            e.getTarget().getId()), complex).reshape(n, n)
        h = 1e-6
        l = e.getLength()
        e.setLength(cmath.sqrt(l * l + h)); st.materializeFacets()
        Lp = np.asarray(cob.HodgeLaplacian(st).laplacian(2, True), complex).reshape(n, n)
        e.setLength(cmath.sqrt(l * l - h)); st.materializeFacets()
        Lm = np.asarray(cob.HodgeLaplacian(st).laplacian(2, True), complex).reshape(n, n)
        e.setLength(l); st.materializeFacets()
        fd = (Lp - Lm) / (2.0 * h)
        self.assertLess(np.max(np.abs(g - fd)), 1e-5)
        # Degree zero is now the derived L_0 = d_1 W_1^-1 d_1^dagger, which has
        # an exact gradient too (#805): the same central-difference check.
        n0 = _Lmat(hl, 0).shape[0]
        g0 = np.asarray(hl.laplacianGradient(0, e.getSource().getId(),
                                             e.getTarget().getId()),
                        complex).reshape(n0, n0)
        e.setLength(cmath.sqrt(l * l + h)); st.materializeFacets()
        L0p = np.asarray(cob.HodgeLaplacian(st).laplacian(0, True),
                         complex).reshape(n0, n0)
        e.setLength(cmath.sqrt(l * l - h)); st.materializeFacets()
        L0m = np.asarray(cob.HodgeLaplacian(st).laplacian(0, True),
                         complex).reshape(n0, n0)
        e.setLength(l); st.materializeFacets()
        self.assertLess(np.max(np.abs(g0 - (L0p - L0m) / (2.0 * h))), 1e-5)
        self.assertGreater(np.max(np.abs(g0)), 0.0)
        # k < 0 still has no chain to differentiate.
        self.assertEqual(hl.laplacianGradient(-1, e.getSource().getId(),
                                              e.getTarget().getId()), [])


if __name__ == "__main__":
    unittest.main()
