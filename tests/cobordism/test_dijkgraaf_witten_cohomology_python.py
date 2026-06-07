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

"""Cohomology-class Dijkgraaf–Witten state sum — T2 robustness (#139).

The DW weight ∏_t ω(g_01,g_12,g_23)^{ε_t} depends only on the cohomology class
[g] ∈ H¹(W;ℤ₂) (the cup-cube ⟨g³,[W]⟩ is a cohomology pairing; the trivial class
has weight 1), so the gauge-redundant brute-force sum over all 2^{dim Z¹} flat
cocycles collapses to a sum over the 2^{b₁} classes:

    Z(W) = 2^{-b₀} · Σ_{[g] ∈ H¹(W;ℤ₂)} weight([g]).

``DijkgraafWitten.partitionFunctionByCohomology`` enumerates H¹ = Z¹/B¹ — the
cocycles ker(∂₂ᵀ) (``gf2_nullspace``) modulo the coboundaries im(∂₁ᵀ), quotiented
by ``gf2_cohomology_basis`` — and sums one representative per class. It is the
*same invariant* as the brute-force ``partitionFunction`` (cross-checked here on
S³, S²×S¹, RP³), but feasible where the brute force is not: T³ has dim Z¹ = 29,
beyond ``gf2_span``'s materializable cap of 24, yet only 2^{b₁} = 2³ = 8 classes.

Acceptance (#139):
  * Agrees with brute-force partitionFunction on S³, S²×S¹, RP³ (both cocycles).
  * Z(T³): brute force is infeasible (raises); the cohomology sum gives
    Z_Trivial = 2^{b₁-1} = 4 and Z_Sign = 4 (T³ is a sign negative control, the
    cup cube g³ vanishes on every 1-class — H*(T³;ℤ₂) = Λ(x,y,z)).
  * Reports the speedup 2^{b₁} vs 2^{dim Z¹} (= 2^{26} for T³) via stateSumTerms.
"""

import unittest

import tessera

cobordism = tessera.cobordism
DijkgraafWitten = cobordism.DijkgraafWitten
Cocycle = cobordism.Cocycle


def _build(topology):
    """Build a closed oriented 3-manifold Spacetime from a fixture topology."""
    signature = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, signature)
    spacetime = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                                  tessera.PREFERRED, topology)
    spacetime.build()  # delegates to topology.build(); the count is ignored
    return spacetime


def _chain(spacetime):
    return cobordism.ChainComplex.fromSpacetime(spacetime)


def _three_sphere():
    # S³ = ∂Δ⁴: 5 vertices, betti (1,0,0,1); dim Z¹ = 4 (brute force feasible).
    return _build(tessera.SimplexBoundarySphere(3))


def _s2_cross_s1():
    # S²×S¹ via the simplicial product: 12 vertices, betti (1,1,1,1); dim Z¹ = 12.
    return _build(tessera.SimplicialProduct(tessera.SimplexBoundarySphere(2),
                                            tessera.SimplexBoundarySphere(1)))


def _rp3():
    # RP³ (Walkup 11-vertex): betti (1,0,0,1), bettiGF2 (1,1,1,1); dim Z¹ = 11.
    # The sign cocycle distinguishes it (cup cube t³ ≠ 0): Z_Sign = 0 ≠ 1.
    return _build(tessera.RealProjectiveSpace())


def _three_torus():
    # T³ = S¹ × S¹ × S¹ via nested simplicial products: 3·3·3 = 27 vertices,
    # betti (1,3,3,1). dim Z¹ = b₁ + |V| − b₀ = 3 + 27 − 1 = 29 > 24, so the
    # brute-force state sum cannot materialize the flat space — only 2³ = 8
    # cohomology classes do.
    circle = tessera.SimplexBoundarySphere(1)          # S¹ = ∂Δ², 3 vertices
    torus2 = tessera.SimplicialProduct(circle, circle)  # T² = S¹×S¹, 9 vertices
    return _build(tessera.SimplicialProduct(torus2, circle))  # T³, 27 vertices


def _dim_z1(spacetime):
    """dim Z¹ = b₁(ℤ₂) + |V| − b₀(ℤ₂) (== nullity of the coboundary ∂₂ᵀ)."""
    chain = _chain(spacetime)
    betti_gf2 = chain.bettiNumbersGF2()
    return betti_gf2[1] + chain.numSimplices(0) - betti_gf2[0]


class TestCohomologyMatchesBruteForce(unittest.TestCase):
    """The cohomology-class sum is the same invariant as the brute-force sum,
    on every fixture small enough for the brute force, for both cocycles."""

    def _check(self, spacetime, cocycle):
        dw = DijkgraafWitten(spacetime, cocycle)
        brute = dw.partitionFunction()
        closed = dw.partitionFunctionByCohomology()
        self.assertAlmostEqual(closed.imag, 0.0, places=9)
        self.assertAlmostEqual(closed.real, brute.real, places=9)
        self.assertAlmostEqual(abs(closed - brute), 0.0, places=9)

    def test_three_sphere(self):
        spacetime = _three_sphere()
        for cocycle in (Cocycle.Trivial, Cocycle.Sign):
            with self.subTest(cocycle=cocycle):
                self._check(spacetime, cocycle)

    def test_s2_cross_s1(self):
        spacetime = _s2_cross_s1()
        for cocycle in (Cocycle.Trivial, Cocycle.Sign):
            with self.subTest(cocycle=cocycle):
                self._check(spacetime, cocycle)

    def test_rp3(self):
        # RP³ is the positive control: the two cocycles disagree (1 vs 0), and
        # the cohomology sum must reproduce *both* brute-force values exactly.
        spacetime = _rp3()
        for cocycle in (Cocycle.Trivial, Cocycle.Sign):
            with self.subTest(cocycle=cocycle):
                self._check(spacetime, cocycle)
        self.assertAlmostEqual(
            DijkgraafWitten(spacetime, Cocycle.Trivial)
            .partitionFunctionByCohomology().real, 1.0, places=9)
        self.assertAlmostEqual(
            DijkgraafWitten(spacetime, Cocycle.Sign)
            .partitionFunctionByCohomology().real, 0.0, places=9)

    def test_matches_convention_anchor(self):
        # Z_Trivial = 2^{b₁(ℤ₂) − 1} for a connected closed oriented W.
        for fixture in (_three_sphere, _s2_cross_s1, _rp3):
            spacetime = fixture()
            b1 = _chain(spacetime).bettiNumbersGF2()[1]
            z = DijkgraafWitten(spacetime, Cocycle.Trivial) \
                .partitionFunctionByCohomology()
            with self.subTest(fixture=fixture.__name__):
                self.assertAlmostEqual(z.real, 2.0 ** (b1 - 1), places=9)


class TestThreeTorus(unittest.TestCase):
    """Z(T³): the headline case — brute force is infeasible, the cohomology sum
    is not, and it lands on the convention value 2^{b₁-1} = 4 for both cocycles."""

    def setUp(self):
        self.t3 = _three_torus()

    def test_fixture_really_is_t3(self):
        chain = _chain(self.t3)
        self.assertEqual(chain.dimension(), 3)
        self.assertEqual(chain.numSimplices(0), 27)
        self.assertEqual(chain.bettiNumbers(), [1, 3, 3, 1])
        self.assertEqual(chain.bettiNumbersGF2(), [1, 3, 3, 1])
        self.assertEqual(_dim_z1(self.t3), 29)

    def test_brute_force_is_infeasible(self):
        # dim Z¹ = 29 > 24: gf2_span refuses to materialize the flat space, so
        # the brute-force partitionFunction cannot run on T³.
        self.assertGreater(_dim_z1(self.t3), 24)
        with self.assertRaises(ValueError):
            DijkgraafWitten(self.t3, Cocycle.Trivial).partitionFunction()

    def test_cohomology_trivial_is_four(self):
        # Z_Trivial = 2^{-b₀} · 2^{b₁} = 2^{b₁-1} = 2² = 4.
        z = DijkgraafWitten(self.t3, Cocycle.Trivial).partitionFunctionByCohomology()
        self.assertAlmostEqual(z.imag, 0.0, places=9)
        self.assertAlmostEqual(z.real, 4.0, places=9)

    def test_cohomology_sign_is_four_negative_control(self):
        # The cup cube g³ vanishes on every 1-class of T³ (H*(T³;ℤ₂) = Λ(x,y,z)),
        # so the sign twist does not distinguish: Z_Sign = Z_Trivial = 4.
        z = DijkgraafWitten(self.t3, Cocycle.Sign).partitionFunctionByCohomology()
        self.assertAlmostEqual(z.imag, 0.0, places=9)
        self.assertAlmostEqual(z.real, 4.0, places=9)

    def test_sign_does_not_distinguish_t3(self):
        z_trivial = DijkgraafWitten(self.t3, Cocycle.Trivial) \
            .partitionFunctionByCohomology()
        z_sign = DijkgraafWitten(self.t3, Cocycle.Sign) \
            .partitionFunctionByCohomology()
        self.assertAlmostEqual(abs(z_trivial - z_sign), 0.0, places=9)


class TestSpeedupReport(unittest.TestCase):
    """stateSumTerms reports the cohomology collapse's speedup: 2^{b₁} classes
    versus 2^{dim Z¹} cocycles."""

    def test_t3_speedup(self):
        terms = DijkgraafWitten(_three_torus(), Cocycle.Trivial).stateSumTerms()
        self.assertEqual(terms.first_betti, 3)
        self.assertEqual(terms.cocycle_dimension, 29)
        self.assertEqual(terms.cohomology_terms, 2.0 ** 3)        # 8
        self.assertEqual(terms.brute_force_terms, 2.0 ** 29)
        self.assertEqual(terms.speedup, 2.0 ** (29 - 3))          # 2^26
        # Report it (visible with -s): the whole point of the ticket.
        print(f"\nZ(T³) speedup: {terms.cohomology_terms:.0f} cohomology classes "
              f"vs {terms.brute_force_terms:.0f} flat cocycles "
              f"= {terms.speedup:.3g}x fewer terms")

    def test_terms_consistent_on_small_fixtures(self):
        for fixture in (_three_sphere, _s2_cross_s1, _rp3, _three_torus):
            spacetime = fixture()
            chain = _chain(spacetime)
            terms = DijkgraafWitten(spacetime, Cocycle.Trivial).stateSumTerms()
            with self.subTest(fixture=fixture.__name__):
                self.assertEqual(terms.first_betti, chain.bettiNumbersGF2()[1])
                self.assertEqual(terms.cocycle_dimension, _dim_z1(spacetime))
                self.assertEqual(terms.cohomology_terms, 2.0 ** terms.first_betti)
                self.assertEqual(terms.brute_force_terms,
                                 2.0 ** terms.cocycle_dimension)
                self.assertEqual(terms.speedup,
                                 terms.brute_force_terms / terms.cohomology_terms)


class TestCohomologyBasisHelper(unittest.TestCase):
    """The GF(2) H¹-basis primitive ``gf2_cohomology_basis`` (Z/B quotient)."""

    def test_quotient_of_full_space_by_a_line(self):
        # Z = GF(2)³ (the three unit vectors), B = span{e0}; H = Z/B has dim 2.
        cocycles = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        coboundaries = [[1, 0, 0]]
        reps = cobordism.gf2_cohomology_basis(cocycles, coboundaries, 3)
        self.assertEqual(len(reps), 2)  # dim Z − dim B = 3 − 1

        # The span of the reps is a transversal: 2² = 4 elements, one per coset
        # of B = {000, 100}, i.e. distinct on the non-pivot coordinates (1, 2).
        classes = cobordism.gf2_span(reps, 3)
        self.assertEqual(len(classes), 4)
        self.assertEqual(len({(v[1], v[2]) for v in classes}), 4)

    def test_trivial_quotient_is_empty(self):
        # B already spans Z ⇒ H = 0 ⇒ no representatives.
        reps = cobordism.gf2_cohomology_basis([[1, 0], [0, 1]],
                                              [[1, 0], [0, 1]], 2)
        self.assertEqual(reps, [])
        self.assertEqual(cobordism.gf2_span(reps, 2), [[0, 0]])


if __name__ == "__main__":
    unittest.main()
