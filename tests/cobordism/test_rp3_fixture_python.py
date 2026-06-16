# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""RP³ fixture + Dijkgraaf–Witten T3/P3 positive control (#124).

The ℤ₂ Dijkgraaf–Witten sign cocycle ω(a,b,c) = (-1)^{abc} twists the untwisted
state sum by the mod-2 cup cube (-1)^{⟨g³,[W]⟩}. That pairing vanishes on T³ and
S²×S¹ (every 1-class has g³ = 0 — the negative controls), so the sign cocycle
cannot distinguish there. It is nonzero exactly where some 1-class has g³ ≠ 0 —
canonically RP³ = L(2,1), whose mod-2 cohomology ring is ℤ₂[t]/t⁴ with t³ ≠ 0.

Two deliverables:

  1. **It really is RP³.** ``RealProjectiveSpace`` is the Walkup vertex-minimal
     11-vertex triangulation; the chain-complex homology is the proof —
     bettiNumbers (ℚ) = [1,0,0,1], bettiNumbersGF2 = [1,1,1,1], and the gap is
     the 2-torsion H₁ = ℤ₂ (torsion(1) = [2]) that separates RP³ from S²×S¹.
     Closed oriented pseudomanifold, so a fundamental class (b₃ = 1) exists.

  2. **T3 / P3 — the positive control.** Z_Sign(RP³) = 0 ≠ 1 = Z_Trivial(RP³),
     so the sign cocycle distinguishes *some* W (it does not on the S²×S¹
     negative control). Z_Trivial = 2^{b₁(ℤ₂) - 1} = 2^0 = 1.
"""

import itertools
import unittest

import tessera

cobordism = tessera.cobordism
DijkgraafWitten = cobordism.DijkgraafWitten
Cocycle = cobordism.Cocycle


def _build(topology):
    signature = tessera.Signature(topology.dimension(), tessera.Lorentzian)
    metric = tessera.Metric(True, signature)
    spacetime = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                                  tessera.PREFERRED, topology)
    spacetime.build()  # delegates to topology.build(); numSimplices ignored
    return spacetime


def _chain(topology):
    return cobordism.ChainComplex.fromSpacetime(_build(topology))


def _rp3():
    return tessera.RealProjectiveSpace()


def _s2_cross_s1():
    # Negative control: S²×S¹ via the simplicial product, betti (1,1,1,1) but no
    # torsion. dim Z¹ = 12 ≤ 24, so the brute-force state sum admits it.
    return tessera.SimplicialProduct(tessera.SimplexBoundarySphere(2),
                                     tessera.SimplexBoundarySphere(1))


class TestRealProjectiveSpaceIsRP3(unittest.TestCase):
    """The homology assertions that validate the hardcoded facet list."""

    def test_f_vector(self):
        # Walkup's vertex- and facet-minimal RP³: f = (11, 51, 80, 40).
        chain = _chain(_rp3())
        self.assertEqual([chain.numSimplices(k) for k in range(4)],
                         [11, 51, 80, 40])

    def test_betti_numbers_rational(self):
        self.assertEqual(_chain(_rp3()).bettiNumbers(), [1, 0, 0, 1])

    def test_betti_numbers_gf2(self):
        self.assertEqual(_chain(_rp3()).bettiNumbersGF2(), [1, 1, 1, 1])

    def test_h1_is_two_torsion(self):
        # The ℤ₂ in H₁ — the invariant that distinguishes RP³ from S²×S¹, which
        # has the same ℤ/2 Betti numbers but no torsion.
        self.assertEqual(_chain(_rp3()).torsion(1), [2])
        self.assertEqual(_chain(_s2_cross_s1()).torsion(1), [])

    def test_euler_characteristic_zero(self):
        # Closed odd-dimensional manifold ⇒ χ = 0; cross-checked against the
        # alternating Betti sum (Euler–Poincaré).
        chain = _chain(_rp3())
        self.assertEqual(chain.eulerCharacteristic(), 0)
        self.assertEqual(sum((-1) ** k * b
                             for k, b in enumerate(chain.bettiNumbers())), 0)

    def test_closed_oriented_pseudomanifold(self):
        # Every tetrahedron is a 4-vertex cell, every triangle bounds exactly two
        # of them (closed 3-pseudomanifold), and a ±1 fundamental class on all 40
        # tetrahedra exists (b₃ = 1 ⇒ orientable).
        chain = _chain(_rp3())
        tops = [tuple(t) for t in chain.orientedTopSimplices()]
        self.assertTrue(all(len(t) == 4 for t in tops))
        facet_counts = {}
        for top in tops:
            for facet in itertools.combinations(top, 3):
                facet_counts[facet] = facet_counts.get(facet, 0) + 1
        self.assertTrue(all(count == 2 for count in facet_counts.values()))
        fundamental = chain.fundamentalClass()
        self.assertEqual(chain.bettiNumbers()[3], 1)
        self.assertEqual(len(fundamental), len(tops))
        self.assertTrue(all(abs(coeff) == 1 for coeff in fundamental))

    def test_flat_space_small_enough_for_state_sum(self):
        # dim Z¹ = b₁(ℤ₂) + (|V| − b₀(ℤ₂)) = 1 + (11 − 1) = 11 ≤ 24, so gf2Span
        # can materialize the whole flat space for the brute-force sum.
        chain = _chain(_rp3())
        betti_gf2 = chain.bettiNumbersGF2()
        dim_z1 = betti_gf2[1] + chain.numSimplices(0) - betti_gf2[0]
        self.assertEqual(dim_z1, 11)
        self.assertLessEqual(dim_z1, 24)


class TestDijkgraafWittenPositiveControl(unittest.TestCase):
    """T3 / P3: the sign cocycle distinguishes RP³ (and not the negative
    control S²×S¹)."""

    @staticmethod
    def _z(spacetime, cocycle):
        return DijkgraafWitten(spacetime, cocycle).partitionFunction()

    def test_trivial_partition_function_is_one(self):
        # Z_Trivial = 2^{b₁(ℤ₂) − 1} = 2^0 = 1 for the connected closed oriented
        # RP³ (b₁(ℤ₂) = 1).
        z = self._z(_build(_rp3()), Cocycle.Trivial)
        self.assertAlmostEqual(z.imag, 0.0, places=9)
        self.assertAlmostEqual(z.real, 1.0, places=9)

    def test_sign_partition_function_vanishes(self):
        # The cup cube t³ ≠ 0 on RP³ ⇒ the sign twist kills the state sum.
        z = self._z(_build(_rp3()), Cocycle.Sign)
        self.assertAlmostEqual(z.imag, 0.0, places=9)
        self.assertAlmostEqual(z.real, 0.0, places=9)

    def test_sign_distinguishes_rp3(self):
        # The positive control: Z_Sign ≠ Z_Trivial on RP³.
        spacetime = _build(_rp3())
        z_trivial = self._z(spacetime, Cocycle.Trivial)
        z_sign = self._z(spacetime, Cocycle.Sign)
        self.assertGreater(abs(z_trivial - z_sign), 0.5)

    def test_sign_does_not_distinguish_negative_control(self):
        # Contrast: on S²×S¹ the cup cube vanishes, so Z_Sign == Z_Trivial.
        spacetime = _build(_s2_cross_s1())
        z_trivial = self._z(spacetime, Cocycle.Trivial)
        z_sign = self._z(spacetime, Cocycle.Sign)
        self.assertAlmostEqual(abs(z_trivial - z_sign), 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
