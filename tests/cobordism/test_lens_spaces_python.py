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

"""Lens-space L(p,q) fixtures (#140).

Closed-3-manifold fixtures beyond RP³ = L(2,1). ``LensSpace(p, q)`` builds
Lutz's vertex-minimal simplicial triangulation of the lens space L(p,q) — the
quotient of S³ by a free ℤ_p action, a closed orientable 3-manifold whose only
nontrivial reduced homology is H₁ = ℤ_p.

The homology is the proof the hardcoded facet lists are right: for every case
the rational Betti numbers are [1,0,0,1] and torsion(1) = [p] (H₁ = ℤ_p),
cross-checked against an independent Smith-normal-form computation of the same
triangulations. Each is a closed oriented pseudomanifold (every triangle in
exactly two tetrahedra, a ±1 fundamental class on all tops, b₃ = 1) with χ = 0.

The three shipped cases keep dim Z¹ ≤ 24, so the brute-force Dijkgraaf–Witten
ℤ₂ state sum is feasible. None is distinguished by the DW sign cocycle (unlike
the RP³ positive control): L(3,1) and L(5,2) have odd torsion invisible to a
ℤ₂ theory, and L(4,1) has p ≡ 0 (mod 4) so the mod-2 cup cube t³ vanishes.
Z_Trivial = 2^{b₁(ℤ₂) − 1} is 1 for L(4,1) and ½ for the odd-p cases.
"""

import itertools
import unittest

import tessera

cobordism = tessera.cobordism
DijkgraafWitten = cobordism.DijkgraafWitten
Cocycle = cobordism.Cocycle


# (p, q) -> expected invariants of Lutz's vertex-minimal L(p,q).
CASES = {
    (3, 1): dict(f_vector=[12, 66, 108, 54], betti_gf2=[1, 0, 0, 1],
                 dim_z1=11, z_trivial=0.5),
    (4, 1): dict(f_vector=[14, 84, 140, 70], betti_gf2=[1, 1, 1, 1],
                 dim_z1=14, z_trivial=1.0),
    (5, 2): dict(f_vector=[14, 86, 144, 72], betti_gf2=[1, 0, 0, 1],
                 dim_z1=13, z_trivial=0.5),
}


def _build(topology):
    signature = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, signature)
    spacetime = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                                  tessera.PREFERRED, topology)
    spacetime.build()  # delegates to topology.build(); numSimplices ignored
    return spacetime


def _chain(topology):
    return cobordism.ChainComplex.fromSpacetime(_build(topology))


class TestLensSpaceHomology(unittest.TestCase):
    """The homology assertions that validate each hardcoded facet list:
    H₁ = ℤ_p is the proof it really is L(p,q)."""

    def test_accessors(self):
        for (p, q) in CASES:
            ls = tessera.LensSpace(p, q)
            self.assertEqual((ls.p(), ls.q()), (p, q))

    def test_f_vector(self):
        for (p, q), want in CASES.items():
            with self.subTest(p=p, q=q):
                chain = _chain(tessera.LensSpace(p, q))
                self.assertEqual([chain.numSimplices(k) for k in range(4)],
                                 want["f_vector"])

    def test_betti_numbers_rational(self):
        # b = (1,0,0,1): a closed orientable 3-manifold with no rational H₁/H₂.
        for (p, q) in CASES:
            with self.subTest(p=p, q=q):
                self.assertEqual(_chain(tessera.LensSpace(p, q)).bettiNumbers(),
                                 [1, 0, 0, 1])

    def test_h1_is_p_torsion(self):
        # The defining invariant: H₁(L(p,q)) = ℤ_p, i.e. torsion(1) = [p]. No
        # torsion in H₂ (torsion(2) = []), so the manifold is orientable.
        for (p, q) in CASES:
            with self.subTest(p=p, q=q):
                chain = _chain(tessera.LensSpace(p, q))
                self.assertEqual(chain.torsion(1), [p])
                self.assertEqual(chain.torsion(2), [])

    def test_betti_numbers_gf2(self):
        # Universal coefficients over ℤ₂: the even-p case L(4,1) (2-torsion in
        # H₁) reads (1,1,1,1), while the odd-p cases match the rational (1,0,0,1).
        for (p, q), want in CASES.items():
            with self.subTest(p=p, q=q):
                self.assertEqual(_chain(tessera.LensSpace(p, q)).bettiNumbersGF2(),
                                 want["betti_gf2"])

    def test_euler_characteristic_zero(self):
        # Closed odd-dimensional manifold ⇒ χ = 0, cross-checked against the
        # alternating Betti sum (Euler–Poincaré).
        for (p, q) in CASES:
            with self.subTest(p=p, q=q):
                chain = _chain(tessera.LensSpace(p, q))
                self.assertEqual(chain.eulerCharacteristic(), 0)
                self.assertEqual(sum((-1) ** k * b
                                     for k, b in enumerate(chain.bettiNumbers())),
                                 0)

    def test_closed_oriented_pseudomanifold(self):
        # Tetrahedra only, every triangle in exactly two of them (closed
        # 3-pseudomanifold), and a ±1 fundamental class on all tops (b₃ = 1 ⇒
        # closed, connected, oriented).
        for (p, q) in CASES:
            with self.subTest(p=p, q=q):
                chain = _chain(tessera.LensSpace(p, q))
                tops = [tuple(t) for t in chain.orientedTopSimplices()]
                self.assertTrue(all(len(t) == 4 for t in tops))
                facet_counts = {}
                for top in tops:
                    for facet in itertools.combinations(top, 3):
                        facet_counts[facet] = facet_counts.get(facet, 0) + 1
                self.assertTrue(all(c == 2 for c in facet_counts.values()))
                fundamental = chain.fundamentalClass()
                self.assertEqual(chain.bettiNumbers()[3], 1)
                self.assertEqual(len(fundamental), len(tops))
                self.assertTrue(all(abs(coeff) == 1 for coeff in fundamental))

    def test_flat_space_small_enough_for_state_sum(self):
        # dim Z¹ = b₁(ℤ₂) + (|V| − b₀(ℤ₂)) ≤ 24, so the brute-force DW sum can
        # materialize the whole flat space.
        for (p, q), want in CASES.items():
            with self.subTest(p=p, q=q):
                chain = _chain(tessera.LensSpace(p, q))
                betti_gf2 = chain.bettiNumbersGF2()
                dim_z1 = betti_gf2[1] + chain.numSimplices(0) - betti_gf2[0]
                self.assertEqual(dim_z1, want["dim_z1"])
                self.assertLessEqual(dim_z1, 24)


class TestLensSpaceDijkgraafWitten(unittest.TestCase):
    """A feasible DW state-sum sanity value per fixture. These lens spaces are
    ℤ₂ negative controls: the sign cocycle does not distinguish any of them, so
    Z_Sign = Z_Trivial (contrast the RP³ positive control, Z_Sign = 0 ≠ 1)."""

    @staticmethod
    def _z(spacetime, cocycle):
        return DijkgraafWitten(spacetime, cocycle).partitionFunction()

    def test_trivial_partition_function(self):
        # Z_Trivial = 2^{b₁(ℤ₂) − 1}: 1 for L(4,1), ½ for the odd-p cases.
        for (p, q), want in CASES.items():
            with self.subTest(p=p, q=q):
                z = self._z(_build(tessera.LensSpace(p, q)), Cocycle.Trivial)
                self.assertAlmostEqual(z.imag, 0.0, places=9)
                self.assertAlmostEqual(z.real, want["z_trivial"], places=9)

    def test_sign_cocycle_does_not_distinguish(self):
        # Odd p: H¹(;ℤ₂) = 0. p = 4 ≡ 0 (mod 4): mod-2 cup cube t³ = 0. Either
        # way the sign twist is trivial, so Z_Sign = Z_Trivial.
        for (p, q) in CASES:
            with self.subTest(p=p, q=q):
                spacetime = _build(tessera.LensSpace(p, q))
                z_trivial = self._z(spacetime, Cocycle.Trivial)
                z_sign = self._z(spacetime, Cocycle.Sign)
                self.assertAlmostEqual(abs(z_trivial - z_sign), 0.0, places=9)


class TestLensSpaceUnsupported(unittest.TestCase):
    """Unsupported (p,q) fail fast at construction."""

    def test_unsupported_raises(self):
        # (2,1) = RP³ is RealProjectiveSpace; (6,1) is simply not shipped.
        for (p, q) in [(2, 1), (6, 1), (3, 2)]:
            with self.subTest(p=p, q=q):
                with self.assertRaises(ValueError):
                    tessera.LensSpace(p, q)


if __name__ == "__main__":
    unittest.main()
