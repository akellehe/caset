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

"""Dijkgraaf–Witten ℤ₂ state sum (#108).

The closed-W state sum

    Z(W) = (1/2^|V|) Σ_{flat g} ∏_t ω(g_01, g_12, g_23)^{ε_t}

over the flat ℤ₂ connections g ∈ C¹ (the GF(2) nullspace of the coboundary
d₁ = ∂₂ᵀ), each tetrahedron weighted by a 3-cocycle ω raised to its orientation
sign ε_t (the fundamental class). ω is either Trivial (ω≡1) or Sign
(ω(a,b,c) = (-1)^{abc}), the two classes of Z³(ℤ₂; U(1)).

Acceptance, in dependency order:

1. **3-cocycle (pentagon) identity — first.** ω must be a genuine 3-cocycle, or
   the state sum is not gauge-invariant and the rest is meaningless. Checked with
   an independent Python oracle (with a deliberately non-cocycle cochain as a
   negative control so the oracle has teeth) against ``DijkgraafWitten.isCocycle``.

2. **Convention anchor.** For a connected closed oriented 3-manifold the
   untwisted partition function is ``Z_Trivial(W) = 2^{b_1(W;ℤ₂) - 1}``. Verified
   exactly on S³ (b₁=0 → ½) and S²×S¹ (b₁=1 → 1), with b₁ read from the chain
   complex, and cross-checked against an independent numpy reimplementation of the
   whole state sum (a true oracle for the edge-indexing/orientation wiring).

3. **The Sign twist.** ``Z_Sign(W)`` twists the untwisted value by
   ``(-1)^{⟨g∪g∪g, [W]⟩}``, the mod-2 cup cube. It therefore differs from
   ``Z_Trivial(W)`` **iff the cup cube is nonzero on W**.

   NOTE — correction to the ticket's worked example. Ticket #108 proposed T³ as
   the manifold where the sign cocycle distinguishes. That is mathematically
   incorrect: H*(T³;ℤ₂) is the exterior algebra Λ(x,y,z), in which every
   degree-one class squares to zero, so the cup cube g³ vanishes for *every*
   g ∈ H¹(T³;ℤ₂) and ``Z_Sign(T³) = Z_Trivial(T³)``. The same vanishing holds on
   S²×S¹ and S³ (so those are negative controls, asserted below). The cobordism
   design note (``docs/.../cobordism.md`` §6, P3) states the requirement
   correctly — "nontrivial ω distinguishable from trivial on **some** W", with
   "Σ×S¹, T³, lens spaces" as candidate topologies; the distinguishers are the
   **lens spaces** (RP³ = L(2,1) and friends), whose cup cube t³ ≠ 0.

   The positive control is therefore deferred: the only closed oriented
   3-manifold with t³ ≠ 0 that is small enough for this brute-force state sum
   (gf2Span enumerates the full flat space, so nullity = |V| must stay ≲ 24) is
   the vertex-minimal RP³, and tessera ships no RP³/lens-space fixture yet (the
   named S²×S¹ fixture is itself a separate, unmerged ticket). Per #108's own
   guidance for unavailable fixtures, this control is noted as deferred; the
   expected values (Z_Trivial(RP³)=1, Z_Sign(RP³)=0) are pinned in
   ``test_lens_space_distinction_is_deferred`` so the follow-up fixture has a
   target.

4. **Negative control.** On S²×S¹ (and S³) the sign cocycle does **not**
   distinguish: ``Z_Sign == Z_Trivial``. A wrong edge map, transpose, or ω would
   generically break this, so it doubles as a correctness check on the twist.
"""

import itertools
import math
import unittest

import numpy as np

import tessera

cobordism = tessera.cobordism
DijkgraafWitten = cobordism.DijkgraafWitten
Cocycle = cobordism.Cocycle


# --------------------------------------------------------------------------- #
# Fixtures: closed oriented 3-manifolds small enough for the brute-force sum.
# --------------------------------------------------------------------------- #
def _build(topology, dimensions=None):
    if dimensions is None:
        dimensions = topology.dimension()
    signature = tessera.Signature(dimensions, tessera.Lorentzian)
    metric = tessera.Metric(True, signature)
    spacetime = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                                  tessera.PREFERRED, topology)
    spacetime.build()
    return spacetime


def _three_sphere():
    # S³ = ∂Δ⁴: 5 vertices, betti (1,0,0,1).
    return _build(tessera.SimplexBoundarySphere(3))


def _s2_cross_s1():
    # S²×S¹ via the simplicial (Eilenberg–Zilber) product: 12 vertices,
    # betti (1,1,1,1). The named fixture is a separate ticket; compose it inline.
    return _build(tessera.SimplicialProduct(tessera.SimplexBoundarySphere(2),
                                            tessera.SimplexBoundarySphere(1)))


def _two_sphere():
    # S²: a closed 2-manifold, used to check the dimension guard.
    return _build(tessera.SimplexBoundarySphere(2))


def _first_betti_gf2(spacetime):
    chain = cobordism.ChainComplex.fromSpacetime(spacetime)
    return chain.bettiNumbersGF2()[1]


# --------------------------------------------------------------------------- #
# Pure-Python 3-cocycle oracle (group cohomology of ℤ₂, multiplicative U(1)).
# --------------------------------------------------------------------------- #
def _omega(kind, a, b, c):
    """ω(a,b,c) for a cochain ``kind`` ∈ {trivial, sign, broken}."""
    if kind == "trivial":
        return 1
    if kind == "sign":  # the nontrivial generator (-1)^{abc}
        return -1 if (a & b & c) else 1
    if kind == "broken":  # NOT a 3-cocycle — only here to give the oracle teeth
        return -1 if (a & b) else 1
    raise ValueError(kind)


def _satisfies_pentagon(kind):
    """Normalized 3-cocycle (pentagon) identity over ℤ₂, brute-forced over ℤ₂⁴:
    ω(b,c,d)·ω(a,b⊕c,d)·ω(a,b,c) == ω(a⊕b,c,d)·ω(a,b,c⊕d)."""
    for a, b, c, d in itertools.product((0, 1), repeat=4):
        left = _omega(kind, b, c, d) * _omega(kind, a, b ^ c, d) * _omega(kind, a, b, c)
        right = _omega(kind, a ^ b, c, d) * _omega(kind, a, b, c ^ d)
        if left != right:
            return False
    return True


# --------------------------------------------------------------------------- #
# Independent numpy reimplementation of the whole state sum (oracle for C++).
# --------------------------------------------------------------------------- #
def _gf2_nullspace(matrix):
    """Basis of {x : matrix·x ≡ 0 (mod 2)} as rows, via RREF (pure numpy)."""
    a = (np.asarray(matrix, dtype=np.int64) & 1).copy()
    rows, cols = a.shape
    pivots, r = [], 0
    for col in range(cols):
        if r >= rows:
            break
        piv = next((i for i in range(r, rows) if a[i, col] & 1), None)
        if piv is None:
            continue
        a[[r, piv]] = a[[piv, r]]
        for i in range(rows):
            if i != r and (a[i, col] & 1):
                a[i] ^= a[r]
        pivots.append(col)
        r += 1
    is_pivot = [False] * cols
    for c in pivots:
        is_pivot[c] = True
    basis = []
    for free in range(cols):
        if is_pivot[free]:
            continue
        x = np.zeros(cols, dtype=np.int64)
        x[free] = 1
        for t, pc in enumerate(pivots):
            x[pc] = a[t, free] & 1
        basis.append(x)
    return basis


def _dw_partition_oracle(spacetime, kind):
    """Recompute Z(W) independently from the chain complex (numpy GF(2))."""
    chain = cobordism.ChainComplex.fromSpacetime(spacetime)
    num_vertices = chain.numSimplices(0)
    num_edges = chain.numSimplices(1)
    num_triangles = chain.numSimplices(2)

    boundary2 = (np.asarray(chain.boundaryMatrix(2), dtype=np.int64)
                 .reshape(num_edges, num_triangles)) & 1   # edges × triangles
    coboundary1 = boundary2.T                              # triangles × edges
    basis = _gf2_nullspace(coboundary1)

    edges = [tuple(e) for e in chain.kSimplexVertices(1)]
    edge_index = {e: i for i, e in enumerate(edges)}
    tets = [tuple(t) for t in chain.orientedTopSimplices()]
    epsilon = list(chain.fundamentalClass())

    total = 0.0
    k = len(basis)
    for mask in range(1 << k):
        g = np.zeros(num_edges, dtype=np.int64)
        for b in range(k):
            if (mask >> b) & 1:
                g ^= basis[b]
        weight = 1
        for tet, sign in zip(tets, epsilon):
            g01 = int(g[edge_index[(tet[0], tet[1])]])
            g12 = int(g[edge_index[(tet[1], tet[2])]])
            g23 = int(g[edge_index[(tet[2], tet[3])]])
            value = _omega(kind, g01, g12, g23)
            # value**sign is a no-op for the real ±1 values of these cocycles.
            weight *= value
        total += weight
    return total / (2.0 ** num_vertices)


class TestThreeCocycleIdentity(unittest.TestCase):
    """Run first: ω must satisfy the pentagon identity (topological-invariance
    prerequisite). Establish the oracle's validity, then check C++ against it."""

    def test_python_oracle_accepts_the_two_cocycles(self):
        self.assertTrue(_satisfies_pentagon("trivial"))
        self.assertTrue(_satisfies_pentagon("sign"))

    def test_python_oracle_rejects_a_non_cocycle(self):
        # Teeth: a cochain that is not a 3-cocycle must fail the pentagon check,
        # otherwise the test above would be vacuous.
        self.assertFalse(_satisfies_pentagon("broken"))

    def test_cpp_iscocycle_matches_oracle(self):
        self.assertEqual(DijkgraafWitten.isCocycle(Cocycle.Trivial),
                         _satisfies_pentagon("trivial"))
        self.assertEqual(DijkgraafWitten.isCocycle(Cocycle.Sign),
                         _satisfies_pentagon("sign"))
        self.assertTrue(DijkgraafWitten.isCocycle(Cocycle.Trivial))
        self.assertTrue(DijkgraafWitten.isCocycle(Cocycle.Sign))


class TestPartitionFunctionConventionAnchor(unittest.TestCase):
    """Z_Trivial(W) = 2^{b₁(W;ℤ₂) - 1} for a connected closed oriented W."""

    def _assert_real(self, z, expected):
        self.assertAlmostEqual(z.imag, 0.0, places=9)
        self.assertAlmostEqual(z.real, expected, places=9)

    def test_three_sphere(self):
        spacetime = _three_sphere()
        b1 = _first_betti_gf2(spacetime)
        self.assertEqual(b1, 0)
        z = DijkgraafWitten(spacetime, Cocycle.Trivial).partitionFunction()
        self._assert_real(z, 2.0 ** (b1 - 1))   # ½
        self._assert_real(z, 0.5)

    def test_s2_cross_s1(self):
        spacetime = _s2_cross_s1()
        b1 = _first_betti_gf2(spacetime)
        self.assertEqual(b1, 1)
        z = DijkgraafWitten(spacetime, Cocycle.Trivial).partitionFunction()
        self._assert_real(z, 2.0 ** (b1 - 1))   # 1
        self._assert_real(z, 1.0)


class TestPartitionFunctionAgainstOracle(unittest.TestCase):
    """The C++ state sum equals an independent numpy reimplementation — a direct
    check on the edge-index map, the ∂₂ transpose, and the orientation signs."""

    def _check(self, spacetime):
        for cocycle, kind in ((Cocycle.Trivial, "trivial"), (Cocycle.Sign, "sign")):
            with self.subTest(cocycle=kind):
                z = DijkgraafWitten(spacetime, cocycle).partitionFunction()
                self.assertAlmostEqual(z.imag, 0.0, places=9)
                self.assertAlmostEqual(z.real, _dw_partition_oracle(spacetime, kind),
                                       places=9)

    def test_three_sphere(self):
        self._check(_three_sphere())

    def test_s2_cross_s1(self):
        self._check(_s2_cross_s1())


class TestSignCocycleNegativeControls(unittest.TestCase):
    """On manifolds with vanishing mod-2 cup cube the sign cocycle does not
    distinguish: Z_Sign == Z_Trivial. (S³ and S²×S¹; T³ would belong here too.)"""

    def _assert_equal_invariants(self, spacetime):
        z_trivial = DijkgraafWitten(spacetime, Cocycle.Trivial).partitionFunction()
        z_sign = DijkgraafWitten(spacetime, Cocycle.Sign).partitionFunction()
        self.assertAlmostEqual(abs(z_sign - z_trivial), 0.0, places=9)

    def test_three_sphere_not_distinguished(self):
        self._assert_equal_invariants(_three_sphere())

    def test_s2_cross_s1_not_distinguished(self):
        self._assert_equal_invariants(_s2_cross_s1())


class TestSignCocyclePositiveControl(unittest.TestCase):
    """The positive control (a manifold the sign cocycle *does* distinguish)."""

    def test_lens_space_distinction_is_deferred(self):
        # The sign cocycle distinguishes W iff the mod-2 cup cube t³ ≠ 0 on W —
        # a lens space such as RP³, NOT T³ (whose cup cube vanishes; see module
        # docstring). The brute-force state sum (gf2Span over the whole flat
        # space, nullity = |V| ≲ 24) admits only the vertex-minimal RP³, and no
        # RP³/lens-space fixture exists yet, so this control is deferred.
        #
        # Pinned target for the follow-up fixture (b₁(RP³;ℤ₂)=1, t³≠0):
        #   Z_Trivial(RP³) = 2^{1-1}                       = 1
        #   Z_Sign(RP³)    = ½·(1 + (-1)^{⟨t³,[RP³]⟩})·... = 0   (Z_Sign ≠ Z_Trivial)
        z_trivial_rp3 = 2.0 ** (1 - 1)
        z_sign_rp3 = 0.5 * (1 + (-1) ** 1)  # gauge-reduced: ½(W(0) + W(t))
        self.assertEqual(z_trivial_rp3, 1.0)
        self.assertEqual(z_sign_rp3, 0.0)
        self.assertNotEqual(z_sign_rp3, z_trivial_rp3)
        self.skipTest("RP³/lens-space fixture not yet available; see module "
                      "docstring (ticket #108's T³ example is incorrect — the "
                      "cup cube vanishes on T³).")


class TestDimensionGuard(unittest.TestCase):
    """The state sum is defined for closed oriented 3-manifolds only."""

    def test_two_sphere_raises(self):
        spacetime = _two_sphere()  # dimension 2
        with self.assertRaises(RuntimeError):
            DijkgraafWitten(spacetime, Cocycle.Trivial).partitionFunction()


if __name__ == "__main__":
    unittest.main()
