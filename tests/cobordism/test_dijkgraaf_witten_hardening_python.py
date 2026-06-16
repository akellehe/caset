# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Dijkgraaf–Witten ℤ₂ state-sum hardening (#153).

A property-test pass over the closed (#108) and bounded (#109) Dijkgraaf–Witten
state sums, layering an independent numpy oracle and a few structural identities
on top of the capability tests in ``test_dijkgraaf_witten*`` and
``test_rp3_fixture``. Acceptance, in dependency order:

1. **3-cocycle (pentagon) identity, both classes.** ω must be a genuine
   3-cocycle or the state sum is not gauge-invariant. ``isCocycle`` is checked
   against a numpy pentagon oracle that has teeth (a deliberately non-cocycle
   cochain must fail it).

2. **Convention anchor.** ``Z_Trivial(W) = 2^{b₁(W;ℤ₂) − 1}`` for a connected
   closed oriented 3-manifold, swept across S³, S²×S¹, and RP³ with b₁ read from
   the chain complex.

3. **The Sign twist.** ``Z_Sign`` differs from ``Z_Trivial`` iff the mod-2 cup
   cube is nonzero on W. The positive control is RP³ (Z_Sign = 0 ≠ 1); the
   negatives are S³ and S²×S¹ (Z_Sign = Z_Trivial). T³ is the third negative,
   but its flat space (dim Z¹ = 29) overflows the brute-force closed sum
   (gf2Span refuses a nullity > 24), so — as the existing tests do — it is
   distinguished from RP³ by its torsion-free H₁ (RP³ carries the 2-torsion that
   supports t³ ≠ 0; T³ does not). The closed-sum guard is asserted to fire.

4. **Independent numpy state-sum oracle.** ``Z(W)`` is recomputed from the chain
   complex through a separate numpy path (GF(2) nullspace of ∂₂ᵀ, the flat
   span, the ω product, the gauge volume) and matched against the C++ value for
   both cocycles on S³, S²×S¹, and RP³.

5. **Boundary map.** The cylinder Σ×[0,T] is the identity on Z(Σ), and its
   categorical trace closes it to the mapping torus Σ×S¹, so
   ``Tr(map(Σ×I)) = Z(Σ×S¹)``: checked numerically on S² (closed sum
   computable) and against the convention formula on T² (T³ closed sum too
   large).
"""

import itertools
import unittest

import numpy as np

import tessera

cobordism = tessera.cobordism
DijkgraafWitten = cobordism.DijkgraafWitten
Cocycle = cobordism.Cocycle


# --------------------------------------------------------------------------- #
# Fixtures.
# --------------------------------------------------------------------------- #
def _build(topology):
    signature = tessera.Signature(topology.dimension(), tessera.Lorentzian)
    metric = tessera.Metric(True, signature)
    spacetime = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                                  tessera.PREFERRED, topology)
    spacetime.build()
    return spacetime


def _circle():
    return tessera.SimplexBoundarySphere(1)            # S¹ = ∂Δ²


def _interval():
    return tessera.SolidSimplex(1)                     # [0, 1]


def _three_sphere():
    return _build(tessera.SimplexBoundarySphere(3))    # S³, b₁=0


def _s2_cross_s1():
    # S²×S¹ via the simplicial product (dim Z¹ = 12 ≤ 24, so brute-forceable).
    return _build(tessera.SimplicialProduct(tessera.SimplexBoundarySphere(2),
                                            _circle()))


def _rp3():
    return _build(tessera.RealProjectiveSpace())       # RP³, t³ ≠ 0


def _three_torus_topology():
    # T³ = S¹×S¹×S¹; dim Z¹ = 29, too large for the closed brute-force sum.
    return tessera.SimplicialProduct(
        tessera.SimplicialProduct(_circle(), _circle()), _circle())


def _sphere_cylinder():
    # S²×[0,T]: trivial cobordism S² → S² (Z(S²) is one-dimensional).
    return _build(tessera.SimplicialProduct(tessera.SimplexBoundarySphere(2),
                                            _interval()))


def _torus_cylinder():
    # T²×[0,T]: trivial cobordism T² → T² (Z(T²) is four-dimensional).
    torus = tessera.SimplicialProduct(_circle(), _circle())
    return _build(tessera.SimplicialProduct(torus, _interval()))


def _chain(spacetime):
    return cobordism.ChainComplex.fromSpacetime(spacetime)


def _first_betti_gf2(spacetime):
    return _chain(spacetime).bettiNumbersGF2()[1]


# --------------------------------------------------------------------------- #
# Independent numpy reimplementation (oracle for the C++ state sum).
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


def _omega(kind, a, b, c):
    """ω(a,b,c) for a cochain ``kind`` ∈ {trivial, sign, broken}."""
    if kind == "trivial":
        return 1
    if kind == "sign":               # the nontrivial generator (-1)^{abc}
        return -1 if (a & b & c) else 1
    if kind == "broken":             # NOT a 3-cocycle — gives the oracle teeth
        return -1 if (a & b) else 1
    raise ValueError(kind)


def _satisfies_pentagon(kind):
    """Normalized 3-cocycle (pentagon) identity over ℤ₂, brute-forced over ℤ₂⁴."""
    for a, b, c, d in itertools.product((0, 1), repeat=4):
        left = _omega(kind, b, c, d) * _omega(kind, a, b ^ c, d) * _omega(kind, a, b, c)
        right = _omega(kind, a ^ b, c, d) * _omega(kind, a, b, c ^ d)
        if left != right:
            return False
    return True


def _dw_partition_oracle(spacetime, kind):
    """Recompute Z(W) independently from the chain complex (numpy GF(2))."""
    chain = _chain(spacetime)
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

    total = 0.0
    k = len(basis)
    for mask in range(1 << k):
        g = np.zeros(num_edges, dtype=np.int64)
        for b in range(k):
            if (mask >> b) & 1:
                g ^= basis[b]
        weight = 1
        for tet in tets:
            g01 = int(g[edge_index[(tet[0], tet[1])]])
            g12 = int(g[edge_index[(tet[1], tet[2])]])
            g23 = int(g[edge_index[(tet[2], tet[3])]])
            # value**ε_t is a no-op for the real ±1 values of these cocycles.
            weight *= _omega(kind, g01, g12, g23)
        total += weight
    return total / (2.0 ** num_vertices)


# --------------------------------------------------------------------------- #
# 1. 3-cocycle (pentagon) identity — both classes.
# --------------------------------------------------------------------------- #
class TestThreeCocycleIdentity(unittest.TestCase):

    def test_numpy_pentagon_oracle_has_teeth(self):
        # The oracle accepts the two genuine cocycles and rejects a non-cocycle,
        # so the C++ cross-check below is not vacuous.
        self.assertTrue(_satisfies_pentagon("trivial"))
        self.assertTrue(_satisfies_pentagon("sign"))
        self.assertFalse(_satisfies_pentagon("broken"))

    def test_cpp_iscocycle_matches_oracle(self):
        self.assertEqual(DijkgraafWitten.isCocycle(Cocycle.Trivial),
                         _satisfies_pentagon("trivial"))
        self.assertEqual(DijkgraafWitten.isCocycle(Cocycle.Sign),
                         _satisfies_pentagon("sign"))
        self.assertTrue(DijkgraafWitten.isCocycle(Cocycle.Trivial))
        self.assertTrue(DijkgraafWitten.isCocycle(Cocycle.Sign))


# --------------------------------------------------------------------------- #
# 2. Convention anchor: Z_Trivial(W) = 2^{b₁(W;ℤ₂) − 1} across all small closed
#    fixtures (S³, S²×S¹, RP³).
# --------------------------------------------------------------------------- #
class TestTrivialConventionAnchor(unittest.TestCase):

    CASES = (("S^3", _three_sphere, 0, 0.5),
             ("S^2xS^1", _s2_cross_s1, 1, 1.0),
             ("RP^3", _rp3, 1, 1.0))

    def test_trivial_partition_function_matches_formula(self):
        for name, factory, expected_b1, expected_z in self.CASES:
            with self.subTest(manifold=name):
                spacetime = factory()
                b1 = _first_betti_gf2(spacetime)
                self.assertEqual(b1, expected_b1)
                z = DijkgraafWitten(spacetime, Cocycle.Trivial).partitionFunction()
                self.assertAlmostEqual(z.imag, 0.0, places=9)
                self.assertAlmostEqual(z.real, 2.0 ** (b1 - 1), places=9)
                self.assertAlmostEqual(z.real, expected_z, places=9)


# --------------------------------------------------------------------------- #
# 3. The Sign twist distinguishes RP³ and not the negatives (T³, S²×S¹, S³).
# --------------------------------------------------------------------------- #
class TestSignCocycleDistinction(unittest.TestCase):

    @staticmethod
    def _z(spacetime, cocycle):
        return DijkgraafWitten(spacetime, cocycle).partitionFunction()

    def test_sign_distinguishes_rp3(self):
        # Positive control: t³ ≠ 0 on RP³ ⇒ Z_Sign = 0 ≠ 1 = Z_Trivial.
        spacetime = _rp3()
        z_trivial = self._z(spacetime, Cocycle.Trivial)
        z_sign = self._z(spacetime, Cocycle.Sign)
        self.assertAlmostEqual(z_trivial.real, 1.0, places=9)
        self.assertAlmostEqual(z_sign.real, 0.0, places=9)
        self.assertGreater(abs(z_trivial - z_sign), 0.5)

    def test_sign_does_not_distinguish_s3_or_s2s1(self):
        # Negatives with a computable closed sum: the cup cube vanishes, so the
        # Sign twist agrees with the trivial value.
        for name, factory in (("S^3", _three_sphere), ("S^2xS^1", _s2_cross_s1)):
            with self.subTest(manifold=name):
                spacetime = factory()
                z_trivial = self._z(spacetime, Cocycle.Trivial)
                z_sign = self._z(spacetime, Cocycle.Sign)
                self.assertAlmostEqual(abs(z_trivial - z_sign), 0.0, places=9)

    def test_three_torus_negative_control_via_torsion(self):
        # T³'s flat space is too large for the brute-force closed sum, so it is
        # separated from the RP³ positive control by its torsion-free H₁: RP³
        # carries the 2-torsion (Bockstein) that supports a nonzero cup cube,
        # T³ (like S²×S¹) does not.
        t3 = _chain(_build(_three_torus_topology()))
        self.assertEqual(t3.torsion(1), [])               # H₁(T³) = ℤ³, no torsion
        self.assertEqual(t3.torsion(2), [])
        self.assertEqual(_chain(_rp3()).torsion(1), [2])  # H₁(RP³) = ℤ₂
        self.assertEqual(_chain(_s2_cross_s1()).torsion(1), [])

    def test_three_torus_closed_sum_guard_fires(self):
        # dim Z¹ = b₁(ℤ₂) + |V| − b₀(ℤ₂) = 3 + 27 − 1 = 29 > 24, so gf2Span
        # refuses to materialize the flat space — the reason T³ uses the torsion
        # proxy above rather than a direct Z_Sign computation.
        t3 = _chain(_build(_three_torus_topology()))
        betti_gf2 = t3.bettiNumbersGF2()
        dim_z1 = betti_gf2[1] + t3.numSimplices(0) - betti_gf2[0]
        self.assertEqual(t3.numSimplices(0), 27)
        self.assertEqual(dim_z1, 29)
        self.assertGreater(dim_z1, 24)
        with self.assertRaises((ValueError, RuntimeError)):
            DijkgraafWitten(_build(_three_torus_topology()),
                            Cocycle.Sign).partitionFunction()


# --------------------------------------------------------------------------- #
# 4. Independent numpy state-sum oracle for the closed partition function.
# --------------------------------------------------------------------------- #
class TestPartitionFunctionAgainstOracle(unittest.TestCase):

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

    def test_rp3(self):
        # RP³ is the discriminating case the existing oracle test omits.
        self._check(_rp3())


# --------------------------------------------------------------------------- #
# 5. Boundary map: cylinder = identity, and Tr(map) = Z(mapping torus).
# --------------------------------------------------------------------------- #
class TestCylinderIsIdentity(unittest.TestCase):

    def test_sphere_cylinder_is_identity(self):
        for cocycle in (Cocycle.Trivial, Cocycle.Sign):
            with self.subTest(cocycle=cocycle):
                matrix = np.asarray(DijkgraafWitten(_sphere_cylinder(), cocycle).map())
                self.assertEqual(matrix.shape, (1, 1))     # Z(S²) is 1-dimensional
                np.testing.assert_allclose(matrix, np.eye(1), atol=1e-9)

    def test_torus_cylinder_is_identity(self):
        for cocycle in (Cocycle.Trivial, Cocycle.Sign):
            with self.subTest(cocycle=cocycle):
                matrix = np.asarray(DijkgraafWitten(_torus_cylinder(), cocycle).map())
                self.assertEqual(matrix.shape, (4, 4))     # 2^{b₁(T²)} = 4
                np.testing.assert_allclose(matrix, np.eye(4), atol=1e-9)


class TestTraceEqualsClosedPartitionFunction(unittest.TestCase):
    """Closing the cylinder Σ×[0,T] by identifying its two ends is the mapping
    torus Σ×S¹; the categorical trace of the boundary map realizes that gluing,
    so Tr(map(Σ×I)) = Z(Σ×S¹). For the identity map the trace is dim Z(Σ) =
    2^{b₁(Σ)}, matching Z(Σ×S¹) = 2^{b₁(Σ×S¹;ℤ₂) − 1} = 2^{b₁(Σ)}."""

    @staticmethod
    def _trace(matrix):
        m = np.asarray(matrix)
        return complex(np.trace(m))

    def test_sphere_trace_equals_three_manifold_partition_function(self):
        # Σ = S²: both sides computable. Tr(map(S²×I)) = Z(S²×S¹) = 1.
        for cocycle in (Cocycle.Trivial, Cocycle.Sign):
            with self.subTest(cocycle=cocycle):
                trace = self._trace(DijkgraafWitten(_sphere_cylinder(), cocycle).map())
                z_closed = DijkgraafWitten(_s2_cross_s1(), cocycle).partitionFunction()
                self.assertAlmostEqual(trace.imag, 0.0, places=9)
                self.assertAlmostEqual(trace.real, 1.0, places=9)
                self.assertAlmostEqual(trace.real, z_closed.real, places=9)

    def test_torus_trace_matches_three_torus_formula(self):
        # Σ = T²: Tr(map(T²×I)) = 2^{b₁(T²)} = 4. The closed sum Z(T³) overflows
        # the brute-force enumeration, so it is cross-checked against the
        # convention value 2^{b₁(T³;ℤ₂) − 1}, with b₁ read from the T³ complex.
        trace = self._trace(DijkgraafWitten(_torus_cylinder(), Cocycle.Trivial).map())
        b1_t3 = _first_betti_gf2(_build(_three_torus_topology()))
        self.assertEqual(b1_t3, 3)
        self.assertAlmostEqual(trace.imag, 0.0, places=9)
        self.assertAlmostEqual(trace.real, 4.0, places=9)
        self.assertAlmostEqual(trace.real, 2.0 ** (b1_t3 - 1), places=9)


if __name__ == "__main__":
    unittest.main()
