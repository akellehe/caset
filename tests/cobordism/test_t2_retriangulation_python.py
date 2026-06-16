# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""T2 triangulation independence — phase 1 (#111).

The Dijkgraaf–Witten state sum Z(W) is a *topological* invariant: it must depend
only on the manifold W, never on the triangulation chosen to represent it. That
invariance under retriangulation (T2) is the make-or-break property of the whole
cobordism program, so it is de-risked here — directly, on two genuinely distinct
triangulations of one manifold — *before* the general Pachner-move engine (#112)
is built.

Phase 1 fixes the manifold to S²×S¹ and compares its two merged triangulations:

  * ``SphereCircleProduct()``                       — the ∂Δ³ × ∂Δ² product
    (12 vertices, 36 tetrahedra), and
  * ``StellarSubdivision(SphereCircleProduct())``   — one stellar (1→4) move on it
    (13 vertices, 39 tetrahedra).

T³ — the natural first choice — is deliberately *not* used: its 27 vertices give
``dim Z¹ = 29`` flat connections, beyond what the brute-force state sum can
enumerate (``gf2Span`` caps the materializable nullity at 24). S²×S¹ and its
subdivision are both well under that bound, and are already merged + tested
(#110 / #121).

Result (T2, phase 1). The two triangulations are **non-isomorphic** as labelled
complexes (``Cobordism.areIsomorphic`` is False) and the stellar move grows the
complex by a vertex and three tetrahedra — yet they share every Betti number
(b = (1, 1, 1, 1)) and produce **identical** partition functions, to machine
precision, for *both* admissible 3-cocycles:

    Z_Trivial(product) == Z_Trivial(subdivided) == 1
    Z_Sign(product)    == Z_Sign(subdivided)    == 1.

This is the phase-1 T2 statement: equal Z from inequivalent triangulations of the
same manifold.

S²×S¹ is the sign-cocycle *negative control* — its mod-2 cup cube vanishes, so
the sign twist does not distinguish it and Z_Sign == Z_Trivial here too (#108).
That is by design: T2 is about invariance under retriangulation, not about the
sign cocycle distinguishing manifolds, so a manifold on which both cocycles agree
is exactly the clean setting in which to isolate the retriangulation question. A
manifold the sign cocycle *does* distinguish (a lens space such as RP³, where the
cup cube t³ ≠ 0) is the subject of the deferred positive control in
``test_dijkgraaf_witten_python.py``.
"""

import unittest

import numpy as np

import tessera

cobordism = tessera.cobordism
DijkgraafWitten = cobordism.DijkgraafWitten
Cocycle = cobordism.Cocycle

# "Equal to machine precision": absolute tolerance only, no relative slack. The
# partition functions here are exact small dyadic rationals (Z = 1), so any
# genuine triangulation dependence would show up far above this floor.
TOLERANCE = 1e-12


def _build(topology):
    """Build a closed oriented 3-manifold Spacetime from a topology."""
    signature = tessera.Signature(topology.dimension(), tessera.Lorentzian)
    metric = tessera.Metric(True, signature)
    spacetime = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                                  tessera.PREFERRED, topology)
    spacetime.build()  # delegates to topology.build(); the count is ignored
    return spacetime


def _top_simplices(spacetime):
    """Top-cell vertex-id tuples (as plain-int lists) of a built complex.

    The Spacetime must stay alive for the duration of this call — the Simplex and
    Vertex handles from ``getSimplices()`` point into its storage — so callers
    pass a live local; only the decoupled int lists escape.
    """
    by_size = {}
    for simplex in spacetime.getSimplices():
        cell = tuple(sorted(vertex.getId() for vertex in simplex.getVertices()))
        by_size.setdefault(len(cell), []).append(cell)
    return [list(cell) for cell in by_size[max(by_size)]]


def _betti(spacetime):
    return cobordism.ChainComplex.fromSpacetime(spacetime).bettiNumbers()


class TestT2RetriangulationInvariance(unittest.TestCase):
    """Z(S²×S¹) is unchanged across two distinct triangulations — T2, phase 1."""

    def setUp(self):
        # Build each triangulation once and keep both Spacetimes alive for the
        # whole test: the partition function and the top-cell list must describe
        # the *same* complex, and the handles from getSimplices() require the
        # Spacetime to outlive their use.
        self.product = _build(tessera.SphereCircleProduct())
        self.subdivided = _build(
            tessera.StellarSubdivision(tessera.SphereCircleProduct()))

    def _assert_machine_precision(self, actual, expected):
        """|actual - expected| <= 1e-12, with no relative tolerance."""
        self.assertTrue(
            np.isclose(actual, expected, rtol=0.0, atol=TOLERANCE),
            msg=f"{actual!r} != {expected!r} to within {TOLERANCE}")

    # -- the inputs are a genuine retriangulation of one manifold ----------- #
    def test_inputs_are_a_genuine_retriangulation(self):
        # Same manifold: identical homology ...
        self.assertEqual(_betti(self.product), [1, 1, 1, 1])
        self.assertEqual(_betti(self.subdivided), [1, 1, 1, 1])
        # ... yet a genuinely different triangulation: the 1→4 stellar move adds
        # one vertex and three tetrahedra, so neither can be a relabeling of the
        # other.
        product_tops = _top_simplices(self.product)
        subdivided_tops = _top_simplices(self.subdivided)
        self.assertEqual(len(subdivided_tops), len(product_tops) + 3)
        self.assertFalse(
            cobordism.Cobordism.areIsomorphic(product_tops, subdivided_tops))
        # Reflexivity guard, so the negative result above is not vacuous.
        self.assertTrue(
            cobordism.Cobordism.areIsomorphic(product_tops, product_tops))

    # -- the headline T2 claim: Z is triangulation independent -------------- #
    def _assert_invariant(self, cocycle, expected):
        z_product = DijkgraafWitten(self.product, cocycle).partitionFunction()
        z_subdivided = DijkgraafWitten(self.subdivided, cocycle).partitionFunction()
        # Invariant across the retriangulation, to machine precision ...
        self._assert_machine_precision(z_product, z_subdivided)
        # ... and equal to the convention anchor Z = 2^{b₁-1} = 1 (real), so a
        # shared degeneracy (e.g. both collapsing to 0) cannot pass vacuously.
        self._assert_machine_precision(z_product, expected)
        self._assert_machine_precision(z_subdivided, expected)

    def test_Z_trivial_invariant_across_retriangulation(self):
        self._assert_invariant(Cocycle.Trivial, 1.0 + 0.0j)

    def test_Z_sign_invariant_across_retriangulation(self):
        self._assert_invariant(Cocycle.Sign, 1.0 + 0.0j)

    # -- S²×S¹ is the sign-cocycle negative control ------------------------- #
    def test_sign_cocycle_does_not_distinguish_either_triangulation(self):
        # The mod-2 cup cube vanishes on S²×S¹, so the sign twist agrees with the
        # trivial one (Z_Sign == Z_Trivial) — on *both* triangulations. T2 is
        # about invariance, not distinguishing, so this is the intended setting.
        for spacetime in (self.product, self.subdivided):
            z_trivial = DijkgraafWitten(spacetime, Cocycle.Trivial).partitionFunction()
            z_sign = DijkgraafWitten(spacetime, Cocycle.Sign).partitionFunction()
            self._assert_machine_precision(z_sign, z_trivial)


if __name__ == "__main__":
    unittest.main()
