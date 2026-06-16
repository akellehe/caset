# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Stage-2 closed 3-manifold fixtures (#110).

Two deliverables, checked here with cobordism homology:

  * ``SphereCircleProduct`` — the closed oriented S^2 x S^1, the negative
    control for the triple cup product (b = (1, 1, 1, 1));
  * two genuinely distinct triangulations of T^3 — the staircase product
    S^1 x S^1 x S^1 and its single stellar subdivision (``StellarSubdivision``).
    They are **non-isomorphic** as labelled complexes yet **homologically
    equal** (b = (1, 3, 3, 1)), the input pair for the T2 retriangulation-
    invariance check.
"""

import itertools
import unittest

import tessera

cobordism = tessera.cobordism


def _build(topology):
    sig = tessera.Signature(topology.dimension(), tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, topology)
    st.build()  # delegates to topology.build(); numSimplices ignored
    return st


def _betti(topology):
    return cobordism.ChainComplex.fromSpacetime(_build(topology)).bettiNumbers()


def _top_simplices(topology):
    """Top-simplex vertex-id tuples of the built complex, as plain ints.

    The Spacetime is held in a local for the duration of the extraction (the
    Simplex handles from getSimplices() point into its storage); only the
    decoupled int tuples escape, so the result is safe to use afterwards.
    """
    st = _build(topology)
    by_size = {}
    for s in st.getSimplices():
        t = tuple(sorted(v.getId() for v in s.getVertices()))
        by_size.setdefault(len(t), []).append(t)
    return [list(t) for t in by_size[max(by_size)]]


def _circle():
    return tessera.SimplexBoundarySphere(1)  # S^1


def _t3_product():
    """T^3 = S^1 x S^1 x S^1 via the (left-nested) staircase product."""
    return tessera.SimplicialProduct(
        tessera.SimplicialProduct(_circle(), _circle()), _circle())


def _t3_subdivided():
    """A second, inequivalent triangulation of T^3: one stellar subdivision."""
    return tessera.StellarSubdivision(_t3_product())


def _is_closed_manifold(tops):
    """Every codimension-one face of a top cell is shared by exactly two top
    cells (the defining property of a closed pseudomanifold)."""
    facet_counts = {}
    for top in tops:
        for facet in itertools.combinations(sorted(top), len(top) - 1):
            facet_counts[facet] = facet_counts.get(facet, 0) + 1
    return all(count == 2 for count in facet_counts.values())


class TestSphereCircleProduct(unittest.TestCase):
    """S^2 x S^1 — the T^3 triple-cup negative control."""

    def test_betti_numbers(self):
        self.assertEqual(_betti(tessera.SphereCircleProduct()), [1, 1, 1, 1])

    def test_is_closed_three_manifold(self):
        tops = _top_simplices(tessera.SphereCircleProduct())
        self.assertTrue(all(len(t) == 4 for t in tops))   # tetrahedra => 3-manifold
        self.assertTrue(_is_closed_manifold(tops))

    def test_minimal_face_counts(self):
        # ∂Δ^3 (4 verts) x ∂Δ^2 (3 verts): 12 vertices, 4*3*C(3,1) = 36 tets.
        tops = _top_simplices(tessera.SphereCircleProduct())
        self.assertEqual(len({v for t in tops for v in t}), 12)
        self.assertEqual(len(tops), 36)


class TestT3Retriangulations(unittest.TestCase):
    """Two distinct triangulations of T^3 for the T2 invariance check."""

    def test_both_have_three_torus_homology(self):
        self.assertEqual(_betti(_t3_product()), [1, 3, 3, 1])
        self.assertEqual(_betti(_t3_subdivided()), [1, 3, 3, 1])

    def test_triangulations_are_non_isomorphic(self):
        product = _top_simplices(_t3_product())
        subdivided = _top_simplices(_t3_subdivided())
        self.assertFalse(cobordism.Cobordism.areIsomorphic(product, subdivided))

    def test_homologically_equal_but_combinatorially_distinct(self):
        # Same manifold (equal Betti) ...
        self.assertEqual(_betti(_t3_product()), _betti(_t3_subdivided()))
        # ... yet the stellar move grows the complex by one vertex and three
        # tetrahedra, so it cannot be a relabeling of the product.
        product = _top_simplices(_t3_product())
        subdivided = _top_simplices(_t3_subdivided())
        self.assertEqual(len({v for t in subdivided for v in t}),
                         len({v for t in product for v in t}) + 1)
        self.assertEqual(len(subdivided), len(product) + 3)

    def test_each_triangulation_self_isomorphic(self):
        # Sanity guard on the negative result above: areIsomorphic is reflexive.
        product = _top_simplices(_t3_product())
        self.assertTrue(cobordism.Cobordism.areIsomorphic(product, product))

    def test_subdivision_remains_a_closed_manifold(self):
        self.assertTrue(_is_closed_manifold(_top_simplices(_t3_subdivided())))


class TestStellarSubdivisionPreservesTopology(unittest.TestCase):
    """A stellar (1->d+1) Pachner move changes the triangulation but not the
    homotopy type. The T^3 case above checks this once; here it is exercised on
    a different base manifold and under iteration -- the depth axis of the T2
    Pachner-invariance sweep (#112)."""

    def test_preserves_sphere_circle_product_homology(self):
        base = tessera.SphereCircleProduct()
        sub = tessera.StellarSubdivision(base)
        self.assertEqual(_betti(sub), [1, 1, 1, 1])          # still S^2 x S^1
        self.assertTrue(_is_closed_manifold(_top_simplices(sub)))
        b, s = _top_simplices(base), _top_simplices(sub)
        # the 1->4 move: +1 vertex, +3 tetrahedra ...
        self.assertEqual(len(s), len(b) + 3)
        self.assertEqual(len({v for t in s for v in t}),
                         len({v for t in b for v in t}) + 1)
        # ... a genuine retriangulation, not a relabeling.
        self.assertFalse(cobordism.Cobordism.areIsomorphic(b, s))

    def test_iterated_subdivision_preserves_three_torus(self):
        once = _t3_subdivided()
        twice = tessera.StellarSubdivision(once)
        self.assertEqual(_betti(twice), [1, 3, 3, 1])        # still T^3
        self.assertTrue(_is_closed_manifold(_top_simplices(twice)))
        o, t = _top_simplices(once), _top_simplices(twice)
        self.assertEqual(len(t), len(o) + 3)                 # one further move
        self.assertFalse(cobordism.Cobordism.areIsomorphic(o, t))


class TestClosedThreeManifoldInvariants(unittest.TestCase):
    """Euler-Poincare and orientability -- the properties the Dijkgraaf-Witten
    state-sum (#108) relies on for these fixtures."""

    @staticmethod
    def _chain(topology):
        return cobordism.ChainComplex.fromSpacetime(_build(topology))

    def test_euler_characteristic_zero(self):
        # A closed odd-dimensional manifold has chi = 0, and Euler-Poincare ties
        # the face-count chi (from |C_k|) to the homological one (the alternating
        # Betti sum) -- an independent cross-check of each triangulation.
        for topo in (tessera.SphereCircleProduct(), _t3_product(), _t3_subdivided()):
            cc = self._chain(topo)
            self.assertEqual(cc.eulerCharacteristic(), 0)
            betti = cc.bettiNumbers()
            self.assertEqual(sum((-1) ** k * b for k, b in enumerate(betti)), 0)

    def test_orientable_with_unique_fundamental_class(self):
        # b_3 = 1 and a +/-1 coefficient on every top cell => closed, connected,
        # oriented: the precondition for the orientation signs eps_t in the DW
        # weight.
        for topo in (tessera.SphereCircleProduct(), _t3_product()):
            cc = self._chain(topo)
            fundamental = cc.fundamentalClass()
            tops = cc.orientedTopSimplices()
            self.assertEqual(cc.bettiNumbers()[3], 1)
            self.assertEqual(len(fundamental), len(tops))
            self.assertTrue(all(abs(coeff) == 1 for coeff in fundamental))


if __name__ == "__main__":
    unittest.main()
