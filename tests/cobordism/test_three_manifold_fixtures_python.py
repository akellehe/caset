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
    sig = tessera.Signature(4, tessera.Lorentzian)
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


if __name__ == "__main__":
    unittest.main()
