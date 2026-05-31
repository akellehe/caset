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

"""The SimplicialProduct topology (staircase triangulation of a product).

A product of two manifolds is itself a manifold, and its homology is the
Kunneth combination of the factors'. We check several products against their
known homology (Betti numbers), Euler characteristic, the chain-complex axiom
(boundary of a boundary is zero), and that the result is a closed manifold
(every codimension-one face is shared by exactly two top cells). Products of
products are included to confirm the construction nests correctly.
"""

import itertools
import unittest

import tessera

cobordism = tessera.cobordism


def _circle():
    return tessera.SimplexBoundarySphere(1)  # S^1


def _two_sphere():
    return tessera.SimplexBoundarySphere(2)  # S^2


def _product(*factors):
    """Left-nested SimplicialProduct of two or more factor topologies."""
    result = factors[0]
    for factor in factors[1:]:
        result = tessera.SimplicialProduct(result, factor)
    return result


def _build(topology):
    signature = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, signature)
    spacetime = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                                  tessera.PREFERRED, topology)
    spacetime.build()
    return spacetime


def _chain_complex(topology):
    return cobordism.ChainComplex.fromSpacetime(_build(topology))


def _top_simplices(spacetime):
    tuples = [tuple(sorted(v.getId() for v in s.getVertices()))
              for s in spacetime.getSimplices()]
    top_size = max(len(t) for t in tuples)
    return [t for t in tuples if len(t) == top_size]


# (name, topology, expected Betti numbers). Euler characteristic follows from
# the alternating sum of the Betti numbers, so we don't list it separately.
PRODUCTS = [
    ("torus = S^1 x S^1", _product(_circle(), _circle()), [1, 2, 1]),
    ("S^1 x S^2", _product(_circle(), _two_sphere()), [1, 1, 1, 1]),
    ("S^2 x S^2", _product(_two_sphere(), _two_sphere()), [1, 0, 2, 0, 1]),
    ("3-torus = S^1 x S^1 x S^1",
     _product(_circle(), _circle(), _circle()), [1, 3, 3, 1]),
]


class TestSimplicialProductHomology(unittest.TestCase):

    def test_betti_numbers(self):
        for name, topology, expected in PRODUCTS:
            with self.subTest(product=name):
                self.assertEqual(_chain_complex(topology).bettiNumbers(), expected)

    def test_euler_characteristic_matches_betti_alternating_sum(self):
        for name, topology, expected_betti in PRODUCTS:
            with self.subTest(product=name):
                chain = _chain_complex(topology)
                expected_euler = sum((-1) ** k * b for k, b in enumerate(expected_betti))
                self.assertEqual(chain.eulerCharacteristic(), expected_euler)

    def test_boundary_of_boundary_is_zero(self):
        for name, topology, _ in PRODUCTS:
            with self.subTest(product=name):
                self.assertTrue(_chain_complex(topology).boundaryComposesToZero())

    def test_is_closed_manifold(self):
        # Every codimension-one face of a top cell is shared by exactly two top
        # cells (the defining property of a closed pseudomanifold).
        for name, topology, _ in PRODUCTS:
            with self.subTest(product=name):
                tops = _top_simplices(_build(topology))
                facet_counts = {}
                for top in tops:
                    for facet in itertools.combinations(top, len(top) - 1):
                        facet_counts[facet] = facet_counts.get(facet, 0) + 1
                self.assertTrue(all(count == 2 for count in facet_counts.values()),
                                f"{name} is not a closed manifold")

    def test_dimension_is_sum_of_factor_dimensions(self):
        # dim(K x L) = dim(K) + dim(L): S^1(1) x S^2(2) -> 3, etc.
        dim = cobordism.CombinatorialDimension()
        self.assertEqual(dim.compute(_build(_product(_circle(), _circle()))), 2.0)
        self.assertEqual(dim.compute(_build(_product(_circle(), _two_sphere()))), 3.0)
        self.assertEqual(dim.compute(_build(_product(_two_sphere(), _two_sphere()))), 4.0)


if __name__ == "__main__":
    unittest.main()
