# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Branch-free multiplicative C* Wilson holonomy tests (#882).

The microscopic connection datum is a nonzero complex link U on the
canonical min(vertex-id)->max(vertex-id) orientation. A Wilson read is the
ordered product of oriented links. These tests deliberately use non-unit
modulus values and never reconstruct an additive phase.
"""

import math
import unittest

import numpy as np

import tessera


def _cycle_host():
    return tessera.Spacetime.fromCellsWithFields(
        1, [[0, 1], [1, 2], [0, 2]],
        squaredLength=complex(-0.75, 1.5),
        canonicalLink=1 + 0j)


def _vertices(st):
    return {v.getId(): v for v in st.getVertexList().toVector()}


def _edge(st, a, b):
    for edge in st.getEdgeList().toVector():
        if {edge.getSource().getId(), edge.getTarget().getId()} == {a, b}:
            return edge
    raise KeyError((a, b))


def _set_links(st, links):
    for edge in st.getEdgeList().toVector():
        a, b = sorted((edge.getSource().getId(), edge.getTarget().getId()))
        edge.setCanonicalLink(complex(links[(a, b)]))


def _oracle(st, cycle):
    product = 1 + 0j
    for index, a in enumerate(cycle):
        b = cycle[(index + 1) % len(cycle)]
        product *= _edge(st, a, b).link(a, b)
    return product


def _evaluate(st, cycle):
    by_id = _vertices(st)
    return tessera.WilsonLoop(st).evaluateU1Connection(
        [by_id[vertex_id] for vertex_id in cycle])


class MultiplicativeHolonomyTest(unittest.TestCase):

    def test_value_is_the_direct_ordered_complex_product(self):
        st = _cycle_host()
        _set_links(st, {
            (0, 1): complex(1.2, -0.4),
            (1, 2): complex(-0.7, 1.1),
            (0, 2): complex(0.3, -1.6),
        })
        expected = _oracle(st, [0, 1, 2])
        result = _evaluate(st, [0, 1, 2])
        self.assertEqual(result.loopSize, 3)
        self.assertEqual(complex(result.value), expected)
        self.assertEqual(complex(result.connectionHolonomy), expected)
        self.assertEqual(complex(result.holonomy()), expected)

    def test_reverse_orientation_returns_the_inverse(self):
        st = _cycle_host()
        _set_links(st, {
            (0, 1): complex(0.9, 0.8),
            (1, 2): complex(-1.3, 0.2),
            (0, 2): complex(0.6, -0.5),
        })
        forward = complex(_evaluate(st, [0, 1, 2]).value)
        reverse = complex(_evaluate(st, [0, 2, 1]).value)
        self.assertLess(abs(reverse - 1 / forward), 1e-13)

    def test_noncompact_modulus_is_preserved_not_normalized(self):
        st = _cycle_host()
        _set_links(st, {
            (0, 1): complex(2.0, 0.5),
            (1, 2): complex(0.75, -0.2),
            (0, 2): complex(-1.1, 0.3),
        })
        result = _evaluate(st, [0, 1, 2])
        expected = _oracle(st, [0, 1, 2])
        self.assertNotAlmostEqual(abs(expected), 1.0, places=6)
        self.assertAlmostEqual(result.holonomyModulus(), abs(expected),
                               places=13)

    def test_vertex_gauge_transform_leaves_closed_holonomy_invariant(self):
        st = _cycle_host()
        _set_links(st, {
            (0, 1): complex(1.1, 0.4),
            (1, 2): complex(-0.8, 1.3),
            (0, 2): complex(0.5, -0.9),
        })
        before = complex(_evaluate(st, [0, 1, 2]).value)
        gauges = {
            0: complex(0.6, -0.2),
            1: complex(-1.4, 0.7),
            2: complex(0.9, 1.1),
        }
        for edge in st.getEdgeList().toVector():
            a, b = sorted((edge.getSource().getId(),
                           edge.getTarget().getId()))
            edge.setCanonicalLink(
                (1 / gauges[a]) * edge.canonicalLink() * gauges[b])
        after = complex(_evaluate(st, [0, 1, 2]).value)
        self.assertLess(abs(after - before), 1e-13)

    def test_single_product_does_not_invent_a_logarithm_lift(self):
        st = _cycle_host()
        _set_links(st, {
            (0, 1): complex(-2.0, 1e-12),
            (1, 2): complex(-0.4, -0.7),
            (0, 2): complex(0.2, 1.5),
        })
        result = _evaluate(st, [0, 1, 2])
        self.assertEqual(complex(result.value), _oracle(st, [0, 1, 2]))
        self.assertEqual(result.windingNumber(), 0)
        # The old additive payload is intentionally absent, not silently zero.
        accumulation = complex(result.connectionAccumulation)
        self.assertTrue(math.isnan(accumulation.real))
        self.assertTrue(math.isnan(accumulation.imag))

    def test_hodge_connection_entries_use_the_same_direct_links(self):
        st = tessera.Spacetime.fromCellsWithFields(
            1, [[0, 1], [1, 2], [0, 2]],
            squaredLength=1 + 0j, canonicalLink=1 + 0j)
        _set_links(st, {
            (0, 1): complex(1.2, 0.1),
            (1, 2): complex(-0.6, 0.9),
            (0, 2): complex(0.75, -0.35),
        })
        ids = sorted(_vertices(st))
        row = {vertex_id: index for index, vertex_id in enumerate(ids)}
        side = len(ids)
        adjacency = np.asarray(
            tessera.cobordism.HodgeLaplacian(st).adjacency(),
            dtype=complex).reshape(side, side)
        product = 1 + 0j
        cycle = [0, 1, 2]
        for index, a in enumerate(cycle):
            b = cycle[(index + 1) % len(cycle)]
            product *= adjacency[row[a], row[b]]
        self.assertLess(abs(product - _oracle(st, cycle)), 1e-13)

    def test_open_or_degenerate_cycle_returns_an_empty_read(self):
        st = tessera.Spacetime.fromCellsWithFields(
            1, [[0, 1], [1, 2]], squaredLength=1 + 2j,
            canonicalLink=0.8 - 0.3j)
        by_id = _vertices(st)
        loop = tessera.WilsonLoop(st)
        self.assertEqual(loop.evaluateU1Connection([]).loopSize, 0)
        self.assertEqual(loop.evaluateU1Connection([by_id[0]]).loopSize, 0)
        self.assertEqual(
            loop.evaluateU1Connection([by_id[0], by_id[1], by_id[2]])
            .loopSize,
            0)


if __name__ == "__main__":
    unittest.main()
