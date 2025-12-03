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

import unittest

from caset import Edge, Vertex, Spacetime


class TestEdge(unittest.TestCase):

    def test_edge_instantiates(self):
        v1 = Vertex(0, [0, 0, 0, 0])
        v2 = Vertex(1, [1, 1, 1, 1])
        edge = Edge(v1.getId(), v2.getId())

        self.assertIsInstance(edge, Edge)
        src = edge.getSourceId()
        tgt = edge.getTargetId()
        self.assertIs(src, v1.getId())
        self.assertIs(tgt, v2.getId())

    def test_sets_of_edges(self):
        edges = [Edge(i, i+1) for i in range(51)]
        self.assertEqual(len(set(edges)), len(edges))

        e1 = Edge(0, 1)
        e2 = Edge(0, 1)
        e3 = Edge(1, 0)
        edges = set()
        edges.add(e1)
        edges.add(e2)
        edges.add(e3)
        self.assertEqual(len(edges), 1)

    def test_maps_of_edges(self):
        edges = [Edge(i, i+1) for i in range(51)]
        edge_dict = {Edge(i, i+1): i for i in range(51)}
        for i, e in enumerate(edges):
            self.assertEqual(e, edges[i])
            self.assertEqual(edge_dict.get(e), i)

        self.assertEqual(len(set(edges)), len(edges))

    def test_equality(self):
        e1 = Edge(0, 1)
        e2 = Edge(1, 0)
        self.assertEqual(e1, e2)
        e3 = Edge(0, 1)
        self.assertEqual(e1, e3)
        e4 = Edge(1, 2)
        self.assertNotEqual(e1, e4)


if __name__ == '__main__':
    unittest.main()
