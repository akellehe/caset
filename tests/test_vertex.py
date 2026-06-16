# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

import unittest

from tessera import Vertex, Spacetime


class TestVertex(unittest.TestCase):

    def test_vertex_creation(self):
        v1 = Vertex(1, [1, 2, 3, 4])
        coords = v1.getCoordinates()
        self.assertEqual(coords, [1, 2, 3, 4])

    def test_remove_outedge(self):
        st = Spacetime()
        v1 = st.createVertex(1, [1, 2, 3, 4])
        v2 = st.createVertex(2, [5, 6, 7, 8])
        edge = st.createEdge(v1, v2)

        self.assertEqual(len(v1.getOutEdges()), 1)
        self.assertEqual([e for e in v1.getOutEdges()][0].getTarget().getId(), v2.getId())

        v1.removeOutEdge(edge)
        self.assertEqual(len(v1.getOutEdges()), 0)

