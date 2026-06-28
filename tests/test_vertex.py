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

    def test_star_collects_closed_star_vertex_ids(self):
        # Two triangles sharing edge (0,1): 0 is in both, 3 is only in the
        # second.  The closed star of vertex 0 is {0,1,2,3}; the closed star of
        # the corner vertex 2 is just its own triangle {0,1,2}.
        st = Spacetime.fromCells(2, [[0, 1, 2], [0, 1, 3]], 1.0, 0.0)
        verts = st.getVertexList()

        self.assertEqual(verts.get(0).star(), {0, 1, 2, 3})
        self.assertEqual(verts.get(2).star(), {0, 1, 2})

    def test_star_includes_self_for_isolated_simplex(self):
        # A single triangle: every vertex's closed star is the whole triangle,
        # and always contains the vertex itself.
        st = Spacetime.fromCells(2, [[5, 6, 7]], 1.0, 0.0)
        verts = st.getVertexList()
        for vertex_id in (5, 6, 7):
            star = verts.get(vertex_id).star()
            self.assertIn(vertex_id, star)
            self.assertEqual(star, {5, 6, 7})

