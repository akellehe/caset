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

from caset import Vertex, Spacetime


class TestVertex(unittest.TestCase):

    def test_vertex_creation(self):
        v1 = Vertex(1, [1, 2, 3, 4])
        coords = v1.getCoordinates()
        self.assertEqual(coords, [1, 2, 3, 4])

    def test_remove_outedge(self):
        st = Spacetime()
        v1 = st.createVertex(1, [1, 2, 3, 4])
        v2 = st.createVertex(2, [5, 6, 7, 8])
        edge = st.createEdge(v1.getId(), v2.getId())

        self.assertEqual(len(v1.getOutEdges()), 1)
        self.assertEqual([e for e in v1.getOutEdges()][0].getTargetId(), v2.getId())

        v1.removeOutEdge(edge)
        self.assertEqual(len(v1.getOutEdges()), 0)

    def test_move_in_edges_to(self):
        st = Spacetime()
        v1 = st.createVertex(1, [1, 2, 3, 4])
        v2 = st.createVertex(2, [5, 6, 7, 8])
        e12 = st.createEdge(v1.getId(), v2.getId())

        v3 = st.createVertex(3, [9, 10, 11, 12])
        v4 = st.createVertex(4, [13, 14, 15, 16])
        e34 = st.createEdge(v3.getId(), v4.getId())

        self.assertEqual(len(v2.getInEdges()), 1)
        self.assertEqual(len(v4.getInEdges()), 1)

        self.assertEqual(e12.getSourceId(), 1)
        self.assertEqual(e12.getTargetId(), 2)
        self.assertEqual(e34.getSourceId(), 3)
        self.assertEqual(e34.getTargetId(), 4)

        old, new = v2.moveInEdgesTo(v4, st.getEdgeList(), st.getVertexList())
        old = [o for o in old][0]
        new = [n for n in new][0]
        self.assertEqual(old.source, 1)
        self.assertEqual(old.target, 2)
        self.assertEqual(new.source, 1)
        self.assertEqual(new.target, 4)

        self.assertEqual(len(v2.getInEdges()), 0)
        self.assertEqual(len(v4.getInEdges()), 2)

        self.assertEqual(e12.getSourceId(), 1)
        self.assertEqual(e12.getTargetId(), 4)
        self.assertEqual(e34.getSourceId(), 3)
        self.assertEqual(e34.getTargetId(), 4)

    def test_move_out_edges_to(self):
        st = Spacetime()
        v1 = st.createVertex(1, [1, 2, 3, 4])
        v2 = st.createVertex(2, [5, 6, 7, 8])
        e12 = st.createEdge(v1.getId(), v2.getId())

        v3 = st.createVertex(3, [9, 10, 11, 12])
        v4 = st.createVertex(4, [13, 14, 15, 16])
        e34 = st.createEdge(v3.getId(), v4.getId())

        self.assertEqual(e12.getSourceId(), 1)
        self.assertEqual(e12.getTargetId(), 2)
        self.assertEqual(e34.getSourceId(), 3)
        self.assertEqual(e34.getTargetId(), 4)

        self.assertEqual(len(v2.getInEdges()), 1)
        self.assertEqual(len(v4.getInEdges()), 1)

        old, new = v1.moveOutEdgesTo(v3, st.getEdgeList(), st.getVertexList())
        old = [o for o in old][0]
        new = [n for n in new][0]
        self.assertEqual(old.source, 1)
        self.assertEqual(old.target, 2)
        self.assertEqual(new.source, 3)
        self.assertEqual(new.target, 2)

        self.assertEqual(len(v1.getOutEdges()), 0)
        self.assertEqual(len(v3.getOutEdges()), 2)

        self.assertEqual(e12.getSourceId(), 3)
        self.assertEqual(e12.getTargetId(), 2)
        self.assertEqual(e34.getSourceId(), 3)
        self.assertEqual(e34.getTargetId(), 4)

