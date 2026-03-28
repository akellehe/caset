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

from caset import Spacetime, Edge, Vertex

class TestSpacetime(unittest.TestCase):

    def test_create_vertex(self):
        st = Spacetime()
        v1 = st.createVertex(1)
        v2 = st.createVertex(2)

        self.assertEqual(v1.getId(), 1)
        self.assertEqual(v2.getId(), 2)

        self.assertNotEqual(v1, v2)

    def test_create_edge(self):
        st = Spacetime()
        v1 = st.createVertex(1)
        v2 = st.createVertex(2)
        v3 = st.createVertex(3)

        self.assertEqual(v1.getId(), 1)
        self.assertEqual(v2.getId(), 2)
        self.assertEqual(v3.getId(), 3)

        e1 = st.createEdge(v1, v2)
        e2 = st.createEdge(v2, v3)

        self.assertNotEqual(v1, v2)
        self.assertNotEqual(v2, v3)
        self.assertNotEqual(e1, e2)

        self.assertEqual(e1.getSource().getId(), v1.getId())
        self.assertEqual(e1.getTarget().getId(), v2.getId())
        self.assertEqual(e2.getSource().getId(), v2.getId())
        self.assertEqual(e2.getTarget().getId(), v3.getId())

    def test_create_simplex(self):
        st = Spacetime()
        simplex, _ = st.createSimplex((2, 3))
        self.assertEqual(len(simplex.getVertices()), 5)
        edges = simplex.getEdges()
        self.assertEqual(len(edges), 10)

        v1, v2, v3, v4, v5 = simplex.getVertices()
        a, b, c, d = v1.getOutEdges()
        self.assertEqual(a.getSource().getId(), v1.getId())
        self.assertEqual(b.getSource().getId(), v1.getId())
        self.assertEqual(c.getSource().getId(), v1.getId())
        self.assertEqual(d.getSource().getId(), v1.getId())
        self.assertEqual(len(v1.getEdges()), 4)

        a, b, c = v2.getOutEdges()
        self.assertEqual(a.getSource().getId(), v2.getId())
        self.assertEqual(b.getSource().getId(), v2.getId())
        self.assertEqual(c.getSource().getId(), v2.getId())
        self.assertEqual(len(v2.getEdges()), 4)
        a, = v2.getInEdges()
        self.assertEqual(a.getTarget().getId(), v2.getId())

        a, b = v3.getOutEdges()
        c, d = v3.getInEdges()
        self.assertEqual(a.getSource().getId(), v3.getId())
        self.assertEqual(b.getSource().getId(), v3.getId())
        self.assertEqual(c.getTarget().getId(), v3.getId())
        self.assertEqual(d.getTarget().getId(), v3.getId())
        self.assertEqual(len(v3.getEdges()), 4)

        a, = v4.getOutEdges()
        b, c, d = v4.getInEdges()
        self.assertEqual(a.getSource().getId(), v4.getId())
        self.assertEqual(b.getTarget().getId(), v4.getId())
        self.assertEqual(c.getTarget().getId(), v4.getId())
        self.assertEqual(d.getTarget().getId(), v4.getId())
        self.assertEqual(len(v4.getEdges()), 4)

        a, b, c, d = v5.getInEdges()
        self.assertEqual(a.getTarget().getId(), v5.getId())
        self.assertEqual(b.getTarget().getId(), v5.getId())
        self.assertEqual(c.getTarget().getId(), v5.getId())
        self.assertEqual(d.getTarget().getId(), v5.getId())
        self.assertEqual(len(v5.getEdges()), 4)

        self.assertEqual(len(st.getSimplicesWithOrientation((2, 3))), 1)
        self.assertEqual(len(st.getSimplicesWithOrientation((1, 1))), 0)

        simplex2, _ = st.createSimplex((2, 3))
        self.assertEqual(len(st.getSimplicesWithOrientation((2, 3))), 2)
        self.assertEqual(len(st.getSimplicesWithOrientation((1, 1))), 0)

    @unittest.skip
    def test_euclidean_embedding(self):
        st = Spacetime()
        simplex14, _ = st.createSimplex((1, 4))
        simplex23, _ = st.createSimplex((2, 3))
        st.embedEuclidean()
        vertices = st.getVertexList().toVector()

    def test_attaching_faces4D(self):
        st = Spacetime()

        firstVertexList = st.getVertexList()
        firstEdgeList = st.getEdgeList()

        simplex14, _ = st.createSimplex((1, 4))
        simplex23, _ = st.createSimplex((2, 3))

        self.assertEqual(len(simplex14.getVertices()), 5)
        self.assertEqual(len(simplex14.getEdges()), 10)

        self.assertEqual(len(simplex23.getVertices()), 5)
        self.assertEqual(len(simplex23.getEdges()), 10)

        allVertices = [v.getId() for v in simplex14.getVertices() + simplex23.getVertices()]
        self.assertEqual(len(allVertices), len(set(allVertices)))

        allEdges = [(e.getSource().getId(), e.getTarget().getId()) for e in [_ for _ in simplex14.getEdges()] + [_ for _ in simplex23.getEdges()]]
        self.assertEqual(len(simplex14.getEdges()), len(set(simplex14.getEdges())))
        self.assertEqual(len(simplex23.getEdges()), len(set(simplex23.getEdges())))

        edges23 = {(e.getSource().getId(), e.getTarget().getId()) for e in simplex23.getEdges()}
        edges14 = {(e.getSource().getId(), e.getTarget().getId()) for e in simplex14.getEdges()}

        self.assertTrue(edges23.isdisjoint(edges14))
        self.assertEqual(len(allEdges), len(set(allEdges)))

        totalVerticesBefore = len(st.getVertexList().toVector())
        self.assertEqual(totalVerticesBefore, 10)

        totalEdgesBefore = len(st.getEdgeList().toVector())
        self.assertEqual(totalEdgesBefore, 20)

        for edge in firstEdgeList.toVector():
            source = firstVertexList.get(edge.getSource().getId())
            target = firstVertexList.get(edge.getTarget().getId())
            self.assertIsNotNone(source)
            self.assertIsNotNone(target)

        left, right = None, None
        facets14 = simplex14.getFacets()
        ntime, nspace = 0, 0
        for facet in facets14:
            print(facet)
            self.assertTrue(facet.isInitialized())
            if facet.isSpatial():
                print("Timelike facet:", facet)
                self.assertTrue(facet.getOrientation().numeric()[0] == 0 or facet.getOrientation().numeric()[1] == 0)
                ntime += 1
                ti, tf = float('inf'), float('-inf')
                for v in facet.getVertices():
                    ti = min(ti, v.getTime())
                    tf = max(tf, v.getTime())
                self.assertEqual(ti, tf)
            else:
                print("Spacelike facet:", facet)
                for vertex in facet.getVertices():
                    print(vertex)
                self.assertNotEqual(facet.getOrientation().numeric()[0], 0)
                self.assertNotEqual(facet.getOrientation().numeric()[1], 0)
                nspace += 1
                ti, tf = float('inf'), float('-inf')
                for v in facet.getVertices():
                    ti = min(ti, v.getTime())
                    tf = max(tf, v.getTime())
                self.assertNotEqual(ti, tf)

            if facet.getOrientation().numeric() == (1, 3):
                left = facet

        self.assertEqual(ntime, 1)
        self.assertEqual(nspace, 4)
        self.assertIsNotNone(left)

        facets23 = simplex23.getFacets()
        nspace = 0
        ntime = 0
        for face in facets23:
            if face.isSpatial():
                ntime += 1
            else:
                nspace += 1
            if face.getOrientation().numeric() == (1, 3):
                right = face

        self.assertEqual(nspace, 5)
        self.assertEqual(ntime, 0)

        self.assertIsNotNone(left)
        self.assertIsNotNone(right)

        totalVerticesBefore = st.getVertexList().toVector()
        totalEdgesBefore = st.getEdgeList().toVector()

        self.assertEqual(len(totalVerticesBefore), 10)
        self.assertEqual(len(totalEdgesBefore), 20)

        leftVerticesBefore = [v.getId() for v in left.getVertices()]
        self.assertEqual(len(leftVerticesBefore), 4)
        leftEdgesBefore = [(e.getSource().getId(), e.getTarget().getId()) for e in left.getEdges()]
        self.assertEqual(len(leftEdgesBefore), 6)

        rightVerticesBefore = [v.getId() for v in right.getVertices()]
        self.assertEqual(len(rightVerticesBefore), 4)
        rightEdgesBefore = [(e.getSource().getId(), e.getTarget().getId()) for e in right.getEdges()]
        self.assertEqual(len(rightEdgesBefore), 6)

    def test_we_get_connected_components_when_constructing_from_primitives(self):
        st = Spacetime()

        vertices = []
        for i in range(10):
            vertices.append(st.createVertex(i))

        edges = []
        for i in range(0, 9, 2):
            edges.append(st.createEdge(vertices[i], vertices[i+1]))

        components = st.getConnectedComponents()
        self.assertEqual(len(components), 5)

    def test_we_get_connected_components_when_constructing_from_simplexes(self):
        st = Spacetime()

        st.createSimplex((1, 4))
        st.createSimplex((2, 3))

        components = st.getConnectedComponents()
        self.assertEqual(len(components), 2)


    def test_vertex_stores_all_simplices_in_which_it_resides(self):
        st = Spacetime()
        s14, _ = st.createSimplex((1, 4))
        k3facets = s14.getFacets()
        for k3facet in k3facets:
            for k2facet in k3facet.getFacets():
                for k1facet in k2facet.getFacets():
                    for vertex in k1facet.getVertices():
                        self.assertIn(vertex, k1facet.getVertices())
                        self.assertIn(vertex, k2facet.getVertices())
                        self.assertIn(vertex, k3facet.getVertices())
                        self.assertIn(k1facet, vertex.getSimplices())
                        self.assertIn(k2facet, vertex.getSimplices())
                        self.assertIn(k3facet, vertex.getSimplices())
                        for coface in k1facet.getCofaces():
                            self.assertIn(coface, vertex.getSimplices())
                        for coface in k2facet.getCofaces():
                            self.assertIn(coface, vertex.getSimplices())
                        for coface in k3facet.getCofaces():
                            self.assertIn(coface, vertex.getSimplices())


if __name__ == '__main__':
    unittest.main()