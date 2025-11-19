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

        e1 = st.createEdge(v1.getId(), v2.getId())
        e2 = st.createEdge(v2.getId(), v3.getId())

        self.assertNotEqual(v1, v2)
        self.assertNotEqual(v2, v3)
        self.assertNotEqual(e1, e2)

        self.assertEqual(e1.getSourceId(), v1.getId())
        self.assertEqual(e1.getTargetId(), v2.getId())
        self.assertEqual(e2.getSourceId(), v2.getId())
        self.assertEqual(e2.getTargetId(), v3.getId())

    def test_create_simplex(self):
        st = Spacetime()
        st.setManual(False)
        simplex = st.createSimplex((2, 3))
        self.assertEqual(len(simplex.getVertices()), 5)
        breakpoint()
        self.assertEqual(len(simplex.getEdges()), 10)

        v1, v2, v3, v4, v5 = simplex.getVertices()
        a, b, c, d = v1.getOutEdges()
        self.assertEqual(a.getSourceId(), v1.getId())
        self.assertEqual(b.getSourceId(), v1.getId())
        self.assertEqual(c.getSourceId(), v1.getId())
        self.assertEqual(d.getSourceId(), v1.getId())
        self.assertEqual(len(v1.getEdges()), 4)

        a, b, c = v2.getOutEdges()
        self.assertEqual(a.getSourceId(), v2.getId())
        self.assertEqual(b.getSourceId(), v2.getId())
        self.assertEqual(c.getSourceId(), v2.getId())
        self.assertEqual(len(v2.getEdges()), 4)
        a, = v2.getInEdges()
        self.assertEqual(a.getTargetId(), v2.getId())

        a, b = v3.getOutEdges()
        c, d = v3.getInEdges()
        self.assertEqual(a.getSourceId(), v3.getId())
        self.assertEqual(b.getSourceId(), v3.getId())
        self.assertEqual(c.getTargetId(), v3.getId())
        self.assertEqual(d.getTargetId(), v3.getId())
        self.assertEqual(len(v3.getEdges()), 4)

        a, = v4.getOutEdges()
        b, c, d = v4.getInEdges()
        self.assertEqual(a.getSourceId(), v4.getId())
        self.assertEqual(b.getTargetId(), v4.getId())
        self.assertEqual(c.getTargetId(), v4.getId())
        self.assertEqual(d.getTargetId(), v4.getId())
        self.assertEqual(len(v4.getEdges()), 4)

        a, b, c, d = v5.getInEdges()
        self.assertEqual(a.getTargetId(), v5.getId())
        self.assertEqual(b.getTargetId(), v5.getId())
        self.assertEqual(c.getTargetId(), v5.getId())
        self.assertEqual(d.getTargetId(), v5.getId())
        self.assertEqual(len(v5.getEdges()), 4)

    def test_euclidean_embedding(self):
        st = Spacetime()
        simplex14 = st.createSimplex((1, 4))
        simplex23 = st.createSimplex((2, 3))
        st.embedEuclidean()
        vertices = st.getVertexList().toVector()
        breakpoint()

    def test_attaching_faces(self):
        st = Spacetime()

        firstVertexList = st.getVertexList()
        firstEdgeList = st.getEdgeList()

        simplex14 = st.createSimplex((1, 4))
        simplex23 = st.createSimplex((2, 3))

        self.assertEqual(len(simplex14.getVertices()), 5)
        self.assertEqual(len(simplex14.getEdges()), 10)

        self.assertEqual(len(simplex23.getVertices()), 5)
        self.assertEqual(len(simplex23.getEdges()), 10)

        allVertices = [v.getId() for v in simplex14.getVertices() + simplex23.getVertices()]
        self.assertEqual(len(allVertices), len(set(allVertices)))

        allEdges = [(e.getSourceId(), e.getTargetId()) for e in simplex14.getEdges() + simplex23.getEdges()]
        self.assertEqual(len(simplex14.getEdges()), len(set(simplex14.getEdges())))
        self.assertEqual(len(simplex23.getEdges()), len(set(simplex23.getEdges())))

        edges23 = {(e.getSourceId(), e.getTargetId()) for e in simplex23.getEdges()}
        edges14 = {(e.getSourceId(), e.getTargetId()) for e in simplex14.getEdges()}

        self.assertTrue(edges23.isdisjoint(edges14))
        self.assertEqual(len(allEdges), len(set(allEdges)))

        totalVerticesBefore = len(st.getVertexList().toVector())
        self.assertEqual(totalVerticesBefore, 10)

        totalEdgesBefore = len(st.getEdgeList().toVector())
        self.assertEqual(totalEdgesBefore, 20)

        for edge in firstEdgeList.toVector():
            source = firstVertexList.get(edge.getSourceId())
            target = firstVertexList.get(edge.getTargetId())
            self.assertIsNotNone(source)
            self.assertIsNotNone(target)

        left, right = None, None
        facets14 = simplex14.getFacets()
        for facet in facets14:
            if facet.isTimelike():
                left = facet
                break

        facets23 = simplex23.getFacets()
        for face in facets23:
            if face.isTimelike():
                right = face
                break

        totalVerticesBefore = len(st.getVertexList().toVector())
        totalEdgesBefore = len(st.getEdgeList().toVector())
        self.assertEqual(totalVerticesBefore, 10)
        self.assertEqual(totalEdgesBefore, 20)

        rightVertexesBefore = [v.getId() for v in right.getVertices()]
        self.assertEqual(len(rightVertexesBefore), 4)
        rightEdgesBefore = [(e.getSourceId(), e.getTargetId()) for e in right.getEdges()]
        self.assertEqual(len(rightEdgesBefore), 4)

        updated, succeeded = st.causallyAttachFaces(left, right)
        self.assertTrue(succeeded)

        v1, v2, v3, v4 = updated.getVertices()

        secondVertexList = st.getVertexList()
        secondEdgeList = st.getEdgeList()
        self.assertIs(firstVertexList, secondVertexList)
        self.assertIs(firstEdgeList, secondEdgeList)
        for edge in secondEdgeList.toVector():
            source = secondVertexList.get(edge.getSourceId())
            target = secondVertexList.get(edge.getTargetId())
            self.assertIsNotNone(source)
            self.assertIsNotNone(target)

        rightVertexesAfter = [v.getId() for v in right.getVertices()]
        rightEdgesAfter = [(e.getSourceId(), e.getTargetId()) for e in right.getEdges()]
        leftVertexesAfter = [v.getId() for v in left.getVertices()]
        leftEdgesAfter = [(e.getSourceId(), e.getTargetId()) for e in left.getEdges()]

        totalVertexesAfter = [v.getId() for v in secondVertexList.toVector()]
        totalEdgesAfter = [(e.getSourceId(), e.getTargetId()) for e in secondEdgeList.toVector()]

        nTotalVerticesAfterAsList = len(totalVertexesAfter)
        nTotalEdgesAfterAsList = len(totalEdgesAfter)
        nTotalVerticesAfterAsSet = len(set(totalVertexesAfter))
        nTotalEdgesAfterAsSet = len(set(totalEdgesAfter))

        breakpoint()

        breakpoint()
        self.assertEqual(totalVerticesAfter, 10)

        breakpoint()
        print('foo')

if __name__ == '__main__':
    unittest.main()