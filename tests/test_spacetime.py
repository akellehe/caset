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
        simplex = st.createSimplex((2, 3))
        self.assertEqual(len(simplex.getVertices()), 5)
        edges = simplex.getEdges()
        self.assertEqual(len(edges), 10)

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

        self.assertEqual(len(st.getSimplicesWithOrientation((2, 3))), 1)
        self.assertEqual(len(st.getSimplicesWithOrientation((1, 1))), 0)

        simplex2 = st.createSimplex((2, 3))
        self.assertEqual(len(st.getSimplicesWithOrientation((2, 3))), 2)
        self.assertEqual(len(st.getSimplicesWithOrientation((1, 1))), 0)

    def test_euclidean_embedding(self):
        st = Spacetime()
        simplex14 = st.createSimplex((1, 4))
        simplex23 = st.createSimplex((2, 3))
        st.embedEuclidean()
        vertices = st.getVertexList().toVector()

    def test_attaching_faces4D(self):
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
        ntime, nspace = 0, 0
        for facet in facets14:
            if facet.isTimelike():
                ntime += 1
            else:
                nspace += 1
            if facet.getOrientation().numeric() == (1, 3):
                left = facet

        self.assertEqual(ntime, 1)
        self.assertEqual(nspace, 4)
        self.assertIsNotNone(left)

        facets23 = simplex23.getFacets()
        nspace = 0
        ntime = 0
        for face in facets23:
            if face.isTimelike():
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
        leftEdgesBefore = [(e.getSourceId(), e.getTargetId()) for e in left.getEdges()]
        self.assertEqual(len(leftEdgesBefore), 6)

        rightVerticesBefore = [v.getId() for v in right.getVertices()]
        self.assertEqual(len(rightVerticesBefore), 4)
        rightEdgesBefore = [(e.getSourceId(), e.getTargetId()) for e in right.getEdges()]
        self.assertEqual(len(rightEdgesBefore), 6)

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

        leftVerticesAfter = [v.getId() for v in left.getVertices()]
        self.assertEqual(len(leftVerticesAfter), 4)
        leftEdgesAfter = [(e.getSourceId(), e.getTargetId()) for e in left.getEdges()]
        self.assertEqual(len(leftEdgesAfter), 6)

        rightVerticesAfter = [v.getId() for v in right.getVertices()]
        self.assertEqual(len(rightVerticesAfter), 4)
        rightEdgesAfter = [(e.getSourceId(), e.getTargetId()) for e in right.getEdges()]
        self.assertEqual(len(rightEdgesAfter), 6)

        shared = {v for v in leftVerticesAfter} & {v for v in rightVerticesAfter}
        self.assertEqual(len(shared), 4)
        self.assertEqual(len(leftVerticesAfter), len(leftVerticesBefore))

        totalVerticesAfter = set([v.getId() for v in secondVertexList.toVector()])
        totalEdgesAfter = set([(e.getSourceId(), e.getTargetId()) for e in secondEdgeList.toVector()])

        self.assertEqual(len(totalVerticesAfter), len(totalVerticesBefore) - 4)
        self.assertEqual(len(totalEdgesAfter), len(totalEdgesBefore) - 6)


    def test_attaching_faces2D(self):
        st = Spacetime()

        firstVertexList = st.getVertexList()
        firstEdgeList = st.getEdgeList()

        simplex21 = st.createSimplex((2, 1))
        simplex12 = st.createSimplex((1, 2))

        self.assertEqual(len(simplex21.getVertices()), 3)
        self.assertEqual(len(simplex21.getEdges()), 3)

        self.assertEqual(len(simplex12.getVertices()), 3)
        self.assertEqual(len(simplex12.getEdges()), 3)

        allVertices = [v.getId() for v in simplex12.getVertices() + simplex21.getVertices()]
        self.assertEqual(len(allVertices), len(set(allVertices)))

        allEdges = [(e.getSourceId(), e.getTargetId()) for e in simplex12.getEdges() + simplex21.getEdges()]
        self.assertEqual(len(simplex12.getEdges()), len(set(simplex21.getEdges())))
        self.assertEqual(len(simplex12.getEdges()), len(set(simplex21.getEdges())))

        edges12 = {(e.getSourceId(), e.getTargetId()) for e in simplex12.getEdges()}
        edges21 = {(e.getSourceId(), e.getTargetId()) for e in simplex21.getEdges()}

        self.assertTrue(edges21.isdisjoint(edges12))
        self.assertEqual(len(allEdges), len(set(allEdges)))

        totalVerticesBefore = len(st.getVertexList().toVector())
        self.assertEqual(totalVerticesBefore, 6)

        totalEdgesBefore = len(st.getEdgeList().toVector())
        self.assertEqual(totalEdgesBefore, 6)

        for edge in firstEdgeList.toVector():
            source = firstVertexList.get(edge.getSourceId())
            target = firstVertexList.get(edge.getTargetId())
            self.assertIsNotNone(source)
            self.assertIsNotNone(target)

        left, right = None, None
        facets12 = simplex12.getFacets()
        ntime, nspace = 0, 0
        for facet in facets12:
            if facet.isTimelike():
                ntime += 1
            else:
                nspace += 1
            if facet.getOrientation().numeric() == (1, 1):
                left = facet

        self.assertEqual(ntime, 1)
        self.assertEqual(nspace, 2)
        self.assertIsNotNone(left)

        facets23 = simplex21.getFacets()
        nspace = 0
        ntime = 0
        for face in facets23:
            if face.isTimelike():
                ntime += 1
            else:
                nspace += 1
            if face.getOrientation().numeric() == (1, 1):
                right = face

        self.assertEqual(nspace, 2)
        self.assertEqual(ntime, 1)

        self.assertIsNotNone(left)
        self.assertIsNotNone(right)

        totalVerticesBefore = st.getVertexList().toVector()
        totalEdgesBefore = st.getEdgeList().toVector()

        self.assertEqual(len(totalVerticesBefore), 6)
        self.assertEqual(len(totalEdgesBefore), 6)

        leftVerticesBefore = [v.getId() for v in left.getVertices()]
        self.assertEqual(len(leftVerticesBefore), 2)
        leftEdgesBefore = [(e.getSourceId(), e.getTargetId()) for e in left.getEdges()]
        self.assertEqual(len(leftEdgesBefore), 1)

        rightVerticesBefore = [v.getId() for v in right.getVertices()]
        self.assertEqual(len(rightVerticesBefore), 2)
        rightEdgesBefore = [(e.getSourceId(), e.getTargetId()) for e in right.getEdges()]
        self.assertEqual(len(rightEdgesBefore), 1)

        updated, succeeded = st.causallyAttachFaces(left, right)
        self.assertTrue(succeeded)

        v1, v2 = updated.getVertices()

        secondVertexList = st.getVertexList()
        secondEdgeList = st.getEdgeList()
        self.assertIs(firstVertexList, secondVertexList)
        self.assertIs(firstEdgeList, secondEdgeList)
        for edge in secondEdgeList.toVector():
            source = secondVertexList.get(edge.getSourceId())
            target = secondVertexList.get(edge.getTargetId())
            self.assertIsNotNone(source)
            self.assertIsNotNone(target)

        leftVerticesAfter = [v.getId() for v in left.getVertices()]
        self.assertEqual(len(leftVerticesAfter), 2)
        leftEdgesAfter = [(e.getSourceId(), e.getTargetId()) for e in left.getEdges()]
        self.assertEqual(len(leftEdgesAfter), 1)

        rightVerticesAfter = [v.getId() for v in right.getVertices()]
        self.assertEqual(len(rightVerticesAfter), 2)
        rightEdgesAfter = [(e.getSourceId(), e.getTargetId()) for e in right.getEdges()]
        self.assertEqual(len(rightEdgesAfter), 1)

        shared = {v for v in leftVerticesAfter} & {v for v in rightVerticesAfter}
        self.assertEqual(len(shared), 2)
        self.assertEqual(len(leftVerticesAfter), len(leftVerticesBefore))

        totalVerticesAfter = set([v.getId() for v in secondVertexList.toVector()])
        totalEdgesAfter = set([(e.getSourceId(), e.getTargetId()) for e in secondEdgeList.toVector()])

        self.assertEqual(len(totalVerticesAfter), len(totalVerticesBefore) - 2)
        self.assertEqual(len(totalEdgesAfter), len(totalEdgesBefore) - 1)

    def test_we_get_connected_components_when_constructing_from_primitives(self):
        st = Spacetime()

        vertices = []
        for i in range(10):
            vertices.append(st.createVertex(i))

        edges = []
        for i in range(0, 9, 2):
            edges.append(st.createEdge(vertices[i].getId(), vertices[i+1].getId()))

        components = st.getConnectedComponents()
        self.assertEqual(len(components), 5)

    def test_we_get_connected_components_when_constructing_from_simplexes(self):
        st = Spacetime()

        st.createSimplex((1, 4))
        st.createSimplex((2, 3))

        components = st.getConnectedComponents()
        self.assertEqual(len(components), 2)

    def test_components_connect_once_glued(self):
        st = Spacetime()

        s14 = st.createSimplex((1, 4))
        s23 = st.createSimplex((2, 3))

        components = st.getConnectedComponents()
        self.assertEqual(len(components), 2)

        leftFace, rightFace = st.chooseSimplexToGlueTo(s14)
        updated, succeeded = st.causallyAttachFaces(leftFace, rightFace)
        self.assertTrue(succeeded)

        components = st.getConnectedComponents()
        self.assertEqual(len(components), 1)

    def test_lots_of_components_connect_once_glued(self):
        st = Spacetime()

        orientations = [(1, 4), (2, 3)]
        for i in range(10):
            s = st.createSimplex(orientations[i % 2])
            if i == 0: continue
            leftFace, rightFace = st.chooseSimplexToGlueTo(s)
            updated, succeeded = st.causallyAttachFaces(leftFace, rightFace)
            self.assertTrue(succeeded)

        components = st.getConnectedComponents()
        self.assertEqual(len(components), 1)


if __name__ == '__main__':
    unittest.main()