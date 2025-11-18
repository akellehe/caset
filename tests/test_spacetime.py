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
        st.embedEuclidean(4, 0.0001, 100000)
        vertices = st.getVertexList().toVector()
        breakpoint()


    def test_attaching_faces(self):
        st = Spacetime()
        simplex14 = st.createSimplex((1, 4))
        simplex23 = st.createSimplex((2, 3))

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

        updated, succeeded = st.causallyAttachFaces(left, right)
        self.assertTrue(succeeded)

        v1, v2, v3, v4 = updated.getVertices()

        # (Pdb) updated.getVertices()[0].getInEdges()
        # {5->0}
        # edges = v1.getEdges()
        # breakpoint()
        # self.assertEqual(e1.getSourceId(), 5)
        # self.assertEqual(e1.getTargetId(), 0)

        # (Pdb) updated.getVertices()[1].getInEdges()
        # {0->2, 5->2, 1->2}
        # e1, e2, e3 = v2.getInEdges()
        # self.assertEqual(e1.getSourceId(), 0)
        # self.assertEqual(e1.getTargetId(), 2)
        # self.assertEqual(e2.getSourceId(), 5)
        # self.assertEqual(e2.getTargetId(), 2)
        # self.assertEqual(e3.getSourceId(), )
        # self.assertEqual(e3.getTargetId(), )

        # (Pdb) updated.getVertices()[2].getInEdges()
        # {2->3, 0->3, 5->3, 1->3}
        # e1, e2, e3, e4 = v3.getInEdges()
        # self.assertEqual(e1.getSourceId(), 2)
        # self.assertEqual(e1.getTargetId(), 3)
        # self.assertEqual(e2.getSourceId(), 0)
        # self.assertEqual(e2.getTargetId(), 3)
        # self.assertEqual(e3.getSourceId(), 5)
        # self.assertEqual(e3.getTargetId(), 3)
        # self.assertEqual(e4.getSourceId(), 1)
        # self.assertEqual(e4.getTargetId(), 3)

        # (Pdb) updated.getVertices()[3].getInEdges()
        # {3->4, 0->4, 2->4, 5->4}
        # e1, e2, e3, e4 = v3.getInEdges()
        # self.assertEqual(e1.getSourceId(), 3)
        # self.assertEqual(e1.getTargetId(), 4)
        # self.assertEqual(e2.getSourceId(), 0)
        # self.assertEqual(e2.getTargetId(), 4)
        # self.assertEqual(e3.getSourceId(), 2)
        # self.assertEqual(e3.getTargetId(), 4)
        # self.assertEqual(e4.getSourceId(), 5)
        # self.assertEqual(e4.getTargetId(), 4)


        # breakpoint()
        # print(updated)
        # TODO: Verify the edges going into the joined face from both simplexes are correct.

if __name__ == '__main__':
    unittest.main()