import unittest

from caset import Spacetime

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

        self.assertEqual(e1.getSource(), v1)
        self.assertEqual(e1.getTarget(), v2)
        self.assertEqual(e2.getSource(), v2)
        self.assertEqual(e2.getTarget(), v3)

    def test_create_simplex(self):
        st = Spacetime()
        st.setManual(False)
        simplex = st.createSimplex((2, 3))
        self.assertEqual(len(simplex.getVertices()), 5)
        self.assertEqual(len(simplex.getEdges()), 10)

        v1, v2, v3, v4, v5 = simplex.getVertices()
        a, b, c, d = v1.getOutEdges()
        self.assertEqual(a.getSource(), v1)
        self.assertEqual(b.getSource(), v1)
        self.assertEqual(c.getSource(), v1)
        self.assertEqual(d.getSource(), v1)
        self.assertEqual(len(v1.getEdges()), 4)

        a, b, c = v2.getOutEdges()
        self.assertEqual(a.getSource(), v2)
        self.assertEqual(b.getSource(), v2)
        self.assertEqual(c.getSource(), v2)
        self.assertEqual(len(v2.getEdges()), 4)
        a, = v2.getInEdges()
        self.assertEqual(a.getTarget(), v2)

        a, b = v3.getOutEdges()
        c, d = v3.getInEdges()
        self.assertEqual(a.getSource(), v3)
        self.assertEqual(b.getSource(), v3)
        self.assertEqual(c.getTarget(), v3)
        self.assertEqual(d.getTarget(), v3)
        self.assertEqual(len(v3.getEdges()), 4)

        a, = v4.getOutEdges()
        b, c, d = v4.getInEdges()
        self.assertEqual(a.getSource(), v4)
        self.assertEqual(b.getTarget(), v4)
        self.assertEqual(c.getTarget(), v4)
        self.assertEqual(d.getTarget(), v4)
        self.assertEqual(len(v4.getEdges()), 4)

        a, b, c, d = v5.getInEdges()
        self.assertEqual(a.getTarget(), v5)
        self.assertEqual(b.getTarget(), v5)
        self.assertEqual(c.getTarget(), v5)
        self.assertEqual(d.getTarget(), v5)
        self.assertEqual(len(v5.getEdges()), 4)


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
        breakpoint()
        print(updated)
        # TODO: Verify the edges going into the joined face from both simplexes are correct.

if __name__ == '__main__':
    unittest.main()