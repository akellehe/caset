import unittest

from caset import Vertex, Face, Spacetime

class TestFace(unittest.TestCase):

    def setUp(self):
        self.spacetime = Spacetime()

    def test_pairty(self):
        simplex41 = self.spacetime.createSimplex((4, 1))
        f1, f2, f3, f4, f5 = simplex41.getFacets()

        self.assertEqual(len(f1.getVertices()), 4)

        # Disjoint faces have pairty flag=0
        self.assertEqual(f1.checkPairty(f2), 0)

        v1, v2, v3, v4 = f1.getVertices()

        #The same face has pairty flag=1
        clone = Face([], [v1, v2, v3, v4])
        self.assertEqual(f1.checkPairty(clone), 1)

        # A single vertex swap has pairty flag=-1
        oneSwap = Face([], [v2, v1, v3, v4])
        self.assertEqual(f1.checkPairty(oneSwap), -1)

        # Two swaps has pairty flag=1
        twoSwaps = Face([], [v2, v1, v4, v3])
        self.assertEqual(f1.checkPairty(twoSwaps), 1)

    def test_get_edges(self):
        simplex41 = self.spacetime.createSimplex((4, 1))

        f1, f2, f3, f4, f5 = simplex41.getFacets()
        v1, v2, v3, v4 = f1.getVertices()
        e1, e2, e3, e4 = f1.getEdges()

        self.assertTrue(e1.getSourceId() == v1.getId() or e1.getTargetId() == v1.getId())
        self.assertTrue(e1.getSourceId() == v2.getId() or e1.getTargetId() == v2.getId())

        self.assertTrue(e2.getSourceId() == v2.getId() or e2.getTargetId() == v2.getId())
        self.assertTrue(e2.getSourceId() == v3.getId() or e2.getTargetId() == v3.getId())

        self.assertTrue(e3.getSourceId() == v3.getId() or e3.getTargetId() == v3.getId())
        self.assertTrue(e3.getSourceId() == v4.getId() or e3.getTargetId() == v4.getId())

        self.assertTrue(e4.getSourceId() == v4.getId() or e4.getTargetId() == v4.getId())
        self.assertTrue(e4.getSourceId() == v1.getId() or e4.getTargetId() == v1.getId())


if __name__ == '__main__':
    unittest.main()

