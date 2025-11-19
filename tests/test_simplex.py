import unittest

from caset import Vertex, Simplex, Metric, Spacetime, Signature, SignatureType


class TestSimplex(unittest.TestCase):
    def setUp(self):
        self.spacetime = Spacetime()
        self.spacetime.setManual(False)

    def test_get_faces(self):
        s1 = self.spacetime.createSimplex((4, 1))
        s2 = self.spacetime.createSimplex((3, 2))
        expected = [
            "<4-Simplex (<V1>→<V2>→<V3>→<V4>→<V1>)>",
            "<4-Simplex (<V0>→<V2>→<V3>→<V4>→<V0>)>",
            "<4-Simplex (<V0>→<V1>→<V3>→<V4>→<V0>)>",
            "<4-Simplex (<V0>→<V1>→<V2>→<V4>→<V0>)>",
            "<4-Simplex (<V0>→<V1>→<V2>→<V3>→<V0>)>"
        ]
        self.assertEqual([str(f) for f in s1.getFacets()], expected)
        expected = [
            "<4-Simplex (<V6>→<V7>→<V8>→<V9>→<V6>)>",
            "<4-Simplex (<V5>→<V7>→<V8>→<V9>→<V5>)>",
            "<4-Simplex (<V5>→<V6>→<V8>→<V9>→<V5>)>",
            "<4-Simplex (<V5>→<V6>→<V7>→<V9>→<V5>)>",
            "<4-Simplex (<V5>→<V6>→<V7>→<V8>→<V5>)>"
        ]
        self.assertEqual([str(f) for f in s2.getFacets()], expected)

    def test_creating_oriented_simplexes(self):
        ti, tf = (4, 1)
        s1 = self.spacetime.createSimplex((ti, tf))
        expected = [
            "<4-Simplex (<V1>→<V2>→<V3>→<V4>→<V1>)>",
            "<4-Simplex (<V0>→<V2>→<V3>→<V4>→<V0>)>",
            "<4-Simplex (<V0>→<V1>→<V3>→<V4>→<V0>)>",
            "<4-Simplex (<V0>→<V1>→<V2>→<V4>→<V0>)>",
            "<4-Simplex (<V0>→<V1>→<V2>→<V3>→<V0>)>"
        ]
        self.assertEqual([str(f) for f in s1.getFacets()], expected)
        oti, otf = (0, 0)
        seen = set()
        initialTime = 0
        finalTime = 0
        for vertex in s1.getVertices():
            initialTime = min(initialTime, vertex.getTime())
            finalTime = max(finalTime, vertex.getTime())
        for vertex in s1.getVertices():
            if vertex.getTime() == initialTime:
                oti += 1
            elif vertex.getTime() == finalTime:
                otf += 1
        self.assertEqual(oti, ti)
        self.assertEqual(otf, tf)

        s2 = self.spacetime.createSimplex((3, 2))
        expected = [
            "<4-Simplex (<V6>→<V7>→<V8>→<V9>→<V6>)>",
            "<4-Simplex (<V5>→<V7>→<V8>→<V9>→<V5>)>",
            "<4-Simplex (<V5>→<V6>→<V8>→<V9>→<V5>)>",
            "<4-Simplex (<V5>→<V6>→<V7>→<V9>→<V5>)>",
            "<4-Simplex (<V5>→<V6>→<V7>→<V8>→<V5>)>"
        ]
        self.assertEqual([str(f) for f in s2.getFacets()], expected)

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
        clone = Simplex([v1, v2, v3, v4])
        self.assertEqual(f1.checkPairty(clone), 1)

        # A single vertex swap has pairty flag=-1
        oneSwap = Simplex([v2, v1, v3, v4])
        self.assertEqual(f1.checkPairty(oneSwap), -1)

        # Two swaps has pairty flag=1
        twoSwaps = Simplex([v2, v1, v4, v3])
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
