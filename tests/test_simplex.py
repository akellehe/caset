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
            "<Face (<V1>→<V2>→<V3>→<V4>→<V1>)>",
            "<Face (<V0>→<V2>→<V3>→<V4>→<V0>)>",
            "<Face (<V0>→<V1>→<V3>→<V4>→<V0>)>",
            "<Face (<V0>→<V1>→<V2>→<V4>→<V0>)>",
            "<Face (<V0>→<V1>→<V2>→<V3>→<V0>)>"
        ]
        self.assertEqual([str(f) for f in s1.getFacets()], expected)
        expected = [
            "<Face (<V6>→<V7>→<V8>→<V9>→<V6>)>",
            "<Face (<V5>→<V7>→<V8>→<V9>→<V5>)>",
            "<Face (<V5>→<V6>→<V8>→<V9>→<V5>)>",
            "<Face (<V5>→<V6>→<V7>→<V9>→<V5>)>",
            "<Face (<V5>→<V6>→<V7>→<V8>→<V5>)>"
        ]
        self.assertEqual([str(f) for f in s2.getFacets()], expected)

    def test_creating_oriented_simplexes(self):
        ti, tf = (4, 1)
        s1 = self.spacetime.createSimplex((ti, tf))
        expected = [
            "<Face (<V1>→<V2>→<V3>→<V4>→<V1>)>",
            "<Face (<V0>→<V2>→<V3>→<V4>→<V0>)>",
            "<Face (<V0>→<V1>→<V3>→<V4>→<V0>)>",
            "<Face (<V0>→<V1>→<V2>→<V4>→<V0>)>",
            "<Face (<V0>→<V1>→<V2>→<V3>→<V0>)>"
        ]
        self.assertEqual([str(f) for f in s1.getFacets()], expected)
        oti, otf = (0, 0)
        for edge in s1.getEdges():
            if edge.getSquaredLength() < 0:
                oti += 1
            else:
                otf += 1
        self.assertEqual(oti, ti)
        self.assertEqual(otf, tf)

        s2 = self.spacetime.createSimplex((3, 2))
        expected = [
            "<Face (<V6>→<V7>→<V8>→<V9>→<V6>)>",
            "<Face (<V5>→<V7>→<V8>→<V9>→<V5>)>",
            "<Face (<V5>→<V6>→<V8>→<V9>→<V5>)>",
            "<Face (<V5>→<V6>→<V7>→<V9>→<V5>)>",
            "<Face (<V5>→<V6>→<V7>→<V8>→<V5>)>"
        ]
        self.assertEqual([str(f) for f in s2.getFacets()], expected)


if __name__ == '__main__':
    unittest.main()
