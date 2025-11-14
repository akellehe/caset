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
            "<Face (1→2→3→4→1)>",
            "<Face (0→2→3→4→0)>",
            "<Face (0→1→3→4→0)>",
            "<Face (0→1→2→4→0)>",
            "<Face (0→1→2→3→0)>"
        ]
        self.assertEqual([str(f) for f in s1.getFacets()], expected)
        expected = [
            "<Face (6→7→8→9→6)>",
            "<Face (5→7→8→9→5)>",
            "<Face (5→6→8→9→5)>",
            "<Face (5→6→7→9→5)>",
            "<Face (5→6→7→8→5)>"
        ]
        self.assertEqual([str(f) for f in s2.getFacets()], expected)

    def test_attach_faces(self):
        s1 = self.spacetime.createSimplex((4, 1))
        s2 = self.spacetime.createSimplex((3, 2))
        expected = [
            "<Face (1→2→3→4→1)>",
            "<Face (0→2→3→4→0)>",
            "<Face (0→1→3→4→0)>",
            "<Face (0→1→2→4→0)>",
            "<Face (0→1→2→3→0)>"
        ]
        self.assertEqual([str(f) for f in s1.getFacets()], expected)
        expected = [
            "<Face (6→7→8→9→6)>",
            "<Face (5→7→8→9→5)>",
            "<Face (5→6→8→9→5)>",
            "<Face (5→6→7→9→5)>",
            "<Face (5→6→7→8→5)>"
        ]
        self.assertEqual([str(f) for f in s2.getFacets()], expected)


if __name__ == '__main__':
    unittest.main()
