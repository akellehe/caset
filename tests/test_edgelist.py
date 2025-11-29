import unittest
from caset import Edge, EdgeList

class TestEdgeList(unittest.TestCase):

    def test_adding_and_removing_unique_edges(self):
        el = EdgeList()
        self.assertEqual(el.size(), 0)
        self.assertEqual(len(el.toVector()), 0)
        el.add(Edge(1, 2))
        self.assertEqual(el.size(), 1)
        self.assertEqual(len(el.toVector()), 1)
        el.add(Edge(1, 2))
        self.assertEqual(el.size(), 1)
        self.assertEqual(len(el.toVector()), 1)
        with self.assertRaisesRegex(RuntimeError, "Fingerprint collision"):
            el.add(Edge(2, 1))
        self.assertEqual(el.size(), 1)
        self.assertEqual(len(el.toVector()), 1)
        el.add(Edge(1, 2, 3.))
        self.assertEqual(el.size(), 1)
        self.assertEqual(len(el.toVector()), 1)

    def test_uniqueness_after_redirecting_edges(self):
        e1 = Edge(1, 2)
        e2 = Edge(2, 5)
        e3 = Edge(3, 4)

        el = EdgeList()
        el.add(e1)
        el.add(e2)
        el.add(e3)

        e1.redirect(1, 3)

        el.add(e1)

        self.assertEqual(el.size(), 3)
