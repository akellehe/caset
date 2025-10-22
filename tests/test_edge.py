import unittest

from caset import Edge, Vertex


class TestEdge(unittest.TestCase):

    def test_edge_instantiates(self):
        v1 = Vertex(0, [0, 0, 0, 0])
        v2 = Vertex(1, [0, 0, 0, 1])
        edge = Edge(v1, v2)

        self.assertIsInstance(edge, Edge)
        self.assertEqual(edge.getLength(), 1)



if __name__ == '__main__':
    unittest.main()
