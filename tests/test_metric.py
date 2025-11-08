import unittest

from caset import Edge, Vertex, Metric


class TestEdge(unittest.TestCase):

    def test_edge_instantiates(self):
        v1 = Vertex([0, 0, 0, 0])
        v2 = Vertex([0, 0, 0, 1])
        edge = Edge(v1, v2)

        self.assertIsInstance(edge, Edge)
        src = edge.getSource()
        tgt = edge.getTarget()
        self.assertIs(src, v1)
        self.assertIs(tgt, v2)

        metric = Metric()


if __name__ == '__main__':
    unittest.main()
