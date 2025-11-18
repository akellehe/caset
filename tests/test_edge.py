import unittest

from caset import Edge, Vertex, Spacetime


class TestEdge(unittest.TestCase):

    def test_edge_instantiates(self):
        v1 = Vertex(0, [0, 0, 0, 0])
        v2 = Vertex(1, [1, 1, 1, 1])
        edge = Edge(v1.getId(), v2.getId())

        self.assertIsInstance(edge, Edge)
        src = edge.getSourceId()
        tgt = edge.getTargetId()
        self.assertIs(src, v1.getId())
        self.assertIs(tgt, v2.getId())


if __name__ == '__main__':
    unittest.main()
