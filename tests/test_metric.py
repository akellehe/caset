import unittest

from caset import Edge, Vertex, Metric, Signature, SignatureType


class TestMetric(unittest.TestCase):

    def test_metric_instantiates(self):
        v1 = Vertex(1, [0, 0, 0, 0])
        v2 = Vertex(2, [0, 0, 0, 1])
        edge = Edge(v1, v2, 25)

        self.assertIsInstance(edge, Edge)
        src = edge.getSource()
        tgt = edge.getTarget()
        self.assertIs(src, v1)
        self.assertIs(tgt, v2)

        signature = Signature(4, SignatureType.Lorentzian)
        self.assertEqual(signature.getDiagonal(), [-1, 1, 1, 1])
        metric = Metric(True, signature)
        self.assertEqual(metric.getSquaredLength(edge), 25)

        signature = Signature(4, SignatureType.Lorentzian)
        self.assertEqual(signature.getDiagonal(), [-1, 1, 1, 1])
        metric = Metric(False, signature)
        self.assertEqual(metric.getSquaredLength(edge), 1)


if __name__ == '__main__':
    unittest.main()
