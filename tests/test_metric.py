import unittest

from caset import Edge, Vertex, Metric, Signature, SignatureType


class TestMetric(unittest.TestCase):

    def test_metric_instantiates(self):
        v1 = Vertex(1, [0, 0, 0, 0])
        v2 = Vertex(2, [0, 0, 0, 1])
        edge = Edge(v1.getId(), v2.getId(), 25)

        self.assertIsInstance(edge, Edge)
        src = edge.getSourceId()
        tgt = edge.getTargetId()
        self.assertIs(src, v1.getId())
        self.assertIs(tgt, v2.getId())

        signature = Signature(4, SignatureType.Lorentzian)
        self.assertEqual(signature.getDiagonal(), [-1, 1, 1, 1])
        metric = Metric(True, signature)
        with self.assertRaisesRegex(RuntimeError, "You asked a coordinate free metric to compute the squared length of an edge"):
            metric.getSquaredLength(v1.getCoordinates(), v2.getCoordinates())

        signature = Signature(4, SignatureType.Lorentzian)
        self.assertEqual(signature.getDiagonal(), [-1, 1, 1, 1])
        metric = Metric(False, signature)
        self.assertEqual(metric.getSquaredLength(v1.getCoordinates(), v2.getCoordinates()), 1)


if __name__ == '__main__':
    unittest.main()
