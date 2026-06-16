# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

import unittest

from tessera import Edge, Vertex, Metric, Signature, SignatureType


class TestMetric(unittest.TestCase):

    def test_metric_instantiates(self):
        v1 = Vertex(1, [0, 0, 0, 0])
        v2 = Vertex(2, [0, 0, 0, 1])
        edge = Edge(v1, v2, complex(5.0, 0.0))  # spacelike length 5 (l^2 = 25)

        self.assertIsInstance(edge, Edge)
        src = edge.getSource().getId()
        tgt = edge.getTarget().getId()
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
