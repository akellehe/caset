# MIT License
# Copyright (c) 2025 Andrew Kelleher
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
