import unittest

from caset import Vertex, Simplex, Metric, Spacetime, Signature, SignatureType

class TestSimplex(unittest.TestCase):

    def test_simplex(self):
        signature = Signature(4, SignatureType.Lorentzian)
        metric = Metric(True, signature)
        spacetime = Spacetime(metric)



