import unittest

from caset import Vertex, Simplex, Metric, Spacetime, Signature, SignatureType

class TestSimplex(unittest.TestCase):

    def test_simplex(self):
        st = Spacetime()
        st.manual = False
        simplex = st.createSimplex(5)




