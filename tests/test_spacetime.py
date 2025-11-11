import unittest

from caset import Spacetime

class TestSpacetime(unittest.TestCase):

    def test_create_vertex(self):
        st = Spacetime()
        v1 = st.createVertex(1)
        v2 = st.createVertex(2)
        print(st)



if __name__ == '__main__':
    unittest.main()