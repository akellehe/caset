import unittest

from caset import Vertex


class TestVertex(unittest.TestCase):

    def test_vertex_creation(self):
        v1 = Vertex(1, [1, 2, 3, 4])
        coords = v1.getCoordinates()
        self.assertEqual(coords, [1, 2, 3, 4])