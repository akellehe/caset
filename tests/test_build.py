import unittest
import importlib.util
import sys

class TestBuilding(unittest.TestCase):
    def test_building(self):
        path = None
        for p in sys.path:
            if "cmake-build-debug" in p:
                path = p
                break

        print("sys.path[0..5]:")
        for p in sys.path[:6]:
            print(" ", p)

        spec = importlib.util.find_spec("caset")
        if spec is not None:
            print("  origin:", spec.origin)
            print("  loader:", spec.loader)

        self.assertNotIn("site-packages", spec.origin)
        self.assertIsNotNone(path)
