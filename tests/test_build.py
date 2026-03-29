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
import importlib.util
import sys

class TestBuilding(unittest.TestCase):
    def test_building(self):
        """Verify that caset is importable — from cmake-build-debug, pip install -e ., or pip install ."""
        spec = importlib.util.find_spec("caset")
        self.assertIsNotNone(spec, "caset module not found")
        self.assertIsNotNone(spec.origin, "caset module has no origin")

        print("  origin:", spec.origin)

        if "site-packages" in (spec.origin or ""):
            # Loaded from a pip install.  Accept both editable (`pip install -e .`)
            # and regular (`pip install .`) local installs — both produce a
            # direct_url.json with a file:// URL pointing at the source tree.
            # Only reject if there's no direct_url at all (stale system wheel).
            from importlib.metadata import distribution
            dist = distribution("caset")
            direct_url = dist.read_text("direct_url.json") or ""
            self.assertTrue(
                direct_url,
                f"caset loaded from site-packages without a direct_url.json — "
                f"stale install? origin: {spec.origin}")
        else:
            # Loaded from cmake-build-debug (conftest.py injected it)
            found = any("cmake-build-debug" in p for p in sys.path)
            self.assertTrue(
                found,
                "cmake-build-debug not on sys.path and caset is not a pip install")
