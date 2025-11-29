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
from caset import Edge, EdgeList

class TestEdgeList(unittest.TestCase):

    def test_adding_and_removing_unique_edges(self):
        el = EdgeList()
        self.assertEqual(el.size(), 0)
        self.assertEqual(len(el.toVector()), 0)
        el.add(Edge(1, 2))
        self.assertEqual(el.size(), 1)
        self.assertEqual(len(el.toVector()), 1)
        el.add(Edge(1, 2))
        self.assertEqual(el.size(), 1)
        self.assertEqual(len(el.toVector()), 1)
        with self.assertRaisesRegex(RuntimeError, "Fingerprint collision"):
            el.add(Edge(2, 1))
        self.assertEqual(el.size(), 1)
        self.assertEqual(len(el.toVector()), 1)
        el.add(Edge(1, 2, 3.))
        self.assertEqual(el.size(), 1)
        self.assertEqual(len(el.toVector()), 1)

    def test_uniqueness_after_redirecting_edges(self):
        e1 = Edge(1, 2)
        e2 = Edge(2, 5)
        e3 = Edge(3, 4)

        el = EdgeList()
        el.add(e1)
        el.add(e2)
        el.add(e3)

        e1.redirect(1, 3)

        el.add(e1)

        self.assertEqual(el.size(), 3)
