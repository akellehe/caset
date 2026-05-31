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

"""Cobordism fixture triangulations (#63), built via Topology subclasses.

These topologies build exact, minimal, pre-geometric (coordinate-free)
triangulations. We verify each built complex against its known f-vector and
Euler characteristic by counting distinct k-faces combinatorially from the top
simplices — independent of any geometry.
"""

import itertools
import unittest

import tessera

cobordism = tessera.cobordism


def _build(topology):
    """Build a topology into a fresh Spacetime via the idiomatic flow."""
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, topology)
    st.build()  # delegates to topology.build(); numSimplices ignored
    return st


def _f_vector(st):
    """Count distinct k-faces (k=0..dim) from the top simplices."""
    tops = [tuple(sorted(v.getId() for v in s.getVertices()))
            for s in st.getSimplices()]
    if not tops:
        return []
    maxsz = max(len(t) for t in tops)
    f = []
    for card in range(1, maxsz + 1):
        faces = set()
        for t in tops:
            if len(t) >= card:
                faces.update(itertools.combinations(t, card))
        f.append(len(faces))
    return f


def _euler(fvec):
    return sum((-1) ** k * n for k, n in enumerate(fvec))


class TestFixtures(unittest.TestCase):

    def _check(self, topology, fvector, euler, dim):
        st = _build(topology)
        self.assertEqual(_f_vector(st), fvector)
        self.assertEqual(_euler(fvector), euler)
        self.assertEqual(cobordism.CombinatorialDimension().compute(st), float(dim))

    # S^n = ∂Δ^{n+1}
    def test_sphere_S1(self):
        self._check(tessera.SimplexBoundarySphere(1), [3, 3], 0, 1)

    def test_sphere_S2(self):
        self._check(tessera.SimplexBoundarySphere(2), [4, 6, 4], 2, 2)

    def test_sphere_S3(self):
        self._check(tessera.SimplexBoundarySphere(3), [5, 10, 10, 5], 0, 3)

    def test_sphere_S4(self):
        self._check(tessera.SimplexBoundarySphere(4), [6, 15, 20, 15, 6], 2, 4)

    # Solid simplices Δ^n (closed n-balls)
    def test_ball_D2(self):
        self._check(tessera.SolidSimplex(2), [3, 3, 1], 1, 2)

    def test_ball_D4(self):
        self._check(tessera.SolidSimplex(4), [5, 10, 10, 5, 1], 1, 4)

    # ℝP² minimal 6-vertex
    def test_rp2(self):
        self._check(tessera.RealProjectivePlane(), [6, 15, 10], 1, 2)

    def test_fixtures_are_pre_geometric(self):
        # Vertices carry no coordinates until geometry is needed.
        st = _build(tessera.SimplexBoundarySphere(2))
        v = st.getVertexList().toVector()[0]
        with self.assertRaises(Exception):
            v.getCoordinates()  # coordinate-free vertex


if __name__ == "__main__":
    unittest.main()
