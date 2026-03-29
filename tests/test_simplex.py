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

from caset import Vertex, Simplex, Metric, Spacetime, Signature, SignatureType


class TestSimplex(unittest.TestCase):
    def setUp(self):
        self.spacetime = Spacetime()

    def test_get_faces(self):
        s1, _ = self.spacetime.createSimplex((4, 1))
        facets = s1.getFacets()
        self.assertEqual(len(facets), 5)
        tio, tfo = (0, 0)
        for vertex in s1.getVertices():
            if vertex.getTime() == 0:
                tio += 1
            elif vertex.getTime() > 0:
                tfo += 1

        self.assertEqual(tio, 4)
        self.assertEqual(tfo, 1)

        nTimelike = 0
        for face in s1.getFacets():
            face.validate()
            self.assertEqual(len(face.getVertices()), 4)
            self.assertEqual(len(face.getEdges()), 6)
            self.assertEqual(len(set([(e.getSource().getId(), e.getTarget().getId()) for e in face.getEdges()])), 6)
            self.assertEqual(len(face.getCofaces()), 1)
            if face.isSpatial():
                nTimelike += 1
                for timelikeFace in face.getFacets():
                    timelikeFace.validate()
                    self.assertTrue(timelikeFace.isSpatial())
                    self.assertEqual(len(timelikeFace.getVertices()), 3)
                    self.assertEqual(len(timelikeFace.getEdges()), 3)
                    self.assertEqual(len(set([(e.getSource().getId(), e.getTarget().getId()) for e in timelikeFace.getEdges()])), 3)
                    self.assertEqual(len(timelikeFace.getCofaces()), 1)

        self.assertEqual(nTimelike, 1)

    def test_creating_oriented_simplices(self):
        ti, tf = (4, 1)
        s1, _ = self.spacetime.createSimplex((ti, tf))
        oti, otf = (0, 0)
        initialTime = 0
        finalTime = 0
        for vertex in s1.getVertices():
            initialTime = min(initialTime, vertex.getTime())
            finalTime = max(finalTime, vertex.getTime())

        for vertex in s1.getVertices():
            if vertex.getTime() == initialTime:
                oti += 1
            elif vertex.getTime() == finalTime:
                otf += 1

        self.assertEqual(oti, ti)
        self.assertEqual(otf, tf)

        ti, tf = (3, 2)
        s2, _ = self.spacetime.createSimplex((ti, tf))
        oti, otf = (0, 0)
        initialTime = 0
        finalTime = 0
        for vertex in s2.getVertices():
            initialTime = min(initialTime, vertex.getTime())
            finalTime = max(finalTime, vertex.getTime())

        for vertex in s2.getVertices():
            if vertex.getTime() == initialTime:
                oti += 1
            elif vertex.getTime() == finalTime:
                otf += 1

        for f in s1.getFacets():
            f.validate()
        for f in s2.getFacets():
            f.validate()
        self.assertEqual(oti, ti)
        self.assertEqual(otf, tf)

    @unittest.skip("Parity check not yet implemented")
    def test_parity(self):
        simplex41, _ = self.spacetime.createSimplex((4, 1))
        f1, f2, f3, f4, f5 = simplex41.getFacets()

        self.assertEqual(len(f1.getVertices()), 4)

        # Disjoint faces have pairty flag=0
        self.assertEqual(f1.checkPairty(f2), 0)

        v1, v2, v3, v4 = f1.getVertices()

        #The same face has pairty flag=1
        clone = Simplex([v1, v2, v3, v4])
        self.assertEqual(f1.checkPairty(clone), 1)

        # A single vertex swap has pairty flag=-1
        oneSwap = Simplex([v2, v1, v3, v4])
        self.assertEqual(f1.checkPairty(oneSwap), -1)

        # Two swaps has pairty flag=1
        twoSwaps = Simplex([v2, v1, v4, v3])
        self.assertEqual(f1.checkPairty(twoSwaps), 1)

        for f in simplex41.getFacets():
            f.validate()

    def test_get_edges(self):
        simplex41, _ = self.spacetime.createSimplex((4, 1))

        f1, f2, f3, f4, f5 = simplex41.getFacets()
        v1, v2, v3, v4 = sorted(v.getId() for v in f1.getVertices())
        e1, e2, e3, e4, e5, e6 = sorted([(e.getSource().getId(), e.getTarget().getId()) for e in f1.getEdges()])

        """
(Pdb) e1
(1, 2)
(Pdb) e2
(1, 3)
(Pdb) e3
(1, 4)
(Pdb) e4
(2, 3)
(Pdb) e5
(2, 4)
(Pdb) e6
(3, 4)
        """

        # 1>2
        self.assertEqual(e1[0], 1)
        self.assertEqual(e1[1], 2)

        # 2>3
        self.assertEqual(e2[0], 1)
        self.assertEqual(e2[1], 3)

        # 1>3
        self.assertEqual(e3[0], 1)
        self.assertEqual(e3[1], 4)

        # 3>4
        self.assertEqual(e4[0], 2)
        self.assertEqual(e4[1], 3)

        # 2>4
        self.assertEqual(e5[0], 2)
        self.assertEqual(e5[1], 4)

        # 1>4
        self.assertEqual(e6[0], 3)
        self.assertEqual(e6[1], 4)


    def test_is_timelike14(self):
        st = Spacetime()
        simplex14, _ = st.createSimplex((1, 4))
        ntime = 0
        nspace = 0
        for facet in simplex14.getFacets():
            if facet.isSpatial():
                ntime += 1
            else:
                nspace += 1

        self.assertEqual(ntime, 1)
        self.assertEqual(nspace, 4)

    def test_is_timelike41(self):
        st = Spacetime()
        simplex41, _ = st.createSimplex((4, 1))
        ntime = 0
        nspace = 0
        for facet in simplex41.getFacets():
            if facet.isSpatial():
                ntime += 1
            else:
                nspace += 1

        self.assertEqual(ntime, 1)
        self.assertEqual(nspace, 4)

    def test_is_timelike23(self):
        st = Spacetime()
        simplex23, _ = st.createSimplex((2, 3))
        ntime = 0
        nspace = 0
        for facet in simplex23.getFacets():
            if facet.isSpatial():
                ntime += 1
            else:
                nspace += 1

        self.assertEqual(ntime, 0)
        self.assertEqual(nspace, 5)

    def test_replace_vertex(self):
        st = Spacetime()
        simplex, _ = st.createSimplex((1, 2))
        facets1 = simplex.getFacets()
        self.assertEqual(len(simplex.getVertices()), 3)
        self.assertEqual(len(simplex.getEdges()), 3)

        v0 = simplex.getVertices()[0]
        v4 = st.createVertex(4, [0])
        v0id = v0.getId()

        simplex.replaceVertex(v0, v4)

        self.assertEqual(len(simplex.getVertices()), 3)
        self.assertEqual(len(simplex.getEdges()), 3)

        self.assertNotIn(v0, simplex.getVertices())
        self.assertNotIn(v0id, simplex.getVertexIdLookup())

        self.assertIn(v4, simplex.getVertices())
        self.assertIn(4, simplex.getVertexIdLookup())

        facets2 = simplex.getFacets()

        for f in facets1:
            f.validate()
        for f in facets2:
            f.validate()

    def test_facets_are_registered_to_vertices(self):
        st = Spacetime()
        s1, created = st.createSimplex((1, 4))
        for v in s1.getVertices():
            self.assertEqual(len(v.getSimplices()), 1)
            self.assertEqual(list(v.getSimplices())[0], s1)

        for v in s1.getVertices():
            self.assertEqual(len(v.getSimplices()), 1)

        facets = s1.getFacets()
        for v in s1.getVertices():
            # A 4-simplex has 4 facets + the 4-simplex itself = 5 simplices
            self.assertEqual(len(v.getSimplices()), 5)
            for facet in facets:
                if v in facet.getVertices():
                    self.assertIn(facet, v.getSimplices())

        for facet in facets:
            cofaces = facet.getCofaces()
            for coface in cofaces:
                self.assertTrue(coface.isCofaceTo(facet))


if __name__ == '__main__':
    unittest.main()
