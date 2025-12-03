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
        s1 = self.spacetime.createSimplex((4, 1))
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
            self.assertEqual(len(set([(e.getSourceId(), e.getTargetId()) for e in face.getEdges()])), 6)
            self.assertEqual(len(face.getCofaces()), 1)
            if face.isTimelike():
                nTimelike += 1
                for timelikeFace in face.getFacets():
                    timelikeFace.validate()
                    self.assertTrue(timelikeFace.isTimelike())
                    self.assertEqual(len(timelikeFace.getVertices()), 3)
                    self.assertEqual(len(timelikeFace.getEdges()), 3)
                    self.assertEqual(len(set([(e.getSourceId(), e.getTargetId()) for e in timelikeFace.getEdges()])), 3)
                    self.assertEqual(len(timelikeFace.getCofaces()), 1)

        self.assertEqual(nTimelike, 1)

    def test_get_vertices_with_pairty_to(self):
        s1 = self.spacetime.createSimplex((4, 1))
        facets41 = s1.getFacets()
        self.assertEqual(len(facets41), 5)
        s2 = self.spacetime.createSimplex((3, 2))
        facets32 = s2.getFacets()
        self.assertEqual(len(facets32), 5)

        left, right = None, None
        print("41 Facets ------------------------------------")
        for face41 in facets41:
            if face41.getOrientation().numeric() == (3, 1):
                left = face41
            face41.validate()
            print(face41, face41.getOrientation().numeric())
        print("32 Facets ------------------------------------")
        for face32 in facets32:
            if face32.getOrientation().numeric() == (3, 1):
                right = face32
            face32.validate()
            print(face32, face32.getOrientation().numeric())

        vertices = left.getVerticesWithPairtyTo(right)
        self.assertEqual(len(vertices), 4)

    def test_creating_oriented_simplices(self):
        ti, tf = (4, 1)
        s1 = self.spacetime.createSimplex((ti, tf))
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
        s2 = self.spacetime.createSimplex((ti, tf))
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

    def setUp(self):
        self.spacetime = Spacetime()

    def test_pairty(self):
        simplex41 = self.spacetime.createSimplex((4, 1))
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
        simplex41 = self.spacetime.createSimplex((4, 1))

        f1, f2, f3, f4, f5 = simplex41.getFacets()
        v1, v2, v3, v4 = sorted(v.getId() for v in f1.getVertices())
        e1, e2, e3, e4, e5, e6 = sorted([(e.getSourceId(), e.getTargetId()) for e in f1.getEdges()])

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
        self.assertTrue(e2[0], 1)
        self.assertTrue(e2[1], 3)

        # 1>3
        self.assertTrue(e3[0], 1)
        self.assertTrue(e3[1], 4)

        # 3>4
        self.assertTrue(e4[0], 2)
        self.assertTrue(e4[1], 3)

        # 2>4
        self.assertTrue(e5[0], 2)
        self.assertTrue(e5[1], 4)

        # 1>4
        self.assertTrue(e6[0], 3)
        self.assertTrue(e6[1], 4)

    def test_get_verticies_with_pairty_to4D(self):
        simplex41 = self.spacetime.createSimplex((4, 1))
        simplex32 = self.spacetime.createSimplex((3, 2))

        f41_1, f41_2, f41_3, f41_4, f41_5 = simplex41.getFacets()
        f32_1, f32_2, f32_3, f32_4, f32_5 = simplex32.getFacets()

        left = None
        right = None

        for face41 in simplex41.getFacets():
            if face41.getOrientation().numeric() == (3, 1):
                left = face41

        for face32 in simplex32.getFacets():
            if face32.getOrientation().numeric() == (3, 1):
                right = face32

        vertices = left.getVerticesWithPairtyTo(right)
        self.assertEqual(len(vertices), 4)

    def test_get_verticies_with_pairty_to2D(self):
        st = Spacetime()

        simplex12 = st.createSimplex((1, 2))
        simplex21 = st.createSimplex((2, 1))

        facets12 = simplex12.getFacets()
        facets21 = simplex21.getFacets()

        vertices12 = facets12[1].getVerticesWithPairtyTo(facets21[0])
        self.assertEqual(len(vertices12), 2)

        vertices21 = facets21[0].getVerticesWithPairtyTo(facets12[1])
        self.assertEqual(len(vertices21), 2)

        for i, f12 in enumerate(facets12):
            for j, f21 in enumerate(facets21):
                if f12.isTimelike() or f21.isTimelike():
                    continue
                v = f12.getVerticesWithPairtyTo(f21)
                if not v:
                    breakpoint()
                    print(i, j)
                self.assertEqual(len(v), 2)

        for f12 in reversed(facets12):
            for f21 in facets21:
                if f12.isTimelike() or f21.isTimelike():
                    continue
                v = f12.getVerticesWithPairtyTo(f21)
                self.assertEqual(len(v), 2)

        for f12 in facets12:
            for f21 in reversed(facets21):
                if f12.isTimelike() or f21.isTimelike():
                    continue
                v = f12.getVerticesWithPairtyTo(f21)
                self.assertEqual(len(v), 2)

        for f12 in reversed(facets12):
            for f21 in reversed(facets21):
                if f12.isTimelike() or f21.isTimelike():
                    continue
                v = f12.getVerticesWithPairtyTo(f21)
                self.assertEqual(len(v), 2)

    def test_replace_vertex(self):
        st = Spacetime()
        simplex = st.createSimplex((1, 2))
        facets1 = simplex.getFacets()
        self.assertEqual(len(simplex.getVertices()), 3)
        self.assertEqual(len(simplex.getEdges()), 3)

        v0 = simplex.getVertices()[0]
        v = st.createVertex(4, [0])
        v0id = v0.getId()

        simplex.attach(v0, v, st.getEdgeList(), st.getVertexList())

        self.assertEqual(len(simplex.getVertices()), 3)
        self.assertEqual(len(simplex.getEdges()), 3)  # replace vertex only replaces it in the simplex, there is no assignment of edges to the new vertex.

        self.assertNotIn(v0, simplex.getVertices())
        self.assertNotIn(v0id, simplex.getVertexIdLookup())

        self.assertIn(v, simplex.getVertices())
        self.assertIn(4, simplex.getVertexIdLookup())

        facets2 = simplex.getFacets()

        for f in facets1:
            f.validate()
        for f in facets2:
            f.validate()

    def test_replace_vertex_on_a_face_replaces_it_on_the_coface(self):
        st = Spacetime()
        simplex = st.createSimplex((1, 2))
        v1, v2, v3 = simplex.getVertices()
        v4 = st.createVertex(4, [0])

        facet = simplex.getFacets()[0]

        self.assertEqual(len(simplex.getVertices()), 3)
        self.assertEqual(len(simplex.getEdges()), 3)

        self.assertEqual(len(facet.getVertices()), 2)
        self.assertEqual(len(facet.getEdges()), 1)

        for f in simplex.getFacets():
            f.validate()

        facet.validate()
        print("Replacing ", v2.getId(), "with ", v4.getId())
        facet.attach(v2, v4, st.getEdgeList(), st.getVertexList())
        facet.validate()

        # When we replace a vertex/edge on one facet; the edge AND vertex gets replaced there. on other facets, though,
        # either the edge is not being rewritten (it's stored by value and not reference?) or the vertex is not being
        # replaced, so the edge still appears to point to the incorrect vertex. The latter here is more likely.
        #
        # We need to ensure that in every facet this edge/vertex appears it's rewritten. So go up to the coface, then
        # down to each facet.
        self.assertIn(facet, simplex.getFacets())
        for f in simplex.getFacets():
            f.validate()

        self.assertEqual(len(simplex.getVertices()), 3)
        self.assertEqual(len(simplex.getEdges()), 3)

        self.assertNotIn(v2, facet.getVertices())
        self.assertNotIn(1, facet.getVertexIdLookup())

        self.assertIn(v4, facet.getVertices())
        self.assertIn(v4.getId(), facet.getVertexIdLookup())

        self.assertNotIn(v2, simplex.getVertices())
        self.assertNotIn(v2.getId(), simplex.getVertexIdLookup())

        self.assertIn(v4, simplex.getVertices())
        self.assertIn(v4.getId(), simplex.getVertexIdLookup())

        for i, f in enumerate(simplex.getFacets()):
            print('validating', i)
            f.validate()

    def test_replace_vertex_on_a_coface_replaces_it_on_the_facets(self):
        st = Spacetime()
        simplex = st.createSimplex((1, 2))

        facet = simplex.getFacets()[0]
        v1, v2, v3 = simplex.getVertices()
        v4 = st.createVertex(4, [0])

        self.assertEqual(len(simplex.getVertices()), 3)
        self.assertEqual(len(simplex.getEdges()), 3)

        self.assertEqual(len(facet.getVertices()), 2)
        self.assertEqual(len(facet.getEdges()), 1)

        simplex.attach(v2, v4, st.getEdgeList(), st.getVertexList())

        self.assertEqual(len(simplex.getVertices()), 3)
        self.assertEqual(len(simplex.getEdges()), 3)

        self.assertNotIn(v2, facet.getVertices())
        self.assertNotIn(v2.getId(), facet.getVertexIdLookup())

        self.assertIn(v4, facet.getVertices())
        self.assertIn(v4.getId(), facet.getVertexIdLookup())

        self.assertNotIn(v2, simplex.getVertices())
        self.assertNotIn(v2.getId(), simplex.getVertexIdLookup())

        self.assertIn(v4, simplex.getVertices())
        self.assertIn(v4.getId(), simplex.getVertexIdLookup())

        facet.validate()
        for i, f in enumerate(simplex.getFacets()):
            print('validating', i)
            f.validate()

if __name__ == '__main__':
    unittest.main()
