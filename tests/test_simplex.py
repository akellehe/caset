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
            self.assertEqual(len(set([(e.getSource().getId(), e.getTarget().getId()) for e in face.getEdges()])), 6)
            self.assertEqual(len(face.getCofaces()), 1)
            if face.isTimelike():
                nTimelike += 1
                for timelikeFace in face.getFacets():
                    timelikeFace.validate()
                    self.assertTrue(timelikeFace.isTimelike())
                    self.assertEqual(len(timelikeFace.getVertices()), 3)
                    self.assertEqual(len(timelikeFace.getEdges()), 3)
                    self.assertEqual(len(set([(e.getSource().getId(), e.getTarget().getId()) for e in timelikeFace.getEdges()])), 3)
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

        vertices = left.getVerticesWithParityTo(right)
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
    
    @unittest.skip
    def test_pairty(self):
        simplex41 = self.spacetime.createSimplex((4, 1))
        f1, f2, f3, f4, f5 = simplex41.getFacets()

        self.assertEqual(len(f1.getVertices()), 4)

        # Disjoint faces have pairty flag=0
        self.assertEqual(f1.checkParity(f2), 0)

        v1, v2, v3, v4 = f1.getVertices()

        #The same face has pairty flag=1
        clone = Simplex([v1, v2, v3, v4])
        self.assertEqual(f1.checkParity(clone), 1)

        # A single vertex swap has pairty flag=-1
        oneSwap = Simplex([v2, v1, v3, v4])
        self.assertEqual(f1.checkParity(oneSwap), -1)

        # Two swaps has pairty flag=1
        twoSwaps = Simplex([v2, v1, v4, v3])
        self.assertEqual(f1.checkParity(twoSwaps), 1)

        for f in simplex41.getFacets():
            f.validate()

    def test_get_edges(self):
        simplex41 = self.spacetime.createSimplex((4, 1))

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
        self.assertEqual(e1[0], 2)
        self.assertEqual(e1[1], 3)

        # 2>3
        self.assertTrue(e2[0], 2)
        self.assertTrue(e2[1], 4)

        # 1>3
        self.assertTrue(e3[0], 2)
        self.assertTrue(e3[1], 5)

        # 3>4
        self.assertTrue(e4[0], 3)
        self.assertTrue(e4[1], 4)

        # 2>4
        self.assertTrue(e5[0], 3)
        self.assertTrue(e5[1], 5)

        # 1>4
        self.assertTrue(e6[0], 4)
        self.assertTrue(e6[1], 5)

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

        vertices = left.getVerticesWithParityTo(right)
        self.assertEqual(len(vertices), 4)

    def test_get_verticies_with_pairty_to2D(self):
        st = Spacetime()

        simplex12 = st.createSimplex((1, 2))
        simplex21 = st.createSimplex((2, 1))

        facets12 = simplex12.getFacets()
        facets21 = simplex21.getFacets()

        vertices12 = facets12[1].getVerticesWithParityTo(facets21[0])
        self.assertEqual(len(vertices12), 2)

        vertices21 = facets21[0].getVerticesWithParityTo(facets12[1])
        self.assertEqual(len(vertices21), 2)

        for i, f12 in enumerate(facets12):
            for j, f21 in enumerate(facets21):
                if f12.isTimelike() or f21.isTimelike():
                    continue
                v = f12.getVerticesWithParityTo(f21)
                if not v:
                    breakpoint()
                    print(i, j)
                self.assertEqual(len(v), 2)

        for f12 in reversed(facets12):
            for f21 in facets21:
                if f12.isTimelike() or f21.isTimelike():
                    continue
                v = f12.getVerticesWithParityTo(f21)
                self.assertEqual(len(v), 2)

        for f12 in facets12:
            for f21 in reversed(facets21):
                if f12.isTimelike() or f21.isTimelike():
                    continue
                v = f12.getVerticesWithParityTo(f21)
                self.assertEqual(len(v), 2)

        for f12 in reversed(facets12):
            for f21 in reversed(facets21):
                if f12.isTimelike() or f21.isTimelike():
                    continue
                v = f12.getVerticesWithParityTo(f21)
                self.assertEqual(len(v), 2)

    def test_attach_at_vertex(self):
        st = Spacetime()
        unattached = st.createSimplex((1, 2))
        attached = st.createSimplex((2, 1))

        facets1 = unattached.getFacets()
        self.assertEqual(len(unattached.getVertices()), 3)
        self.assertEqual(len(unattached.getEdges()), 3)

        unattachedVertex = unattached.getVertices()[0]
        for vertex in attached.getVertices():
            if (vertex.getTime() == unattachedVertex.getTime()):
                attachedVertex = vertex
                break

        st.attachAtVertex(unattached, attached, unattachedVertex, attachedVertex)

        self.assertEqual(len(unattached.getVertices()), 3)
        self.assertEqual(len(unattached.getEdges()), 3)  # replace vertex only replaces it in the unattached, there is no assignment of edges to the new vertex.

        self.assertNotIn(unattachedVertex, unattached.getVertices())
        self.assertNotIn(unattachedVertex.getId(), unattached.getVertexIdLookup())

        self.assertIn(attachedVertex, unattached.getVertices())
        self.assertIn(attachedVertex.getId(), unattached.getVertexIdLookup())

        for facet in unattached.getFacets():
            self.assertNotIn(unattachedVertex.getId(), [v.getId() for v in facet.getVertices()])

        for coface in unattached.getCofaces():
            self.assertNotIn(unattachedVertex.getId(), [v.getId() for v in coface.getVertices()])

        self.assertEqual(len(unattachedVertex.getSimplices()), 0)
        self.assertEqual(len(unattachedVertex.getEdges()), 0)

        for simplex in st.getSimplices():
            self.assertNotIn(unattachedVertex.getId(), [v.getId() for v in simplex.getVertices()])

        for edge in st.getEdgeList().toVector():
            self.assertNotEqual(edge.getSource(), unattachedVertex)
            self.assertNotEqual(edge.getTarget(), unattachedVertex)

        self.assertIsNone(st.getVertexList().get(unattachedVertex.getId()))

    def test_attaching_vertex_on_a_face_replaces_it_on_the_coface(self):
        st = Spacetime()
        unattached = st.createSimplex((1, 2))
        attached = st.createSimplex((2, 1))

        facet = unattached.getFacets()[0]
        unattachedVertex = facet.getVertices()[0]
        for vertex in attached.getVertices():
            if (vertex.getTime() == unattachedVertex.getTime()):
                attachedVertex = vertex

        self.assertEqual(len(unattached.getVertices()), 3)
        self.assertEqual(len(unattached.getEdges()), 3)

        self.assertEqual(len(facet.getVertices()), 2)
        self.assertEqual(len(facet.getEdges()), 1)

        st.attachAtVertex(unattached, attached, unattachedVertex, attachedVertex)

        self.assertEqual(len(unattached.getVertices()), 3)
        self.assertEqual(len(unattached.getEdges()), 3)

        self.assertNotIn(unattachedVertex, unattached.getVertices())
        self.assertNotIn(unattachedVertex.getId(), unattached.getVertexIdLookup())

        self.assertIn(attachedVertex, unattached.getVertices())
        self.assertIn(attachedVertex.getId(), unattached.getVertexIdLookup())

        self.assertNotIn(unattachedVertex, unattached.getVertices())
        self.assertNotIn(unattachedVertex.getId(), unattached.getVertexIdLookup())

        self.assertIn(attachedVertex, unattached.getVertices())
        self.assertIn(attachedVertex.getId(), unattached.getVertexIdLookup())

        facet.validate()
        for i, f in enumerate(unattached.getFacets()):
            print('validating', i)
            f.validate()

    def test_replace_vertex_on_a_coface_replaces_it_on_the_facets(self):
        st = Spacetime()
        unattached = st.createSimplex((1, 2))
        attached = st.createSimplex((2, 1))

        facet = unattached.getFacets()[0]
        unattachedVertex = facet.getVertices()[0]
        for vertex in attached.getVertices():
            if (vertex.getTime() == unattachedVertex.getTime()):
                attachedVertex = vertex

        self.assertEqual(len(unattached.getVertices()), 3)
        self.assertEqual(len(unattached.getEdges()), 3)

        self.assertEqual(len(facet.getVertices()), 2)
        self.assertEqual(len(facet.getEdges()), 1)

        st.attachAtVertex(unattached, attached, unattachedVertex, attachedVertex)

        self.assertEqual(len(unattached.getVertices()), 3)
        self.assertEqual(len(unattached.getEdges()), 3)

        self.assertNotIn(unattachedVertex, facet.getVertices())
        self.assertNotIn(unattachedVertex.getId(), facet.getVertexIdLookup())

        self.assertIn(attachedVertex, facet.getVertices())
        self.assertIn(attachedVertex.getId(), facet.getVertexIdLookup())

        self.assertNotIn(unattachedVertex, unattached.getVertices())
        self.assertNotIn(unattachedVertex.getId(), unattached.getVertexIdLookup())

        self.assertIn(attachedVertex, unattached.getVertices())
        self.assertIn(attachedVertex.getId(), unattached.getVertexIdLookup())

        facet.validate()
        for i, f in enumerate(unattached.getFacets()):
            print('validating', i)
            f.validate()

    def test_replace_vertex(self):
        """
        This method does not change edge assignments!
        """
        st = Spacetime()
        s12 = st.createSimplex((1, 2))
        s21 = st.createSimplex((2, 1))

        v12 = s12.getVertices()[0]
        v21 = s21.getVertices()[0]
        s12.replaceVertex(v12, v21)

        self.assertEqual(s21.getVertices()[0], s12.getVertices()[0])

    def test_attach_at_edge(self):
        """
        Test attaching two simplices at an edge (two adjacent compatible vertices).
        Vertices are compatible when they are neighbors and have the same getTime() return value.
        """
        st = Spacetime()
        unattached = st.createSimplex((1, 2))
        attached = st.createSimplex((1, 2))

        # Find an edge on the unattached simplex
        unattachedEdge = unattached.getEdges()[0]
        unattachedV1 = unattachedEdge.getSource()
        unattachedV2 = unattachedEdge.getTarget()
        unattachedT1 = unattachedV1.getTime()
        unattachedT2 = unattachedV2.getTime()

        # Find a compatible edge on the attached simplex
        # Compatible means: edge endpoints have matching times
        attachedV1 = None
        attachedV2 = None
        for edge in attached.getEdges():
            source = edge.getSource()
            target = edge.getTarget()
            sourceTime = source.getTime()
            targetTime = target.getTime()
            print(source.getId(), sourceTime, target.getId(), targetTime) # , unattachedT1, unattachedT2)

            # Check if this edge is compatible (times match in either order)
            if ((sourceTime == unattachedT1 and targetTime == unattachedT2) or
                (sourceTime == unattachedT2 and targetTime == unattachedT1)):
                # Found a compatible edge
                if sourceTime == unattachedT1:
                    attachedV1 = source
                    attachedV2 = target
                else:
                    attachedV1 = target
                    attachedV2 = source
                break

        # Verify we found compatible vertices
        self.assertIsNotNone(attachedV1, "Could not find compatible edge on attached simplex")
        self.assertIsNotNone(attachedV2, "Could not find compatible edge on attached simplex")
        self.assertEqual(attachedV1.getTime(), unattachedV1.getTime())
        self.assertEqual(attachedV2.getTime(), unattachedV2.getTime())

        # Attach at the first vertex
        st.attachAtVertex(unattached, attached, unattachedV1, attachedV1)

        # Attach at the second vertex
        st.attachAtVertex(unattached, attached, unattachedV2, attachedV2)

        # Verify both vertices were replaced in the unattached simplex
        self.assertNotIn(unattachedV1, unattached.getVertices())
        self.assertNotIn(unattachedV2, unattached.getVertices())
        self.assertNotIn(unattachedV1.getId(), unattached.getVertexIdLookup())
        self.assertNotIn(unattachedV2.getId(), unattached.getVertexIdLookup())

        self.assertIn(attachedV1, unattached.getVertices())
        self.assertIn(attachedV2, unattached.getVertices())
        self.assertIn(attachedV1.getId(), unattached.getVertexIdLookup())
        self.assertIn(attachedV2.getId(), unattached.getVertexIdLookup())

        # Verify the unattached vertices are no longer in use
        self.assertEqual(len(unattachedV1.getSimplices()), 0)
        self.assertEqual(len(unattachedV1.getEdges()), 0)
        self.assertEqual(len(unattachedV2.getSimplices()), 0)
        self.assertEqual(len(unattachedV2.getEdges()), 0)

        # Verify the vertices are removed from the vertex list
        self.assertIsNone(st.getVertexList().get(unattachedV1.getId()))
        self.assertIsNone(st.getVertexList().get(unattachedV2.getId()))

        # Verify no edges reference the unattached vertices
        for edge in st.getEdgeList().toVector():
            self.assertNotEqual(edge.getSource(), unattachedV1)
            self.assertNotEqual(edge.getTarget(), unattachedV1)
            self.assertNotEqual(edge.getSource(), unattachedV2)
            self.assertNotEqual(edge.getTarget(), unattachedV2)

        # Verify no simplices reference the unattached vertices
        for simplex in st.getSimplices():
            self.assertNotIn(unattachedV1.getId(), [v.getId() for v in simplex.getVertices()])
            self.assertNotIn(unattachedV2.getId(), [v.getId() for v in simplex.getVertices()])


if __name__ == '__main__':
    unittest.main()
