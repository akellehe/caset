import unittest

from caset import Vertex, Simplex, Metric, Spacetime, Signature, SignatureType


class TestSimplex(unittest.TestCase):
    def setUp(self):
        self.spacetime = Spacetime()
        self.spacetime.setManual(False)

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
            self.assertEqual(len(face.getVertices()), 4)
            self.assertEqual(len(face.getEdges()), 6)
            self.assertEqual(len(set([(e.getSourceId(), e.getTargetId()) for e in face.getEdges()])), 6)
            self.assertEqual(len(face.getCofaces()), 1)
            if face.isTimelike():
                nTimelike += 1
                for timelikeFace in face.getFacets():
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
            print(face41, face41.getOrientation().numeric())
        print("32 Facets ------------------------------------")
        for face32 in facets32:
            if face32.getOrientation().numeric() == (3, 1):
                right = face32
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

    def test_get_edges(self):
        simplex41 = self.spacetime.createSimplex((4, 1))

        f1, f2, f3, f4, f5 = simplex41.getFacets()
        v1, v2, v3, v4 = f1.getVertices()

        e1, e2, e3, e4, e5, e6 = f1.getEdges()

        # 1>2
        self.assertTrue(e1.getSourceId() == v1.getId() or e1.getTargetId() == v1.getId())
        self.assertTrue(e1.getSourceId() == v2.getId() or e1.getTargetId() == v2.getId())

        # 2>3
        self.assertTrue(e2.getSourceId() == v2.getId() or e2.getTargetId() == v2.getId())
        self.assertTrue(e2.getSourceId() == v3.getId() or e2.getTargetId() == v3.getId())

        #1>3
        self.assertTrue(e3.getSourceId() == v3.getId() or e3.getTargetId() == v3.getId())
        self.assertTrue(e3.getSourceId() == v1.getId() or e3.getTargetId() == v1.getId())

        #3>4
        self.assertTrue(e4.getSourceId() == v3.getId() or e4.getTargetId() == v3.getId())
        self.assertTrue(e4.getSourceId() == v4.getId() or e4.getTargetId() == v4.getId())

        #2>4
        self.assertTrue(e5.getSourceId() == v2.getId() or e5.getTargetId() == v2.getId())
        self.assertTrue(e5.getSourceId() == v4.getId() or e5.getTargetId() == v4.getId())

        #1>4
        self.assertTrue(e6.getSourceId() == v1.getId() or e6.getTargetId() == v1.getId())
        self.assertTrue(e6.getSourceId() == v4.getId() or e6.getTargetId() == v4.getId())

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
        simplex21 = self.spacetime.createSimplex((2, 1))
        simplex12 = self.spacetime.createSimplex((1, 2))

        f21_1, f21_2, f21_3 = simplex21.getFacets()
        f12_1, f12_2, f12_3 = simplex12.getFacets()

        left = None
        right = None

        for face21 in simplex21.getFacets():
            if face21.getOrientation().numeric() == (1, 1):
                left = face21

        for face12 in simplex12.getFacets():
            if face12.getOrientation().numeric() == (1, 1):
                right = face12

        vertices = left.getVerticesWithPairtyTo(right)
        self.assertEqual(len(vertices), 2)

    def test_get_verticies_with_pairty_to2D(self):
        st = Spacetime()
        vertices = []
        for i in range(6):
            vertices.append(st.createVertex(i, [i % 2]))

        st.createEdge(vertices[0].getId(), vertices[1].getId())
        st.createEdge(vertices[1].getId(), vertices[2].getId())
        st.createEdge(vertices[2].getId(), vertices[0].getId())

        st.createEdge(vertices[3].getId(), vertices[4].getId())
        st.createEdge(vertices[4].getId(), vertices[5].getId())
        st.createEdge(vertices[5].getId(), vertices[3].getId())

        simplex12 = Simplex(vertices[0:3])
        simplex21 = Simplex(vertices[3:6])

        facets12 = simplex12.getFacets()
        facets21 = simplex21.getFacets()

        vertices12 = facets12[0].getVerticesWithPairtyTo(facets21[0])
        self.assertEqual(len(vertices12), 2)
        vertices21 = facets21[0].getVerticesWithPairtyTo(facets12[0])
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

    def test_remove_vertex(self):
        st = Spacetime()
        v1 = st.createVertex(0, [0])
        v2 = st.createVertex(1, [1])
        v3 = st.createVertex(2, [2])

        e1 = st.createEdge(v1.getId(), v2.getId())
        e2 = st.createEdge(v2.getId(), v3.getId())
        e3 = st.createEdge(v3.getId(), v1.getId())

        simplex = Simplex([v1, v2, v3])

        self.assertEqual(len(simplex.getVertices()), 3)
        self.assertEqual(len(simplex.getEdges()), 3)

        simplex.removeVertex(v2)
        simplex.removeEdge(e1)
        simplex.removeEdge(e2)

        self.assertEqual(len(simplex.getVertices()), 2)
        self.assertEqual(len(simplex.getEdges()), 1)


if __name__ == '__main__':
    unittest.main()
