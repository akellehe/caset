# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

import gc
import unittest

import tessera
from tessera import Spacetime, Edge, Vertex
import cmath

class TestSpacetime(unittest.TestCase):

    def test_create_vertex(self):
        st = Spacetime()
        v1 = st.createVertex(1)
        v2 = st.createVertex(2)

        self.assertEqual(v1.getId(), 1)
        self.assertEqual(v2.getId(), 2)

        self.assertNotEqual(v1, v2)

    def test_create_edge(self):
        st = Spacetime()
        v1 = st.createVertex(1)
        v2 = st.createVertex(2)
        v3 = st.createVertex(3)

        self.assertEqual(v1.getId(), 1)
        self.assertEqual(v2.getId(), 2)
        self.assertEqual(v3.getId(), 3)

        e1 = st.createEdge(v1, v2)
        e2 = st.createEdge(v2, v3)

        self.assertNotEqual(v1, v2)
        self.assertNotEqual(v2, v3)
        self.assertNotEqual(e1, e2)

        self.assertEqual(e1.getSource().getId(), v1.getId())
        self.assertEqual(e1.getTarget().getId(), v2.getId())
        self.assertEqual(e2.getSource().getId(), v2.getId())
        self.assertEqual(e2.getTarget().getId(), v3.getId())

    def test_create_simplex(self):
        st = Spacetime()
        simplex, _ = st.createSimplex((2, 3))
        self.assertEqual(len(simplex.getVertices()), 5)
        edges = simplex.getEdges()
        self.assertEqual(len(edges), 10)

        v1, v2, v3, v4, v5 = simplex.getVertices()
        a, b, c, d = v1.getOutEdges()
        self.assertEqual(a.getSource().getId(), v1.getId())
        self.assertEqual(b.getSource().getId(), v1.getId())
        self.assertEqual(c.getSource().getId(), v1.getId())
        self.assertEqual(d.getSource().getId(), v1.getId())
        self.assertEqual(len(v1.getEdges()), 4)

        a, b, c = v2.getOutEdges()
        self.assertEqual(a.getSource().getId(), v2.getId())
        self.assertEqual(b.getSource().getId(), v2.getId())
        self.assertEqual(c.getSource().getId(), v2.getId())
        self.assertEqual(len(v2.getEdges()), 4)
        a, = v2.getInEdges()
        self.assertEqual(a.getTarget().getId(), v2.getId())

        a, b = v3.getOutEdges()
        c, d = v3.getInEdges()
        self.assertEqual(a.getSource().getId(), v3.getId())
        self.assertEqual(b.getSource().getId(), v3.getId())
        self.assertEqual(c.getTarget().getId(), v3.getId())
        self.assertEqual(d.getTarget().getId(), v3.getId())
        self.assertEqual(len(v3.getEdges()), 4)

        a, = v4.getOutEdges()
        b, c, d = v4.getInEdges()
        self.assertEqual(a.getSource().getId(), v4.getId())
        self.assertEqual(b.getTarget().getId(), v4.getId())
        self.assertEqual(c.getTarget().getId(), v4.getId())
        self.assertEqual(d.getTarget().getId(), v4.getId())
        self.assertEqual(len(v4.getEdges()), 4)

        a, b, c, d = v5.getInEdges()
        self.assertEqual(a.getTarget().getId(), v5.getId())
        self.assertEqual(b.getTarget().getId(), v5.getId())
        self.assertEqual(c.getTarget().getId(), v5.getId())
        self.assertEqual(d.getTarget().getId(), v5.getId())
        self.assertEqual(len(v5.getEdges()), 4)

        self.assertEqual(len(st.getSimplicesWithOrientation((2, 3))), 1)
        self.assertEqual(len(st.getSimplicesWithOrientation((1, 1))), 0)

        simplex2, _ = st.createSimplex((2, 3))
        self.assertEqual(len(st.getSimplicesWithOrientation((2, 3))), 2)
        self.assertEqual(len(st.getSimplicesWithOrientation((1, 1))), 0)

    @unittest.skip
    def test_euclidean_embedding(self):
        st = Spacetime()
        simplex14, _ = st.createSimplex((1, 4))
        simplex23, _ = st.createSimplex((2, 3))
        st.embedEuclidean()
        vertices = st.getVertexList().toVector()

    def test_attaching_faces4D(self):
        st = Spacetime()

        firstVertexList = st.getVertexList()
        firstEdgeList = st.getEdgeList()

        simplex14, _ = st.createSimplex((1, 4))
        simplex23, _ = st.createSimplex((2, 3))

        self.assertEqual(len(simplex14.getVertices()), 5)
        self.assertEqual(len(simplex14.getEdges()), 10)

        self.assertEqual(len(simplex23.getVertices()), 5)
        self.assertEqual(len(simplex23.getEdges()), 10)

        allVertices = [v.getId() for v in simplex14.getVertices() + simplex23.getVertices()]
        self.assertEqual(len(allVertices), len(set(allVertices)))

        allEdges = [(e.getSource().getId(), e.getTarget().getId()) for e in [_ for _ in simplex14.getEdges()] + [_ for _ in simplex23.getEdges()]]
        self.assertEqual(len(simplex14.getEdges()), len(set(simplex14.getEdges())))
        self.assertEqual(len(simplex23.getEdges()), len(set(simplex23.getEdges())))

        edges23 = {(e.getSource().getId(), e.getTarget().getId()) for e in simplex23.getEdges()}
        edges14 = {(e.getSource().getId(), e.getTarget().getId()) for e in simplex14.getEdges()}

        self.assertTrue(edges23.isdisjoint(edges14))
        self.assertEqual(len(allEdges), len(set(allEdges)))

        totalVerticesBefore = len(st.getVertexList().toVector())
        self.assertEqual(totalVerticesBefore, 10)

        totalEdgesBefore = len(st.getEdgeList().toVector())
        self.assertEqual(totalEdgesBefore, 20)

        for edge in firstEdgeList.toVector():
            source = firstVertexList.get(edge.getSource().getId())
            target = firstVertexList.get(edge.getTarget().getId())
            self.assertIsNotNone(source)
            self.assertIsNotNone(target)

        left, right = None, None
        facets14 = simplex14.getFacets()
        ntime, nspace = 0, 0
        for facet in facets14:
            self.assertTrue(facet.isInitialized())
            if facet.isSpatial():
                self.assertTrue(facet.getOrientation().numeric()[0] == 0 or facet.getOrientation().numeric()[1] == 0)
                ntime += 1
                ti, tf = float('inf'), float('-inf')
                for v in facet.getVertices():
                    ti = min(ti, v.getTime())
                    tf = max(tf, v.getTime())
                self.assertEqual(ti, tf)
            else:
                self.assertNotEqual(facet.getOrientation().numeric()[0], 0)
                self.assertNotEqual(facet.getOrientation().numeric()[1], 0)
                nspace += 1
                ti, tf = float('inf'), float('-inf')
                for v in facet.getVertices():
                    ti = min(ti, v.getTime())
                    tf = max(tf, v.getTime())
                self.assertNotEqual(ti, tf)

            if facet.getOrientation().numeric() == (1, 3):
                left = facet

        self.assertEqual(ntime, 1)
        self.assertEqual(nspace, 4)
        self.assertIsNotNone(left)

        facets23 = simplex23.getFacets()
        nspace = 0
        ntime = 0
        for face in facets23:
            if face.isSpatial():
                ntime += 1
            else:
                nspace += 1
            if face.getOrientation().numeric() == (1, 3):
                right = face

        self.assertEqual(nspace, 5)
        self.assertEqual(ntime, 0)

        self.assertIsNotNone(left)
        self.assertIsNotNone(right)

        totalVerticesBefore = st.getVertexList().toVector()
        totalEdgesBefore = st.getEdgeList().toVector()

        self.assertEqual(len(totalVerticesBefore), 10)
        self.assertEqual(len(totalEdgesBefore), 20)

        leftVerticesBefore = [v.getId() for v in left.getVertices()]
        self.assertEqual(len(leftVerticesBefore), 4)
        leftEdgesBefore = [(e.getSource().getId(), e.getTarget().getId()) for e in left.getEdges()]
        self.assertEqual(len(leftEdgesBefore), 6)

        rightVerticesBefore = [v.getId() for v in right.getVertices()]
        self.assertEqual(len(rightVerticesBefore), 4)
        rightEdgesBefore = [(e.getSource().getId(), e.getTarget().getId()) for e in right.getEdges()]
        self.assertEqual(len(rightEdgesBefore), 6)

    def test_we_get_connected_components_when_constructing_from_primitives(self):
        st = Spacetime()

        vertices = []
        for i in range(10):
            vertices.append(st.createVertex(i))

        edges = []
        for i in range(0, 9, 2):
            edges.append(st.createEdge(vertices[i], vertices[i+1]))

        components = st.getConnectedComponents()
        self.assertEqual(len(components), 5)

    def test_we_get_connected_components_when_constructing_from_simplexes(self):
        st = Spacetime()

        st.createSimplex((1, 4))
        st.createSimplex((2, 3))

        components = st.getConnectedComponents()
        self.assertEqual(len(components), 2)


    def test_vertex_stores_all_simplices_in_which_it_resides(self):
        st = Spacetime()
        s14, _ = st.createSimplex((1, 4))
        k3facets = s14.getFacets()
        for k3facet in k3facets:
            for k2facet in k3facet.getFacets():
                for k1facet in k2facet.getFacets():
                    for vertex in k1facet.getVertices():
                        self.assertIn(vertex, k1facet.getVertices())
                        self.assertIn(vertex, k2facet.getVertices())
                        self.assertIn(vertex, k3facet.getVertices())
                        self.assertIn(k1facet, vertex.getSimplices())
                        self.assertIn(k2facet, vertex.getSimplices())
                        self.assertIn(k3facet, vertex.getSimplices())
                        for coface in k1facet.getCofaces():
                            self.assertIn(coface, vertex.getSimplices())
                        for coface in k2facet.getCofaces():
                            self.assertIn(coface, vertex.getSimplices())
                        for coface in k3facet.getCofaces():
                            self.assertIn(coface, vertex.getSimplices())


class TestExternalSimplicesNonCDT(unittest.TestCase):
    """getExternalSimplices() on hand-built (non-CDT) complexes.

    Regression: facets are materialized lazily by Simplex.getFacets(), which
    registers them back into the spacetime's simplex vector. A from-scratch
    complex has no facets until something asks for them, so getExternalSimplices
    used to (a) grow the vector it was iterating — a segfault — and (b) read
    incomplete coface counts mid-pass. CDT-built complexes never hit this
    because gluing materializes facets up front.
    """

    def _tetra_boundary(self):
        """S^2 = boundary of a tetrahedron: 4 vertices, 4 triangles, closed."""
        import itertools
        st = Spacetime()
        V = [st.createVertex(i, [0.0]) for i in range(4)]
        for a, b in itertools.combinations(range(4), 2):
            st.createEdge(V[a], V[b], complex(1.0))
        tris = [st.createSimplex([V[i] for i in c])[0]
                for c in itertools.combinations(range(4), 3)]
        return st, V, tris

    def test_closed_surface_has_no_external_top_simplices(self):
        # Every edge of S^2 is shared by exactly two triangles, so no triangle
        # has a boundary facet. (Must not segfault.)
        st, _V, tris = self._tetra_boundary()
        ext = st.getExternalSimplices()
        triangles = [s for s in ext if len(s.getVertices()) == 3]
        self.assertEqual(triangles, [],
                         "closed S^2 should have no boundary triangles")

    def test_open_complex_reports_boundary(self):
        # Drop one triangle: the three edges it covered now have a single
        # coface, so the three remaining triangles each gain a boundary edge.
        st, _V, tris = self._tetra_boundary()
        st.removeSimplex(tris[0])
        ext = st.getExternalSimplices()
        triangles = [s for s in ext if len(s.getVertices()) == 3]
        self.assertEqual(len(triangles), 3,
                         "each remaining triangle should touch the new boundary")


class TestCreateSimplexVertexCap(unittest.TestCase):
    """createSimplex must reject simplices beyond the Fingerprint capacity.

    The Fingerprint stores at most kMax = 8 vertex IDs; past that it truncates,
    so a >8-vertex simplex would collide with another and be silently dropped
    (returned created=true but never registered). createSimplex now raises
    instead of corrupting the complex (issue #77).
    """

    KMAX = 8  # mesh/Fingerprint.h

    def _verts(self, st, n):
        return [st.createVertex(i) for i in range(n)]

    def test_max_capacity_simplex_registers(self):
        # A kMax-vertex simplex (dimension 7) is the largest that round-trips.
        st = Spacetime()
        s, created = st.createSimplex(self._verts(st, self.KMAX))
        self.assertTrue(created)
        self.assertEqual(len(s.getVertices()), self.KMAX)
        self.assertIn(s, st.getSimplices())

    def test_over_capacity_simplex_raises(self):
        st = Spacetime()
        verts = self._verts(st, self.KMAX + 1)  # 9 vertices
        with self.assertRaises(Exception):
            st.createSimplex(verts)

    def test_no_silent_drop_for_S8(self):
        # S^8 = ∂Δ^9 has ten 8-simplices (9 vertices each). Previously only 9 of
        # the 10 registered silently; now each over-capacity create raises, so
        # the complex is never partially built behind the caller's back.
        st = Spacetime()
        V = self._verts(st, 10)
        raised = 0
        for omit in range(10):
            verts = [V[i] for i in range(10) if i != omit]  # 9 vertices
            try:
                st.createSimplex(verts)
            except Exception:
                raised += 1
        self.assertEqual(raised, 10)
        self.assertEqual(len([s for s in st.getSimplices()
                              if len(s.getVertices()) == 9]), 0)


class TestHandleLifetime(unittest.TestCase):
    """Simplex/Vertex handles returned by the query methods point into the
    Spacetime's storage, so the Spacetime must stay alive while they are used.
    The bindings enforce this (keep_alive), so the handles remain valid even
    after the Spacetime variable goes out of scope. Each helper builds a
    Spacetime in a local, returns handles, and lets the local drop — without
    the keep_alive these accesses are use-after-free (segfault).
    """

    def _sphere_spacetime(self, n=2):
        sig = tessera.Signature(4, tessera.Lorentzian)
        metric = tessera.Metric(True, sig)
        st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                               tessera.PREFERRED, tessera.SimplexBoundarySphere(n))
        st.build()
        return st

    def test_getSimplices_outlives_spacetime_local(self):
        def handles():
            return self._sphere_spacetime(2).getSimplices()  # temporary Spacetime
        simplices = handles()
        gc.collect()
        # S^2 = boundary of a tetrahedron has 4 triangles; accessing them must
        # not touch freed storage.
        self.assertEqual(len(simplices), 4)
        self.assertTrue(all(len(s.getVertices()) == 3 for s in simplices))

    def test_getExternalSimplices_outlives_spacetime_local(self):
        def handles():
            sig = tessera.Signature(4, tessera.Lorentzian)
            metric = tessera.Metric(True, sig)
            st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                                   tessera.PREFERRED, tessera.SolidSimplex(4))
            st.build()
            return st.getExternalSimplices()
        external = handles()
        gc.collect()
        self.assertGreater(len(external), 0)
        for s in external:
            _ = s.getVertices()  # no crash

    def test_getRandomVertex_outlives_spacetime_local(self):
        v = self._sphere_spacetime(2).getRandomVertex()
        gc.collect()
        self.assertIsNotNone(v)
        _ = v.getId()  # no crash

    def test_getRandomTopSimplex_outlives_spacetime_local(self):
        def handle():
            sig = tessera.Signature(4, tessera.Lorentzian)
            metric = tessera.Metric(True, sig)
            st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                                   tessera.PREFERRED, tessera.Toroid())
            st.build(50)
            return st.getRandomTopSimplex()
        s = handle()
        gc.collect()
        self.assertEqual(len(s.getVertices()), 5)  # 4D top simplex


if __name__ == '__main__':
    unittest.main()
