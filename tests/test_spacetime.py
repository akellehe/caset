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

from caset import Spacetime, Edge, Vertex

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

        e1 = st.createEdge(v1.getId(), v2.getId())
        e2 = st.createEdge(v2.getId(), v3.getId())

        self.assertNotEqual(v1, v2)
        self.assertNotEqual(v2, v3)
        self.assertNotEqual(e1, e2)

        self.assertEqual(e1.getSourceId(), v1.getId())
        self.assertEqual(e1.getTargetId(), v2.getId())
        self.assertEqual(e2.getSourceId(), v2.getId())
        self.assertEqual(e2.getTargetId(), v3.getId())

    def test_create_simplex(self):
        st = Spacetime()
        simplex = st.createSimplex((2, 3))
        self.assertEqual(len(simplex.getVertices()), 5)
        edges = simplex.getEdges()
        self.assertEqual(len(edges), 10)

        v1, v2, v3, v4, v5 = simplex.getVertices()
        a, b, c, d = v1.getOutEdges()
        self.assertEqual(a.getSourceId(), v1.getId())
        self.assertEqual(b.getSourceId(), v1.getId())
        self.assertEqual(c.getSourceId(), v1.getId())
        self.assertEqual(d.getSourceId(), v1.getId())
        self.assertEqual(len(v1.getEdges()), 4)

        a, b, c = v2.getOutEdges()
        self.assertEqual(a.getSourceId(), v2.getId())
        self.assertEqual(b.getSourceId(), v2.getId())
        self.assertEqual(c.getSourceId(), v2.getId())
        self.assertEqual(len(v2.getEdges()), 4)
        a, = v2.getInEdges()
        self.assertEqual(a.getTargetId(), v2.getId())

        a, b = v3.getOutEdges()
        c, d = v3.getInEdges()
        self.assertEqual(a.getSourceId(), v3.getId())
        self.assertEqual(b.getSourceId(), v3.getId())
        self.assertEqual(c.getTargetId(), v3.getId())
        self.assertEqual(d.getTargetId(), v3.getId())
        self.assertEqual(len(v3.getEdges()), 4)

        a, = v4.getOutEdges()
        b, c, d = v4.getInEdges()
        self.assertEqual(a.getSourceId(), v4.getId())
        self.assertEqual(b.getTargetId(), v4.getId())
        self.assertEqual(c.getTargetId(), v4.getId())
        self.assertEqual(d.getTargetId(), v4.getId())
        self.assertEqual(len(v4.getEdges()), 4)

        a, b, c, d = v5.getInEdges()
        self.assertEqual(a.getTargetId(), v5.getId())
        self.assertEqual(b.getTargetId(), v5.getId())
        self.assertEqual(c.getTargetId(), v5.getId())
        self.assertEqual(d.getTargetId(), v5.getId())
        self.assertEqual(len(v5.getEdges()), 4)

        self.assertEqual(len(st.getSimplicesWithOrientation((2, 3))), 1)
        self.assertEqual(len(st.getSimplicesWithOrientation((1, 1))), 0)

        simplex2 = st.createSimplex((2, 3))
        self.assertEqual(len(st.getSimplicesWithOrientation((2, 3))), 2)
        self.assertEqual(len(st.getSimplicesWithOrientation((1, 1))), 0)

    @unittest.skip
    def test_euclidean_embedding(self):
        st = Spacetime()
        simplex14 = st.createSimplex((1, 4))
        simplex23 = st.createSimplex((2, 3))
        st.embedEuclidean()
        vertices = st.getVertexList().toVector()

    def test_attaching_faces4D(self):
        st = Spacetime()

        firstVertexList = st.getVertexList()
        firstEdgeList = st.getEdgeList()

        simplex14 = st.createSimplex((1, 4))
        simplex23 = st.createSimplex((2, 3))

        self.assertEqual(len(simplex14.getVertices()), 5)
        self.assertEqual(len(simplex14.getEdges()), 10)

        self.assertEqual(len(simplex23.getVertices()), 5)
        self.assertEqual(len(simplex23.getEdges()), 10)

        allVertices = [v.getId() for v in simplex14.getVertices() + simplex23.getVertices()]
        self.assertEqual(len(allVertices), len(set(allVertices)))

        allEdges = [(e.getSourceId(), e.getTargetId()) for e in simplex14.getEdges() + simplex23.getEdges()]
        self.assertEqual(len(simplex14.getEdges()), len(set(simplex14.getEdges())))
        self.assertEqual(len(simplex23.getEdges()), len(set(simplex23.getEdges())))

        edges23 = {(e.getSourceId(), e.getTargetId()) for e in simplex23.getEdges()}
        edges14 = {(e.getSourceId(), e.getTargetId()) for e in simplex14.getEdges()}

        self.assertTrue(edges23.isdisjoint(edges14))
        self.assertEqual(len(allEdges), len(set(allEdges)))

        totalVerticesBefore = len(st.getVertexList().toVector())
        self.assertEqual(totalVerticesBefore, 10)

        totalEdgesBefore = len(st.getEdgeList().toVector())
        self.assertEqual(totalEdgesBefore, 20)

        for edge in firstEdgeList.toVector():
            source = firstVertexList.get(edge.getSourceId())
            target = firstVertexList.get(edge.getTargetId())
            self.assertIsNotNone(source)
            self.assertIsNotNone(target)

        left, right = None, None
        facets14 = simplex14.getFacets()
        ntime, nspace = 0, 0
        for facet in facets14:
            if facet.isTimelike():
                ntime += 1
            else:
                nspace += 1
            if facet.getOrientation().numeric() == (1, 3):
                left = facet

        self.assertEqual(ntime, 1)
        self.assertEqual(nspace, 4)
        self.assertIsNotNone(left)

        facets23 = simplex23.getFacets()
        nspace = 0
        ntime = 0
        for face in facets23:
            if face.isTimelike():
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
        leftEdgesBefore = [(e.getSourceId(), e.getTargetId()) for e in left.getEdges()]
        self.assertEqual(len(leftEdgesBefore), 6)

        rightVerticesBefore = [v.getId() for v in right.getVertices()]
        self.assertEqual(len(rightVerticesBefore), 4)
        rightEdgesBefore = [(e.getSourceId(), e.getTargetId()) for e in right.getEdges()]
        self.assertEqual(len(rightEdgesBefore), 6)

        updated, succeeded = st.causallyAttachFaces(left, right)
        self.assertTrue(succeeded)
        for f in simplex23.getFacets(): f.validate()
        for f in simplex23.getFacets(): f.validate()

        v1, v2, v3, v4 = updated.getVertices()

        secondVertexList = st.getVertexList()
        secondEdgeList = st.getEdgeList()
        self.assertIs(firstVertexList, secondVertexList)
        self.assertIs(firstEdgeList, secondEdgeList)
        for edge in secondEdgeList.toVector():
            source = secondVertexList.get(edge.getSourceId())
            target = secondVertexList.get(edge.getTargetId())
            self.assertIsNotNone(source)
            self.assertIsNotNone(target)

        leftVerticesAfter = [v.getId() for v in left.getVertices()]
        self.assertEqual(len(leftVerticesAfter), 4)
        leftEdgesAfter = [(e.getSourceId(), e.getTargetId()) for e in left.getEdges()]
        self.assertEqual(len(leftEdgesAfter), 6)

        rightVerticesAfter = [v.getId() for v in right.getVertices()]
        self.assertEqual(len(rightVerticesAfter), 4)
        rightEdgesAfter = [(e.getSourceId(), e.getTargetId()) for e in right.getEdges()]
        self.assertEqual(len(rightEdgesAfter), 6)

        shared = {v for v in leftVerticesAfter} & {v for v in rightVerticesAfter}
        self.assertEqual(len(shared), 4)
        self.assertEqual(len(leftVerticesAfter), len(leftVerticesBefore))

        totalVerticesAfter = set([v.getId() for v in secondVertexList.toVector()])
        totalEdgesAfter = set([(e.getSourceId(), e.getTargetId()) for e in secondEdgeList.toVector()])

        self.assertEqual(len(totalVerticesAfter), len(totalVerticesBefore) - 4)
        self.assertEqual(len(totalEdgesAfter), len(totalEdgesBefore) - 6)


    def test_attaching_faces2D(self):
        st = Spacetime()

        firstVertexList = st.getVertexList()
        firstEdgeList = st.getEdgeList()

        simplex21 = st.createSimplex((2, 1))
        simplex12 = st.createSimplex((1, 2))

        self.assertEqual(len(simplex21.getVertices()), 3)
        self.assertEqual(len(simplex21.getEdges()), 3)

        self.assertEqual(len(simplex12.getVertices()), 3)
        self.assertEqual(len(simplex12.getEdges()), 3)

        allVertices = [v.getId() for v in simplex12.getVertices() + simplex21.getVertices()]
        self.assertEqual(len(allVertices), len(set(allVertices)))

        allEdges = [(e.getSourceId(), e.getTargetId()) for e in simplex12.getEdges() + simplex21.getEdges()]
        self.assertEqual(len(simplex12.getEdges()), len(set(simplex21.getEdges())))
        self.assertEqual(len(simplex12.getEdges()), len(set(simplex21.getEdges())))

        edges12 = {(e.getSourceId(), e.getTargetId()) for e in simplex12.getEdges()}
        edges21 = {(e.getSourceId(), e.getTargetId()) for e in simplex21.getEdges()}

        self.assertTrue(edges21.isdisjoint(edges12))
        self.assertEqual(len(allEdges), len(set(allEdges)))

        totalVerticesBefore = len(st.getVertexList().toVector())
        self.assertEqual(totalVerticesBefore, 6)

        totalEdgesBefore = len(st.getEdgeList().toVector())
        self.assertEqual(totalEdgesBefore, 6)

        for edge in firstEdgeList.toVector():
            source = firstVertexList.get(edge.getSourceId())
            target = firstVertexList.get(edge.getTargetId())
            self.assertIsNotNone(source)
            self.assertIsNotNone(target)

        left, right = None, None
        facets12 = simplex12.getFacets()
        ntime, nspace = 0, 0
        for facet in facets12:
            if facet.isTimelike():
                ntime += 1
            else:
                nspace += 1
            if facet.getOrientation().numeric() == (1, 1):
                left = facet

        self.assertEqual(ntime, 1)
        self.assertEqual(nspace, 2)
        self.assertIsNotNone(left)

        facets23 = simplex21.getFacets()
        nspace = 0
        ntime = 0
        for face in facets23:
            if face.isTimelike():
                ntime += 1
            else:
                nspace += 1
            if face.getOrientation().numeric() == (1, 1):
                right = face

        self.assertEqual(nspace, 2)
        self.assertEqual(ntime, 1)

        self.assertIsNotNone(left)
        self.assertIsNotNone(right)

        totalVerticesBefore = st.getVertexList().toVector()
        totalEdgesBefore = st.getEdgeList().toVector()

        self.assertEqual(len(totalVerticesBefore), 6)
        self.assertEqual(len(totalEdgesBefore), 6)

        leftVerticesBefore = [v.getId() for v in left.getVertices()]
        self.assertEqual(len(leftVerticesBefore), 2)
        leftEdgesBefore = [(e.getSourceId(), e.getTargetId()) for e in left.getEdges()]
        self.assertEqual(len(leftEdgesBefore), 1)

        rightVerticesBefore = [v.getId() for v in right.getVertices()]
        self.assertEqual(len(rightVerticesBefore), 2)
        rightEdgesBefore = [(e.getSourceId(), e.getTargetId()) for e in right.getEdges()]
        self.assertEqual(len(rightEdgesBefore), 1)

        left.validate()
        right.validate()
        updated, succeeded = st.causallyAttachFaces(left, right)
        left.validate()
        right.validate()
        self.assertTrue(succeeded)

        v1, v2 = updated.getVertices()

        secondVertexList = st.getVertexList()
        secondEdgeList = st.getEdgeList()
        self.assertIs(firstVertexList, secondVertexList)
        self.assertIs(firstEdgeList, secondEdgeList)
        for edge in secondEdgeList.toVector():
            source = secondVertexList.get(edge.getSourceId())
            target = secondVertexList.get(edge.getTargetId())
            self.assertIsNotNone(source)
            self.assertIsNotNone(target)

        leftVerticesAfter = [v.getId() for v in left.getVertices()]
        self.assertEqual(len(leftVerticesAfter), 2)
        leftEdgesAfter = [(e.getSourceId(), e.getTargetId()) for e in left.getEdges()]
        self.assertEqual(len(leftEdgesAfter), 1)

        rightVerticesAfter = [v.getId() for v in right.getVertices()]
        self.assertEqual(len(rightVerticesAfter), 2)
        rightEdgesAfter = [(e.getSourceId(), e.getTargetId()) for e in right.getEdges()]
        self.assertEqual(len(rightEdgesAfter), 1)

        shared = {v for v in leftVerticesAfter} & {v for v in rightVerticesAfter}
        self.assertEqual(len(shared), 2)
        self.assertEqual(len(leftVerticesAfter), len(leftVerticesBefore))

        totalVerticesAfter = set([v.getId() for v in secondVertexList.toVector()])
        totalEdgesAfter = set([(e.getSourceId(), e.getTargetId()) for e in secondEdgeList.toVector()])

        self.assertEqual(len(totalVerticesAfter), len(totalVerticesBefore) - 2)
        self.assertEqual(len(totalEdgesAfter), len(totalEdgesBefore) - 1)

    def test_we_get_connected_components_when_constructing_from_primitives(self):
        st = Spacetime()

        vertices = []
        for i in range(10):
            vertices.append(st.createVertex(i))

        edges = []
        for i in range(0, 9, 2):
            edges.append(st.createEdge(vertices[i].getId(), vertices[i+1].getId()))

        components = st.getConnectedComponents()
        self.assertEqual(len(components), 5)

    def test_we_get_connected_components_when_constructing_from_simplexes(self):
        st = Spacetime()

        st.createSimplex((1, 4))
        st.createSimplex((2, 3))

        components = st.getConnectedComponents()
        self.assertEqual(len(components), 2)

    def test_components_connect_once_glued(self):
        st = Spacetime()

        unattachedSimplex = st.createSimplex((1, 4))
        s23 = st.createSimplex((2, 3))

        components = st.getConnectedComponents()
        self.assertEqual(len(components), 2)

        unattachedSimplexFace, attachedSimplexFace = st.chooseSimplexFacesToGlue(unattachedSimplex)
        print('validating faces...')
        unattachedSimplexFace.validate()
        attachedSimplexFace.validate()
        unattachedVertices = [v for v in unattachedSimplexFace.getVertices()]
        attachedVertices = [v for v in attachedSimplexFace.getVertices()]

        st.attachAtVertices(unattachedSimplexFace, s23, [(u, a) for u, a in zip(unattachedVertices, attachedVertices)])

        components = st.getConnectedComponents()

        self.assertEqual(len(components), 1)

    def test_components_connect_once_causally_glued(self):
        st = Spacetime()

        s14 = st.createSimplex((1, 4))
        s23 = st.createSimplex((2, 3))

        components = st.getConnectedComponents()
        self.assertEqual(len(components), 2)

        leftFace, rightFace = st.chooseSimplexFacesToGlue(s14)
        updated, succeeded = st.causallyAttachFaces(leftFace, rightFace)
        self.assertTrue(succeeded)

        components = st.getConnectedComponents()

        self.assertEqual(len(components), 1)

    def test_move_in_edges_from_vertex(self):
        # This only moves edges within the vertex state. The edges themselves remain attached to the same simplices.
        st = Spacetime()

        toSimplex = st.createSimplex((1, 4))
        fromSimplex = st.createSimplex((2, 3))

        fromSimplexFace, toSimplexFace = st.chooseSimplexFacesToGlue(fromSimplex)

        fromVertex = fromSimplexFace.getVertices()[0]
        toVertex = toSimplexFace.getVertices()[0]

        orig_in_edges_before = [e for e in fromVertex.getInEdges()]
        dest_in_edges_before = [e for e in toVertex.getInEdges()]

        st.moveInEdgesFromVertex(fromVertex, toVertex)

        orig_in_edges_after = [e for e in fromVertex.getInEdges()]
        dest_in_edges_after = [e for e in toVertex.getInEdges()]

        self.assertEqual(orig_in_edges_before, dest_in_edges_after)
        self.assertEqual(orig_in_edges_after, dest_in_edges_before)

    def test_move_out_edges_from_vertex(self):
        st = Spacetime()

        toSimplex = st.createSimplex((1, 4))
        fromSimplex = st.createSimplex((2, 3))

        fromSimplexFace, toSimplexFace = st.chooseSimplexFacesToGlue(fromSimplex)

        fromVertex = fromSimplexFace.getVertices()[0]
        toVertex = toSimplexFace.getVertices()[0]

        orig_out_edges_before = [e for e in fromVertex.getOutEdges()]
        dest_out_edges_before = [e for e in toVertex.getOutEdges()]

        st.moveOutEdgesFromVertex(fromVertex, toVertex)

        orig_out_edges_after = [e for e in fromVertex.getOutEdges()]
        dest_out_edges_after = [e for e in toVertex.getOutEdges()]

        self.assertEqual(len(orig_out_edges_before), 3)
        self.assertEqual(len(orig_out_edges_after), 0)

        self.assertEqual(len(dest_out_edges_before), 4)
        self.assertEqual(len(dest_out_edges_after), 7)

        for edge in orig_out_edges_after:
            self.assertNotIn(edge, fromVertex.getEdges())

    def test_attach_at_vertices(self):
        st = Spacetime()

        attachedSimplex = st.createSimplex((1, 4))
        unattachedSimplex = st.createSimplex((2, 3))  # Unattached

        attachedSimplex.validate()
        unattachedSimplex.validate()

        unattachedSimplexFace, attachedSimplexFace = st.chooseSimplexFacesToGlue(unattachedSimplex)
        unattachedVertices = [v for v in unattachedSimplexFace.getVertices()]
        attachedVertices = [v for v in attachedSimplexFace.getVertices()]

        unattachedSimplexFace.validate()
        attachedSimplexFace.validate()

        st.attachAtVertices(unattachedSimplexFace, attachedSimplexFace, [(u, a) for u, a in zip(unattachedVertices, attachedVertices)])

        attachedSimplex.validate()
        unattachedSimplex.validate()
        unattachedSimplexFace.validate()
        attachedSimplexFace.validate()

        unattachedVertex = unattachedVertices[0]  # V6
        attachedVertex = attachedVertices[0]      # V0

        # 1. unattachedVertex (V6) is not being removed unattached unattachedSimplex OR unattachedSimplexFace, even though the degree=0
        self.assertEqual(unattachedVertex.degree(), 0)
        self.assertNotIn(unattachedVertex, unattachedSimplex.getVertices())
        self.assertNotIn(unattachedVertex, unattachedSimplexFace.getVertices())

        # 2. attachedVertex (V0) is not gaining enough new edges unattached unattachedVertex (V6)
        self.assertEqual(attachedVertex.degree(), 5)
        self.assertIn(attachedVertex, attachedSimplexFace.getVertices())
        self.assertIn(attachedVertex, attachedSimplex.getVertices())

        # 3. overall edge count is wrong after the moves (17): len(attachedSimplex.getEdges()) is 10, so is len(unattachedSimplex.getEdges())
        total_edges = attachedSimplex.getEdges() + unattachedSimplex.getEdges()

        for edge in attachedSimplex.getEdges():
            self.assertTrue(attachedSimplex.hasVertex(edge.getSourceId()))
            self.assertTrue(attachedSimplex.hasVertex(edge.getTargetId()))

        for edge in unattachedSimplex.getEdges():
            self.assertTrue(unattachedSimplex.hasVertex(edge.getSourceId()))
            self.assertTrue(unattachedSimplex.hasVertex(edge.getTargetId()))


        self.assertEqual(len(total_edges), 20)  # 6 shared/doubled.
        self.assertEqual(len(set(total_edges)), 14)
        shared_edges = set(attachedSimplex.getEdges()) & set(unattachedSimplex.getEdges())
        self.assertEqual(len(shared_edges), 6)
        self.assertEqual(len(set(total_edges)), len(st.getEdgeList().toVector()))

        # 4. unattachedSimplex still has edges containing unattachedVertex (V6) even though it should have been removed (6>7, 6>8, 6>9).
        for edge in unattachedSimplex.getEdges():
            self.assertNotEqual(edge.getSourceId(), unattachedVertex.getId())
            self.assertNotEqual(edge.getTargetId(), unattachedVertex.getId())


    def test_attach_at_vertices_removes_old_edges(self):
        st = Spacetime()

        attachedSimplex = st.createSimplex((1, 4))
        unattachedSimplex = st.createSimplex((2, 3))  # Unattached

        self.assertEqual(len(unattachedSimplex.getEdges()), 10)
        self.assertEqual(len(attachedSimplex.getEdges()), 10)
        self.assertEqual(len(st.getEdgeList().toVector()), 20)

        unattachedSimplexFace, attachedSimplexFace = st.chooseSimplexFacesToGlue(unattachedSimplex)
        unattachedVertices = [v for v in unattachedSimplexFace.getVertices()]
        attachedVertices = [v for v in attachedSimplexFace.getVertices()]

        st.attachAtVertices(unattachedSimplexFace, attachedSimplexFace, [(u, a) for u, a in zip(unattachedVertices, attachedVertices)])

        unattachedVertex = unattachedVertices[0]  # V6
        attachedVertex = attachedVertices[0]      # V0

        # 3. overall edge count is wrong after the moves (17): len(attachedSimplex.getEdges()) is 10, so is len(unattachedSimplex.getEdges())
        simplex_edges = attachedSimplex.getEdges() + unattachedSimplex.getEdges()
        total_edges = st.getEdgeList().toVector()

        self.assertEqual(len(unattachedSimplex.getEdges()), 10)
        self.assertEqual(len(attachedSimplex.getEdges()), 10)

        self.assertEqual(len(simplex_edges), 20)  # 6 shared/doubled.
        self.assertEqual(len(set(simplex_edges)), 14)

        shared_edges = set(attachedSimplex.getEdges()) & set(unattachedSimplex.getEdges())
        self.assertEqual(len(shared_edges), 6)
        self.assertEqual(len(set(simplex_edges)), len(st.getEdgeList().toVector()))

        self.assertEqual(set(simplex_edges), set(total_edges))

    def test_three_simplices_can_share_an_edge(self):
        st = Spacetime()

        sa = st.createSimplex((1, 4))
        sb = st.createSimplex((2, 3))
        sc = st.createSimplex((1, 4))
        sd = st.createSimplex((2, 3))
        se = st.createSimplex((1, 4))
        sg = st.createSimplex((2, 3))


        fa = [f for f in sa.getFacets() if f.getOrientation().numeric() == (1, 3)][0]
        fb = [f for f in sb.getFacets() if f.getOrientation().numeric() == (1, 3)][0]
        fc = [f for f in sc.getFacets() if f.getOrientation().numeric() == (1, 3)][0]
        fd = [f for f in sd.getFacets() if f.getOrientation().numeric() == (1, 3)][0]
        fe = [f for f in se.getFacets() if f.getOrientation().numeric() == (1, 3)][0]
        fg = [f for f in sg.getFacets() if f.getOrientation().numeric() == (1, 3)][0]

        fa, _ = st.causallyAttachFaces(fa, fb)
        self.assertTrue(_)
        self.assertEqual(len(fa.getCofaces()), 2)

        self.assertEqual(len(fb.getCofaces()), 1)
        fb, _ = st.causallyAttachFaces(fb, fc)
        self.assertTrue(_)
        self.assertEqual(len(fa.getCofaces()), 2)

        self.assertEqual(len(fc.getCofaces()), 1)
        fc, _ = st.causallyAttachFaces(fc, fd)
        self.assertTrue(_)
        self.assertEqual(len(fc.getCofaces()), 2)

        fd, _ = st.causallyAttachFaces(fd, fe)
        self.assertTrue(_)

        fe, _ = st.causallyAttachFaces(fe, fg)
        self.assertTrue(_)

        for f in [fa, fb, fc, fd, fe, fg]:
            f.validate()
        for s in [sa, sb, sc, sd, se, sg]:
            s.validate()


    def test_lots_of_components_connect_once_glued4D(self):
        # I think when we connect one face of a simplex to another, then attach a third along some vertices shared by
        # the first two then we have troubles.
        st = Spacetime()

        orientations = [(1, 4), (2, 3)]
        for i in range(10):
            print("Creating simplex %s" % i)
            s = st.createSimplex(orientations[i % 2])
            print("Validating simplex %s" % i)
            s.validate()
            print("Validated simplex %s" % i)
            if i == 0: continue
            for f in s.getFacets():
                f.validate()

            print("Choosing a face to glue...");
            leftFace, rightFace = st.chooseSimplexFacesToGlue(s)
            print("validating faces before attachment")
            leftFace.validate(), rightFace.validate()
            print('validated faces.')
            if leftFace.isTimelike() != rightFace.isTimelike():
                continue

            updated, succeeded = st.causallyAttachFaces(leftFace, rightFace)
            print('validating faces after attachment')
            leftFace.validate(), rightFace.validate()
            print('validated.')
            print('validating complex after attachment')
            updated.validate()
            print('validated')
            self.assertTrue(succeeded)

        components = st.getConnectedComponents()
        self.assertEqual(len(components), 1)

    def test_lots_of_components_connect_once_glued2D(self):
        st = Spacetime()

        orientations = [(1, 2), (2, 1)]
        for i in range(10):
            print('1')
            unattached = st.createSimplex(orientations[i % 2])
            print('2')
            if i == 0: continue
            print('3')
            leftFace, rightFace = st.chooseSimplexFacesToGlue(unattached)
            print('validating faces...')
            leftFace.validate(), rightFace.validate()

            print('4')
            leftCofaces = leftFace.getCofaces()
            print('5')
            rightCofaces = rightFace.getCofaces()
            print('6')

            self.assertEqual(len(leftCofaces), 1)
            self.assertEqual(len(rightCofaces), 1)

            print('7')
            updated, succeeded = st.causallyAttachFaces(leftFace, rightFace)
            print('8')

            if not succeeded:
                breakpoint()
                print('failed')
            self.assertTrue(succeeded)

        print('9')
        components = st.getConnectedComponents()
        print('10')
        self.assertEqual(len(components), 1)

    def test_get_gluable_faces(self):
        st = Spacetime()

        s12 = st.createSimplex((1, 2))
        s21 = st.createSimplex((2, 1))

        gluableFaces = st.getGluableFaces(s12, s21)

        self.assertEqual(len(gluableFaces), 2)

        self.assertNotEqual(gluableFaces[0].getOrientation().numeric(), (2, 0))
        self.assertNotEqual(gluableFaces[0].getOrientation().numeric(), (0, 2))

        self.assertNotEqual(gluableFaces[1].getOrientation().numeric(), (2, 0))
        self.assertNotEqual(gluableFaces[1].getOrientation().numeric(), (0, 2))

        gluableFaces[0].validate()
        gluableFaces[1].validate()


if __name__ == '__main__':
    unittest.main()