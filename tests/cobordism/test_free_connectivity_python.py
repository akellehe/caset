# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Free interior connectivity in the realizability engine (#200).

`growInterior` only ever **cones** a fresh interior vertex into one top cell,
wiring it to exactly the ``d+1`` vertices of that cell — a biased move. This
suite covers the cone-free generalization: a purely-additive growth primitive
``EigenstateSynthesis.attachInteriorVertex(incident_simplices)`` that adds an
interior vertex with an **arbitrary** incidence (coning is one special case), its
exact inverse ``detachLastInteriorVertex``, and the bounded connectivity-search
layer on ``RealizabilityOracle.decide`` (``growth_mode=FREE_CONNECTIVITY``).

The only invariant the primitive enforces is the one the experiment allows: the
result is a valid downward-closed abstract simplicial complex, and the pinned
boundary ``dW`` is **bit-exact** untouched. No manifold / pseudomanifold /
purity / topology constraint is imposed — the realized topology is whatever the
residual selects.

Key fact the engine leans on: the ``k=0`` Hodge Laplacian ``L = D - A`` (the
objective) is assembled **only from the 1-skeleton** (edge weights/phases), so
"free interior connectivity" here is exactly "free interior graph connectivity"
— which existing vertices a new interior vertex wires to.
"""

import math
import unittest

import numpy as np

import tessera

cob = tessera.cobordism


# --------------------------------------------------------------------------- #
# Fixtures — the #147 bulk-synthesis idiom, reused verbatim from the oracle
# suite: Signature(d) so the d-cells register as top simplices; built through
# the topology so the vertex-id counter advances.
# --------------------------------------------------------------------------- #
def _spacetime(dim, topology):
    sig = tessera.Signature(dim, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    return tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                             tessera.PREFERRED, topology)


def _solid_triangle():
    """Delta^2: triangle 0-1-2 whose boundary dW is the three sides (S^1)."""
    st = _spacetime(2, tessera.SolidSimplex(2))
    st.build()
    return st


def _bipyramid():
    """Triangles 012 and 013 sharing interior edge 01; the four outer edges
    02,12,03,13 are the pinned boundary dW (the smallest fixed-boundary complex
    with exactly one interior parameter)."""
    st = _solid_triangle()
    v = {x.getId(): x for x in st.getVertexList().toVector()}
    v3 = st.createVertex(3)
    st.createSimplex([v[0], v[1], v3])
    return st


def _pin_all(st, w=1.0, phase=0.0):
    for e in st.getEdgeList().toVector():
        e.setSquaredLength(w)
        e.setPhase(phase)


def _edge_map(st):
    out = {}
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        out[(min(a, b), max(a, b))] = e
    return out


def _edge_keys(st):
    return set(_edge_map(st).keys())


def _vertex_ids(st):
    return sorted(v.getId() for v in st.getVertexList().toVector())


def _cvec(v):
    return [complex(z) for z in v]


def _vec(U):
    return np.asarray(U, dtype=complex).reshape(-1)


# --------------------------------------------------------------------------- #
class AttachInteriorVertexTest(unittest.TestCase):
    """The general growth primitive: arbitrary incidence, validity + boundary
    are the only gates, coning is a special case."""

    def test_attach_edges_grows_order_and_interior_connectivity(self):
        st = _bipyramid()
        _pin_all(st)
        es = cob.EigenstateSynthesis(st)

        order0 = es.order()
        ninterior0 = es.numInteriorEdges()
        nboundary0 = es.numBoundaryEdges()
        verts0 = set(es.vertexIds())

        # Wire a fresh interior vertex to two existing vertices by edges
        # (singleton specs -> 1-simplices). A purely additive, boundary-safe move.
        self.assertTrue(es.attachInteriorVertex([[0], [3]]))

        # Order grew by exactly one; the new vertex has the largest id.
        self.assertEqual(es.order(), order0 + 1)
        new_ids = set(es.vertexIds()) - verts0
        self.assertEqual(len(new_ids), 1)
        self.assertEqual(max(new_ids), max(es.vertexIds()))
        new_id = next(iter(new_ids))

        # The two new edges are interior (free), boundary unchanged in count.
        self.assertEqual(es.numInteriorEdges(), ninterior0 + 2)
        self.assertEqual(es.numBoundaryEdges(), nboundary0)
        interior = set(es.interiorEdges())
        self.assertIn((0, new_id), interior)
        self.assertIn((3, new_id), interior)

    def test_attach_validates_downward_closure_of_a_higher_simplex(self):
        # A single 3-vertex spec {1,2,3} cones the new vertex into a *triangle*
        # spanning interior vertices: the new simplex {1,2,3,new} materializes its
        # whole 1-skeleton, so every pair carries an edge (downward closed).
        st = _bipyramid()
        _pin_all(st)
        # 2-3 is not an edge of the bipyramid; attaching {1,2,3,new} must create it
        # (along with the new vertex's edges) to stay a valid complex.
        self.assertNotIn((2, 3), _edge_keys(st))
        es = cob.EigenstateSynthesis(st)
        self.assertTrue(es.attachInteriorVertex([[1, 2, 3]]))
        new_id = max(es.vertexIds())
        keys = _edge_keys(st)
        for a, b in [(1, 2), (1, 3), (2, 3),
                     (1, new_id), (2, new_id), (3, new_id)]:
            self.assertIn((min(a, b), max(a, b)), keys)

    def test_invalid_specs_are_rejected_and_leave_complex_unchanged(self):
        st = _bipyramid()
        _pin_all(st)
        es = cob.EigenstateSynthesis(st)
        order0, edges0 = es.order(), _edge_keys(st)

        for bad in (
            [],            # empty incidence -> would isolate the new vertex
            [[]],          # an empty face
            [[99]],        # dangling vertex reference
            [[0, 0]],      # repeated vertex within a face
        ):
            self.assertFalse(es.attachInteriorVertex(bad))
            self.assertEqual(es.order(), order0)
            self.assertEqual(_edge_keys(st), edges0)

    def test_attach_that_would_perturb_boundary_is_rejected(self):
        # Spec {0,2} forms the *triangle* {0,2,new}: edge 02 is a boundary facet
        # of triangle 012, so the new triangle would give it a second coface and
        # flip it interior — perturbing dW. The primitive must reject + roll back.
        st = _bipyramid()
        _pin_all(st)
        es = cob.EigenstateSynthesis(st)
        boundary0 = set(es.boundaryEdges())
        order0, edges0 = es.order(), _edge_keys(st)
        self.assertIn((0, 2), boundary0)

        self.assertFalse(es.attachInteriorVertex([[0, 2]]))

        # Bit-exact unchanged: order, edges, and the boundary partition.
        self.assertEqual(es.order(), order0)
        self.assertEqual(_edge_keys(st), edges0)
        self.assertEqual(set(es.boundaryEdges()), boundary0)

    def test_boundary_is_bit_exact_across_an_edge_attach(self):
        st = _bipyramid()
        _pin_all(st, w=1.0, phase=0.0)
        es = cob.EigenstateSynthesis(st)
        before = {k: (e.getSquaredLength().real, e.getPhase())
                  for k, e in _edge_map(st).items()
                  if k in set(es.boundaryEdges())}

        self.assertTrue(es.attachInteriorVertex([[0], [1], [2]]))

        live = _edge_map(st)
        for k, (w, ph) in before.items():
            self.assertEqual((live[k].getSquaredLength().real, live[k].getPhase()),
                             (w, ph))
        # The boundary edge *set* is preserved exactly.
        self.assertEqual(set(es.boundaryEdges()), set(before.keys()))

    def test_coning_is_a_special_case_of_the_general_attach(self):
        # Wiring the new vertex to the d+1 = 3 vertices of a top cell reproduces
        # growInterior's 1-skeleton (the new vertex joined to that cell's
        # vertices) — cone connectivity is a subset of what attach can express.
        st = _bipyramid()
        _pin_all(st)
        es = cob.EigenstateSynthesis(st)
        cell = es.topCells()[0]                       # e.g. [0,1,2]
        self.assertEqual(len(cell), 3)
        self.assertTrue(es.attachInteriorVertex([[c] for c in cell]))
        new_id = max(es.vertexIds())
        interior = set(es.interiorEdges())
        for c in cell:
            self.assertIn((min(c, new_id), max(c, new_id)), interior)


class DetachInteriorVertexTest(unittest.TestCase):
    """detachLastInteriorVertex is the exact inverse of attachInteriorVertex."""

    def test_attach_then_detach_round_trips_the_complex(self):
        st = _bipyramid()
        _pin_all(st, w=1.0, phase=0.0)
        es = cob.EigenstateSynthesis(st)

        order0 = es.order()
        edges0 = _edge_keys(st)
        verts0 = _vertex_ids(st)
        weights0 = {k: (e.getSquaredLength().real, e.getPhase())
                    for k, e in _edge_map(st).items()}
        boundary0 = set(es.boundaryEdges())

        # A rich attach (an interior triangle that even creates a new edge 2-3).
        self.assertTrue(es.attachInteriorVertex([[1, 2, 3], [0]]))
        self.assertGreater(es.order(), order0)
        self.assertNotEqual(_edge_keys(st), edges0)

        # Detach restores everything bit-exactly.
        self.assertTrue(es.detachLastInteriorVertex())
        self.assertEqual(es.order(), order0)
        self.assertEqual(_vertex_ids(st), verts0)
        self.assertEqual(_edge_keys(st), edges0)
        self.assertEqual(set(es.boundaryEdges()), boundary0)
        live = _edge_map(st)
        for k, (w, ph) in weights0.items():
            self.assertEqual((live[k].getSquaredLength().real, live[k].getPhase()),
                             (w, ph))

    def test_detach_with_nothing_to_undo_returns_false(self):
        st = _bipyramid()
        _pin_all(st)
        es = cob.EigenstateSynthesis(st)
        self.assertFalse(es.detachLastInteriorVertex())

    def test_detach_is_lifo_over_repeated_attaches(self):
        st = _bipyramid()
        _pin_all(st)
        es = cob.EigenstateSynthesis(st)
        base = es.order()
        self.assertTrue(es.attachInteriorVertex([[0], [1]]))
        self.assertTrue(es.attachInteriorVertex([[2], [3]]))
        self.assertEqual(es.order(), base + 2)
        self.assertTrue(es.detachLastInteriorVertex())
        self.assertEqual(es.order(), base + 1)
        self.assertTrue(es.detachLastInteriorVertex())
        self.assertEqual(es.order(), base)
        self.assertFalse(es.detachLastInteriorVertex())


# --------------------------------------------------------------------------- #
def _fan():
    """Triangles 012, 013, 014 sharing the interior edge 01 — a fan whose three
    boundary 'leaves' 2,3,4 each sit in exactly one top cell. The cone move wires
    a fresh vertex to one top cell's 3 vertices, so it can reach at most ONE leaf
    and always misses the other two: a genuine connectivity bottleneck the
    free-connectivity search can step around."""
    st = _solid_triangle()
    v = {x.getId(): x for x in st.getVertexList().toVector()}
    v3 = st.createVertex(3)
    st.createSimplex([v[0], v[1], v3])
    v4 = st.createVertex(4)
    st.createSimplex([v[0], v[1], v4])
    return st


class ConeVsFreeConnectivityTest(unittest.TestCase):
    """Cone-only vs free-connectivity growth on the same target, reported
    honestly. The cone baseline is ``decide(FREE_CONNECTIVITY,
    connectivity_candidates=1)`` — exactly the cone-equivalent candidate (the
    d+1 vertices of one top cell), grown by the same boundary-fixed attach as the
    free search, so the two arms differ ONLY in the connectivity breadth searched.
    (``growth_mode=CONE`` is the Pachner cone ``growInterior``; on hand-built
    explicit-id fixtures it reissues a stale vertex id, which the additive attach's
    max-id allocation avoids — so candidates=1 is the faithful, robust cone arm.)"""

    # U whose boundary amplitudes on the three fan leaves 2,3,4 are distinct: a
    # cone vertex reaching only one leaf forces two rigid boundary rows to demand
    # inconsistent eigenvalues -> a residual floor; a free vertex coupling the
    # leaves reconciles them.
    U_FAN = [[1.0, 1.0, 2.0, 3.0, 5.0]]   # 1 x 5; vec on vertices 0,1,2,3,4

    def test_free_connectivity_realizes_a_target_cone_growth_floors_on(self):
        FREE = cob.RealizabilityOracle.GrowthMode.FREE_CONNECTIVITY

        st = _fan()
        _pin_all(st)
        cone = cob.RealizabilityOracle(st).decide(
            _cvec(_vec(self.U_FAN)), 1, 5, epsilon=1e-10, restarts=64,
            max_cones=1, seed=0, growth_mode=FREE, connectivity_candidates=1)

        st = _fan()
        _pin_all(st)
        free = cob.RealizabilityOracle(st).decide(
            _cvec(_vec(self.U_FAN)), 1, 5, epsilon=1e-10, restarts=64,
            max_cones=1, seed=0, growth_mode=FREE, connectivity_candidates=8)

        # Cone connectivity floors away from zero — certified non-realizable.
        self.assertFalse(cone.realizable)
        self.assertGreater(cone.residual, 1e-4)

        # Free connectivity realizes the SAME target with a single interior vertex.
        self.assertTrue(free.realizable)
        self.assertLess(free.residual, 1e-9)
        self.assertEqual(free.interior_vertex_count, 1)

        # The headline: removing the cone bias drops the residual by many orders.
        self.assertLess(free.residual, cone.residual / 1e4)

        # The realized connectivity is genuinely emergent (not a cone star): the
        # grown vertex couples boundary leaves directly rather than the hub.
        es = cob.EigenstateSynthesis(free.witness)
        self.assertLess(es.residual(_cvec(free.state)), 1e-9)
        new_id = max(es.vertexIds())
        nbrs = {a if b == new_id else b
                for (a, b) in es.interiorEdges() if new_id in (a, b)}
        # It reaches at least two of the fan leaves {2,3,4} — impossible for a
        # single cone (one top cell holds only one leaf).
        self.assertGreaterEqual(len(nbrs & {2, 3, 4}), 2)

    def test_cone_growth_is_already_sufficient_on_the_bipyramid(self):
        # Honest counterpoint: on the existing bipyramid fixture a single missed
        # boundary vertex only *fixes* the eigenvalue (the rest of the interior
        # adapts), so cone connectivity already realizes U=[[1,2],[3,4]] — free
        # connectivity is not needed here. Both reported, neither hidden.
        FREE = cob.RealizabilityOracle.GrowthMode.FREE_CONNECTIVITY
        U = [[1.0, 2.0], [3.0, 4.0]]

        st = _bipyramid()
        _pin_all(st)
        cone = cob.RealizabilityOracle(st).decide(
            _cvec(_vec(U)), 2, 2, epsilon=1e-10, restarts=48, max_cones=1,
            seed=0, growth_mode=FREE, connectivity_candidates=1)

        st = _bipyramid()
        _pin_all(st)
        free = cob.RealizabilityOracle(st).decide(
            _cvec(_vec(U)), 2, 2, epsilon=1e-10, restarts=48, max_cones=1,
            seed=0, growth_mode=FREE, connectivity_candidates=8)

        self.assertTrue(cone.realizable)
        self.assertTrue(free.realizable)
        # Free is never worse than cone (it includes the cone candidate).
        self.assertLessEqual(free.residual, max(cone.residual, 1e-9) * 10)

    def test_connectivity_search_is_bounded_and_logged(self):
        # No silent cap: the Verdict surfaces exactly how many candidates were
        # scored vs the full 2^N-1 incidence space they were pruned from.
        FREE = cob.RealizabilityOracle.GrowthMode.FREE_CONNECTIVITY
        st = _fan()
        _pin_all(st)
        v = cob.RealizabilityOracle(st).decide(
            _cvec(_vec(self.U_FAN)), 1, 5, epsilon=1e-10, restarts=32,
            max_cones=1, seed=0, growth_mode=FREE, connectivity_candidates=8)

        # 5 vertices at the (single) growth step -> 2^5 - 1 = 31 incidence patterns.
        self.assertEqual(v.connectivity_space_size, 31)
        self.assertGreaterEqual(v.connectivity_candidates, 3)   # >= the 3 anchors
        self.assertLessEqual(v.connectivity_candidates, 8)
        self.assertLess(v.connectivity_candidates, v.connectivity_space_size)

        # Cone mode does not search connectivity (reports 0 candidates).
        st = _fan()
        _pin_all(st)
        c = cob.RealizabilityOracle(st).decide(
            _cvec(_vec(self.U_FAN)), 1, 5, epsilon=1e-10, restarts=8,
            max_cones=0, seed=0)                       # default growth_mode=CONE
        self.assertEqual(c.connectivity_candidates, 0)

    def test_decide_free_connectivity_is_deterministic(self):
        FREE = cob.RealizabilityOracle.GrowthMode.FREE_CONNECTIVITY
        out = []
        for _ in range(2):
            st = _fan()
            _pin_all(st)
            out.append(cob.RealizabilityOracle(st).decide(
                _cvec(_vec(self.U_FAN)), 1, 5, epsilon=1e-10, restarts=48,
                max_cones=1, seed=3, growth_mode=FREE, connectivity_candidates=8))
        self.assertEqual(out[0].residual, out[1].residual)
        self.assertEqual(out[0].realizable, out[1].realizable)
        np.testing.assert_array_equal(np.asarray(out[0].state),
                                      np.asarray(out[1].state))


if __name__ == "__main__":
    unittest.main()
