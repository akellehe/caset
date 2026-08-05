# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The two-phase boundary-then-bulk construction (cobordism::SlicedCobordism).

Phase 1 is a closed 3-complex -- a spatial slice with no temporal extent, every
edge spacelike. Phase 2 cones it to a single shared apex through timelike edges,
giving a 4-complex whose boundary is the slice.

Why the apex is shared: a 4D CDT slab between consecutive slices needs BOTH
(4,1) and (3,2) simplices -- four vertices on one slice with one on the next,
and three with two. Coning each tetrahedron to its own apex yields only (4,1)
cells, which leave gaps and do not tile a manifold. One shared apex tiles
exactly, giving cone(S^3) = D^4, at the price of a conical singularity.

Why this exists at all: on canonically built hosts every edge is measured
spacelike and Im S = 0 at every stage of the drive, because fromCells starts
every edge at l^2 = +1 and nothing in the drive seeds causal content. The bulk
is reached spacelike-ly, which is not what a Lorentzian cobordism should look
like. Seeding the disposition at construction addresses that at its origin.

These tests pin the CONSTRUCTION. Whether the timelike dispositions survive the
geometric relaxation is deliberately not asserted here -- nothing prevents stage
2 from driving them spacelike, that would be a runtime guard on the dynamics,
and their survival is a measurement rather than an invariant.
"""

import collections
import unittest

import tessera

cob = tessera.cobordism


def _dimension(st):
    return int(tessera.cobordism.CombinatorialDimension().compute(st))


def _simplices_by_dimension(st):
    return collections.Counter(len(s.getVertices()) - 1 for s in st.getSimplices())


def _dispositions(st):
    hist = collections.Counter()
    for e in st.getEdgeList().toVector():
        if e.isTimelike():
            hist["timelike"] += 1
        elif e.isNull():
            hist["null"] += 1
        else:
            hist["spacelike"] += 1
    return hist


class TestClosedSlice(unittest.TestCase):
    """Phase 1: the closed 3-complex spatial slice."""

    def test_is_a_three_complex_of_five_tetrahedra(self):
        st = cob.SlicedCobordism.closed_slice()
        self.assertEqual(_dimension(st), 3,
                         "the phase-1 slice must be genuinely 3-dimensional; a "
                         "4-complex would defeat the point of the two phases")
        cells = cob.SlicedCobordism.top_cells(st)
        self.assertEqual(len(cells), 5, "dDelta^4 has five tetrahedra")
        for cell in cells:
            self.assertEqual(len(cell), 4, "a 3-complex top cell is a tetrahedron")
        self.assertEqual(len({v for c in cells for v in c}), 5,
                         "dDelta^4 spans five vertices")

    def test_is_closed(self):
        """Every triangle must lie in exactly two tetrahedra.

        cone_to_bulk requires this: coning a slice WITH boundary would give a
        complex whose boundary is not the slice.
        """
        cells = cob.SlicedCobordism.top_cells(cob.SlicedCobordism.closed_slice())
        faces = collections.Counter()
        for cell in cells:
            for drop in range(len(cell)):
                faces[tuple(v for i, v in enumerate(cell) if i != drop)] += 1
        self.assertTrue(faces, "the slice must have faces to count")
        self.assertEqual(set(faces.values()), {2},
                         "a closed 3-manifold has every triangle in exactly two "
                         "tetrahedra")

    def test_every_edge_is_spacelike(self):
        """A spatial slice has no temporal extent; Im S = 0 is correct here."""
        st = cob.SlicedCobordism.closed_slice()
        self.assertEqual(_dispositions(st),
                         collections.Counter(spacelike=10),
                         "dDelta^4 has C(5,2) = 10 edges, all spacelike")


class TestConeToBulk(unittest.TestCase):
    """Phase 2: the shared-apex cone to a 4-complex."""

    def setUp(self):
        self.slice = cob.SlicedCobordism.closed_slice()
        self.bulk, self.reason = cob.SlicedCobordism.cone_to_bulk(self.slice)
        self.assertEqual(self.reason, "ok", f"cone_to_bulk failed: {self.reason}")
        self.assertIsNotNone(self.bulk)

    def test_produces_a_four_complex_one_cell_per_tetrahedron(self):
        self.assertEqual(_dimension(self.bulk), 4)
        cells = cob.SlicedCobordism.top_cells(self.bulk)
        self.assertEqual(len(cells), 5,
                         "one 4-simplex per tetrahedron of the slice")
        for cell in cells:
            self.assertEqual(len(cell), 5, "a 4-complex top cell is a pentatope")

    def test_apex_is_shared_by_every_cell(self):
        """One apex, not one per tetrahedron -- (4,1)-only does not tile."""
        cells = cob.SlicedCobordism.top_cells(self.bulk)
        shared = set(cells[0])
        for cell in cells[1:]:
            shared &= set(cell)
        self.assertEqual(len(shared), 1,
                         "every cell must share exactly one vertex: the apex")
        self.assertEqual(len({v for c in cells for v in c}), 6,
                         "five slice vertices plus one apex")

    def test_apex_edges_are_timelike_and_slice_edges_are_not(self):
        cells = cob.SlicedCobordism.top_cells(self.bulk)
        shared = set(cells[0])
        for cell in cells[1:]:
            shared &= set(cell)
        apex = shared.pop()

        for e in self.bulk.getEdgeList().toVector():
            a, b = e.getSource().getId(), e.getTarget().getId()
            if apex in (a, b):
                self.assertTrue(
                    e.isTimelike(),
                    f"apex edge ({a},{b}) must be timelike: it runs from the "
                    f"spatial slice into the bulk, which is time evolution")
            else:
                self.assertFalse(
                    e.isTimelike(),
                    f"edge ({a},{b}) lies within the slice and must stay "
                    f"spacelike; only apex-incident edges are written")

    def test_disposition_counts(self):
        """Five apex edges (one per slice vertex), ten slice edges."""
        self.assertEqual(_dispositions(self.bulk),
                         collections.Counter(spacelike=10, timelike=5))

    def test_boundary_is_the_slice(self):
        """cone(S^3) = D^4: the boundary of the bulk is the original slice."""
        boundary = {tuple(sorted(f)) for f in self.bulk.getBoundary()}
        slice_cells = {tuple(c) for c in
                       cob.SlicedCobordism.top_cells(self.slice)}
        self.assertEqual(boundary, slice_cells)

    def test_rejects_a_slice_that_is_not_closed(self):
        """A 3-ball would cone to something whose boundary is not the slice."""
        st = cob.SlicedCobordism.closed_slice()
        cells = cob.SlicedCobordism.top_cells(st)
        opened = tessera.Spacetime.fromCells(3, [list(c) for c in cells[:-1]],
                                             1.0, 0.0)
        bulk, reason = cob.SlicedCobordism.cone_to_bulk(opened)
        self.assertIsNone(bulk)
        self.assertIn("closed", reason)

    def test_spacelike_lengths_are_carried_over(self):
        """Slice geometry passes through unchanged; only apex edges are written."""
        for e in self.slice.getEdgeList().toVector():
            e.setSquaredLength(complex(2.25, 0.0))
        bulk, reason = cob.SlicedCobordism.cone_to_bulk(self.slice)
        self.assertEqual(reason, "ok")
        cells = cob.SlicedCobordism.top_cells(bulk)
        shared = set(cells[0])
        for cell in cells[1:]:
            shared &= set(cell)
        apex = shared.pop()
        for e in bulk.getEdgeList().toVector():
            a, b = e.getSource().getId(), e.getTarget().getId()
            if apex not in (a, b):
                self.assertAlmostEqual(e.getSquaredLength().real, 2.25, places=12)


if __name__ == "__main__":
    unittest.main()
