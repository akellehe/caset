# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The ordinary-Lorentzian on-axis invariant at the geometry read (#597).

Storage (get/setSquaredLength) stays general-complex — rollback records and
historical dump rehydration depend on exact round-trips — but the geometry
stack reads l^2 through Edge.getRealSquaredLength(), which throws on a nonzero
Im l^2 instead of silently projecting it away (#589 keeps l^2 real and signed
by construction, so any resident Im is an upstream bug, not a value to drop).

Also covers the one genuine truncation the #580-census residue audit found:
Vertex.moveEdgesTo used to launder the exact edge state through the
double-typed createEdge funnel — Re-only l^2 and a zeroed U(1) phase on every
relocated edge. It must preserve both verbatim.
"""
import unittest

import tessera


def _spacetime():
    sig = tessera.Signature(3, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, tessera.SolidSimplex(3))
    st.build()
    return st


class RealSquaredLengthInvariant(unittest.TestCase):
    def test_on_axis_read_returns_the_signed_real_value(self):
        st = _spacetime()
        edge = st.getEdgeList().toVector()[0]
        edge.setSquaredLength(complex(-2.25, 0.0))
        self.assertEqual(edge.getRealSquaredLength(), -2.25)

    def test_nonzero_im_l2_throws_instead_of_truncating(self):
        st = _spacetime()
        edge = st.getEdgeList().toVector()[0]
        edge.setSquaredLength(complex(1.0, 0.5))
        with self.assertRaisesRegex(RuntimeError, "Im l\\^2"):
            edge.getRealSquaredLength()

    def test_storage_still_round_trips_general_complex_exactly(self):
        # The invariant lives at geometry consumption ONLY: rollback records
        # and historical dumps must keep writing/reading complex l^2 verbatim.
        st = _spacetime()
        edge = st.getEdgeList().toVector()[0]
        edge.setSquaredLength(complex(0.75, -0.3))
        self.assertEqual(edge.getSquaredLength(), complex(0.75, -0.3))


class MoveEdgesPreservesExactState(unittest.TestCase):
    def test_moved_edge_keeps_complex_l2_and_phase(self):
        # Two disjoint tetrahedra: the mover's and recipient's neighbourhoods
        # are disjoint, so every relocated edge is a genuinely new edge. (On a
        # complete graph the mover-recipient edge becomes a self-loop and the
        # noexcept createEdge funnel terminates on it — pre-existing funnel
        # behaviour, #599's territory, deliberately not exercised here.)
        st = tessera.spacetime.Spacetime.fromCells(
            3, [[0, 1, 2, 3], [4, 5, 6, 7]])
        by_id = {v.getId(): v for v in st.getVertexList().toVector()}
        mover, recipient = by_id[0], by_id[4]
        planted_l2, planted_phase = complex(2.5, -0.75), 0.6
        # Plant through the LIVE EdgeList handle: Vertex.getOutEdges/getInEdges
        # return deep COPIES in Python (return_value_policy::copy propagates to
        # the elements — the getFacets/getCofaces trap), so a plant through
        # them never reaches the mesh.
        moved = [e for e in st.getEdgeList().toVector()
                 if mover.getId() in (e.getSource().getId(),
                                      e.getTarget().getId())][0]
        moved.setSquaredLength(planted_l2)
        moved.setPhase(planted_phase)
        _old, new_edges = mover.moveEdgesTo(recipient, st)
        carriers = [e for e in new_edges
                    if e.getSquaredLength() == planted_l2
                    and e.getPhase() == planted_phase]
        self.assertTrue(
            carriers,
            "no relocated edge carries the exact complex l^2 + U(1) phase — "
            "the createEdge funnel laundered the state again")


if __name__ == "__main__":
    unittest.main()
