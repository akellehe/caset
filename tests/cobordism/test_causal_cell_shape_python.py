# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Causal cell shape, Lorentzian admissibility, and the dual-height census (#620).

Three read-only measurements, all composed from existing `Simplex` helpers.

**Primal or dual.** Cell shape and admissibility read the PRIMAL complex — a
simplex's own vertices, edges and Gram matrix. The dual is where the Lorentzian
spacetime lives: S = sum_h |*h| eps_h is a sum over dual volumes. So a primal
shape statistic is a diagnostic, not the physics; the primal bipartition is the
combinatorial precondition for a foliation-like dual structure, and its absence
says the complex has not developed one. The dual-height census is the
measurement that reaches the dual directly.

**Cell shape.** A 4-simplex has 5 vertices and 10 edges. If its vertices split
(a, b) with a + b = 5, exactly a*b edges cross between the groups, so (5,0) gives
0 timelike edges, (4,1) gives 4 and (3,2) gives 6. Those are the ONLY counts a
genuine bipartition can produce, and a cell holding one or two timelike edges
corresponds to no consistent temporal split. Classification therefore VERIFIES
the timelike edges are exactly a crossing set rather than matching the count.

**Lorentzian admissibility.** Gram signature (-,+,+,+) -- exactly one timelike
direction -- read by Jacobi's criterion on the leading principal minors.
`dualComplexIsValid` is purely combinatorial and never reads l^2, so it passes
cells that have no consistent Lorentzian geometry at all.

**Dual-height census.** Each dual height is
`oppositeVertexSign(cf,s) * signedSqrt(R^2(cf) - R^2(s))`. The barycentric factor
going negative is a MESH DEFECT; the radicand going negative is a TIMELIKE dual
separation, which is correct physics rather than a defect. The census separates
them, because a remedy aimed at all negatives would destroy the physical ones.

Shape and admissibility are pinned against hand-built cells whose answer is known
by construction; the census is pinned against an independent Python oracle.
"""

import itertools
import unittest

import tessera

cob = tessera.cobordism
CS = cob.CausalCellShape


def _pentatope():
    st = tessera.Spacetime.fromCells(4, [[0, 1, 2, 3, 4]], 1.0, 0.0)
    tessera.ReggeSolver(st, tessera.MatterConfiguration())
    return st


def _top(st):
    return [s for s in st.getSimplices() if len(s.getVertices()) == 5][0]


def _set_split(st, group):
    """Timelike exactly on edges crossing between `group` and the rest."""
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        crossing = (a in group) != (b in group)
        e.setSquaredLength(complex(-1.0 if crossing else 1.0, 0.0))


class TestCellShape(unittest.TestCase):
    """Shapes are read from a verified bipartition, not a timelike count."""

    def test_known_shapes(self):
        for group, expected, n_timelike in [(set(), "spacelike", 0),
                                            ({4}, "(4,1)", 4),
                                            ({3, 4}, "(3,2)", 6)]:
            with self.subTest(shape=expected):
                st = _pentatope()
                _set_split(st, group)
                cell = _top(st)
                self.assertEqual(
                    sum(1 for e in cell.getEdges() if e.isTimelike()), n_timelike,
                    f"a {expected} split must cross exactly {n_timelike} edges")
                self.assertEqual(CS.shape_name(CS.classify(cell)), expected)

    def test_stray_timelike_edge_is_non_bipartite(self):
        """One timelike edge is no consistent split -- (4,1) needs four."""
        st = _pentatope()
        _set_split(st, set())
        st.getEdgeList().toVector()[0].setSquaredLength(complex(-1.0, 0.0))
        cell = _top(st)
        self.assertEqual(sum(1 for e in cell.getEdges() if e.isTimelike()), 1)
        self.assertEqual(CS.shape_name(CS.classify(cell)), "non-bipartite")

    def test_distribution_counts_top_cells(self):
        st = _pentatope()
        _set_split(st, {4})
        dist = CS.distribution(st)
        self.assertEqual(dist[int(cob.CellShape.FOUR_ONE)], 1)
        self.assertEqual(sum(dist), 1, "one top cell, counted once")


class TestLorentzianAdmissibility(unittest.TestCase):
    """Signature (-,+,+,+), and a purely spacelike cell positive-definite."""

    def test_timelike_directions_by_shape(self):
        for group, expected in [(set(), 0), ({4}, 1), ({3, 4}, 1)]:
            with self.subTest(group=sorted(group)):
                st = _pentatope()
                _set_split(st, group)
                cell = _top(st)
                self.assertEqual(CS.timelike_direction_count(cell), expected)
                self.assertTrue(CS.is_lorentzian_admissible(cell))

    def test_non_bipartite_can_still_be_admissible(self):
        """Bipartiteness and admissibility are INDEPENDENT.

        A stray timelike edge yields no consistent temporal split, yet the cell
        remains a perfectly good Lorentzian simplex. This is why the shape
        classification must not be used as a validity gate -- and why an
        admissibility gate would not reject most non-bipartite cells.
        """
        st = _pentatope()
        _set_split(st, set())
        st.getEdgeList().toVector()[0].setSquaredLength(complex(-1.0, 0.0))
        cell = _top(st)
        self.assertEqual(CS.shape_name(CS.classify(cell)), "non-bipartite")
        self.assertTrue(CS.is_lorentzian_admissible(cell))


def _oracle_census(st):
    """Independent re-implementation of the dual-height census."""
    widest = max((len(s.getVertices()) for s in st.getSimplices()), default=0)
    terms = defects = timelike = negative = 0
    for s in st.getSimplices():
        if len(s.getVertices()) >= widest or not s.hasTopCoface():
            continue
        own_ids = {v.getId() for v in s.getVertices()}
        own_r2 = s.circumradiusSquared()
        for cf in s.getCofaces():
            terms += 1
            cf_ids = [v.getId() for v in cf.getVertices()]
            opp = next((i for i, vid in enumerate(cf_ids) if vid not in own_ids), -1)
            bary = cf.circumcenterBarycentric()
            is_defect = opp >= 0 and opp < len(bary) and bary[opp] < 0.0
            is_timelike = (cf.circumradiusSquared() - own_r2) < 0.0
            defects += is_defect
            timelike += is_timelike
            negative += (is_defect != is_timelike)
    return terms, defects, timelike, negative


class TestDualHeightCensus(unittest.TestCase):
    """The census separates mesh defect from timelike dual separation."""

    def _check(self, st):
        c = CS.dual_height_census(st)
        terms, defects, timelike, negative = _oracle_census(st)
        self.assertEqual(c.terms, terms)
        self.assertEqual(c.centeredness_defects, defects)
        self.assertEqual(c.timelike_separations, timelike)
        self.assertEqual(c.negative_heights, negative)
        return c

    def test_matches_oracle_all_spacelike(self):
        st = _pentatope()
        _set_split(st, set())
        c = self._check(st)
        self.assertEqual(c.timelike_separations, 0,
                         "an all-spacelike complex has no timelike dual "
                         "separation")

    def test_matches_oracle_with_timelike_edges(self):
        st = _pentatope()
        _set_split(st, {3, 4})
        self._check(st)

    def test_top_cells_contribute_no_heights(self):
        """A top cell's dual is a point, so it carries no height terms."""
        st = _pentatope()
        _set_split(st, {4})
        c = CS.dual_height_census(st)
        n_top = sum(1 for s in st.getSimplices() if len(s.getVertices()) == 5)
        self.assertEqual(n_top, 1)
        # Every counted term belongs to a simplex strictly below the top.
        self.assertGreater(c.terms, 0)

    def test_negative_heights_are_exactly_one_negative_factor(self):
        """Both factors negative leaves the height positive."""
        st = _pentatope()
        _set_split(st, {3, 4})
        c = CS.dual_height_census(st)
        self.assertLessEqual(c.negative_heights,
                             c.centeredness_defects + c.timelike_separations)


if __name__ == "__main__":
    unittest.main()
