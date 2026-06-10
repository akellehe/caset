"""One-forward-step Van Raamsdonk metric (#245).

Covers the three acceptance criteria:

  1. the spacelike VR metric law d_VR² = (−log(I/iMax))² with the
     I < ε·iMax floor — ``QuantumVertex.vanRaamsdonkSquaredLength``;
  2. the time-aware signed squared length — a forward-time *worldline* edge
     (different time slice) is null (squaredLength = 0), a same-slice edge is
     spacelike — ``QuantumVertex.vanRaamsdonkSquaredLengthTo``;
  3. fail-loudly admissibility — an inadmissible spacelike simplex raises, a
     valid one does not, and a simplex carrying a null/timelike (worldline)
     edge is skipped — ``QuantumSimplex.assertSpacelikeAdmissible``.

Continuous/spectral only; Lorentzian (signed squared lengths), not Euclidean.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import pytest

try:
    import tessera
    from tessera.quantum import QuantumSimplex, QuantumVertex
    _IMPORT_OK = True
except Exception:  # pragma: no cover
    _IMPORT_OK = False

pytestmark = pytest.mark.skipif(not _IMPORT_OK, reason="tessera not built")

IMAX = 2.0 * math.log(2.0)  # kIMax: maximally-entangled qubit pair


# --------------------------------------------------------------------------- #
# Fixtures (mirror the cobordism bulk-synthesis idiom).
# --------------------------------------------------------------------------- #
def _spacetime(dim, topology):
    sig = tessera.Signature(dim, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    return tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                             tessera.PREFERRED, topology)


def _solid_triangle():
    st = _spacetime(2, tessera.SolidSimplex(2))
    st.build()
    return st


def _edge_map(st):
    out = {}
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        out[(min(a, b), max(a, b))] = e
    return out


def _triangle(st):
    for s in st.getSimplices():
        if len(s.getEdges()) == 3:
            return s
    raise AssertionError("no triangle simplex in spacetime")


def _maximally_mixed_qubit():
    return np.array([[0.5, 0.0], [0.0, 0.5]], dtype=complex)


# --------------------------------------------------------------------------- #
# (1) The spacelike VR metric law.
# --------------------------------------------------------------------------- #
class TestMetricLaw(unittest.TestCase):
    def test_spacelike_matches_minus_log_squared(self):
        for frac in (0.9, 0.5, 0.1):
            I = frac * IMAX
            expected = (-math.log(I / IMAX)) ** 2
            got = QuantumVertex.vanRaamsdonkSquaredLength(I, IMAX, 1e-10)
            self.assertAlmostEqual(got, expected, places=12)

    def test_maximal_correlation_is_zero_length(self):
        # I = iMax -> d_VR = 0 -> squared length 0.
        self.assertAlmostEqual(
            QuantumVertex.vanRaamsdonkSquaredLength(IMAX, IMAX), 0.0, places=12)

    def test_floor_below_epsilon(self):
        for eps in (1e-10, 1e-3, 0.1):
            cap2 = (-math.log(eps)) ** 2
            # I = 0 and I just below eps*iMax both floor to (-log eps)^2.
            self.assertAlmostEqual(
                QuantumVertex.vanRaamsdonkSquaredLength(0.0, IMAX, eps),
                cap2, places=9)
            self.assertAlmostEqual(
                QuantumVertex.vanRaamsdonkSquaredLength(0.5 * eps * IMAX, IMAX, eps),
                cap2, places=9)
        # Just above the floor is NOT capped.
        eps = 0.1
        self.assertLess(
            QuantumVertex.vanRaamsdonkSquaredLength(2.0 * eps * IMAX, IMAX, eps),
            (-math.log(eps)) ** 2)


# --------------------------------------------------------------------------- #
# (2) Time-aware signed squared length: worldline edges are null.
# --------------------------------------------------------------------------- #
class TestForwardTimeNullEdges(unittest.TestCase):
    def test_worldline_edge_is_null(self):
        a = QuantumVertex(0, _maximally_mixed_qubit())
        b = QuantumVertex(1, _maximally_mixed_qubit())
        a.setTime(0.0)
        b.setTime(1.0)  # forward step: different slice -> worldline -> null
        self.assertEqual(a.vanRaamsdonkSquaredLengthTo(b, IMAX, 1e-10), 0.0)

    def test_same_slice_is_spacelike(self):
        a = QuantumVertex(0, _maximally_mixed_qubit())
        b = QuantumVertex(1, _maximally_mixed_qubit())
        a.setTime(0.0)
        b.setTime(0.0)  # same slice -> spacelike
        got = a.vanRaamsdonkSquaredLengthTo(b, IMAX, 1e-10)
        # Two product marginals have I = 0, so the spacelike length floors.
        self.assertAlmostEqual(got, (-math.log(1e-10)) ** 2, places=6)
        self.assertGreater(got, 0.0)  # spacelike, not null


# --------------------------------------------------------------------------- #
# (3) Fail-loudly admissibility of spacelike simplices.
# --------------------------------------------------------------------------- #
class TestSpacelikeAdmissibility(unittest.TestCase):
    def _triangle_with(self, s01, s02, s12):
        st = _solid_triangle()
        em = _edge_map(st)
        em[(0, 1)].setSquaredLength(s01)
        em[(0, 2)].setSquaredLength(s02)
        em[(1, 2)].setSquaredLength(s12)
        return _triangle(st)

    def test_valid_spacelike_triangle_ok(self):
        tri = self._triangle_with(1.0, 1.0, 1.0)  # equilateral, lengths 1
        QuantumSimplex.assertSpacelikeAdmissible(tri)  # must not raise

    def test_inadmissible_spacelike_triangle_raises(self):
        # lengths 1, 1, 3 -> 1 + 1 < 3: triangle inequality violated.
        tri = self._triangle_with(1.0, 1.0, 9.0)
        with self.assertRaises(RuntimeError):
            QuantumSimplex.assertSpacelikeAdmissible(tri)

    def test_worldline_edge_is_skipped(self):
        # One timelike (squaredLength < 0) edge: not a spacelike cell, so the
        # spacelike check is skipped (no raise) even though 1,1 + timelike would
        # not form a Euclidean triangle.
        tri = self._triangle_with(1.0, 1.0, -1.0)
        QuantumSimplex.assertSpacelikeAdmissible(tri)  # must not raise


if __name__ == "__main__":
    unittest.main()
