"""RemoveMove rollback restores the full complex l^2 AND the U(1) phase (#581).

``RemoveMove::EdgeRecord`` was a plain double: rollback re-created the deleted
vertex's edges with ``Re(l^2)`` only and phase 0, silently projecting
analytically continued geometry onto the real axis and erasing the connection
phase on every rejected/rolled-back move.  Both capture paths are covered:

* the CDT ``apply()`` path (an order-2d vertex on a grown toroid);
* the pre-geometric ``applyPreGeometric()`` path (the (d+1)->1 stellar weld
  of a vertex added by the pre-geometric ``AddMove``).

Each test seeds every edge with a distinct synthetic ``Im l^2 != 0`` and
``phase != 0``, applies the move, rolls it back, and asserts every edge's
``getSquaredLength()`` (Re and Im) and ``getPhase()`` are bit-exact.  The
rollback path is storage-level (no geometry evaluation), so the synthetic Im
never meets the geometry stack.  A real signed-l^2 (timelike-carrying) action
round trip on the CDT toroid extends the hinge-exactness contract.
"""

import unittest

import pytest
import cmath

try:
    import tessera
    _IMPORT_OK = True
except Exception:  # pragma: no cover
    _IMPORT_OK = False

pytestmark = pytest.mark.skipif(not _IMPORT_OK, reason="tessera not built")

PRE = None if not _IMPORT_OK else tessera.PachnerMode.PreGeometric


def _seed_complex_geometry(st):
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        lo, hi = min(a, b), max(a, b)
        e.setLength(cmath.sqrt(complex(complex(1.0 + 0.001 * lo, 0.02 + 0.001 * hi))))
        e.setPhase(0.05 + 0.002 * (lo * 7 + hi))


def _edge_state(st):
    out = {}
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        out[(min(a, b), max(a, b))] = ((complex(e.getLength() * complex(e.getLength())),
                                       e.getPhase())
    return out


def _tops(st):
    return sorted(tuple(sorted(v.getId() for v in s.getVertices()))
                  for s in st.getTopSimplices())


def _assert_state_equal(before, after):
    assert set(before) == set(after), (
        f"edge set drifted: only-before={set(before) - set(after)} "
        f"only-after={set(after) - set(before)}")
    for k, (sq, ph) in after.items():
        sq0, ph0 = before[k]
        assert sq == sq0, f"edge {k}: l^2 {sq0!r} -> {sq!r} (not bit-exact)"
        assert ph == ph0, f"edge {k}: phase {ph0!r} -> {ph!r} (not bit-exact)"


def _grown_cdt_with_removable_vertex(d=4, n_simplices=60,
                                     batch=40, max_batches=25):
    """The test_pachner_remove_move growth-and-probe recipe: grow a CDT
    toroid until RemoveMove.propose() can find an order-2d vertex."""
    sig = tessera.Signature(d, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    st.build(n_simplices)
    cdt = tessera.CDTSimulation(st, 2.2, 0.5, 0.6, 0.02, st.getN41())
    for _ in range(max_batches):
        for _ in range(batch):
            cdt.add()
        for s in range(200):
            if tessera.RemoveMove(st, s).propose():
                return st
    raise RuntimeError("no removable vertex after growth")


def _pregeometric_host():
    """SolidSimplex(3) with one pre-geometric 1->(d+1) add applied: the added
    interior vertex is exactly the (d+1)->1 weld's target."""
    sig = tessera.Signature(3, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, tessera.SolidSimplex(3))
    st.build()
    for seed in range(50):
        m = tessera.AddMove(st, seed, False, PRE, False)
        if m.propose() and m.apply():
            return st
    raise RuntimeError("pre-geometric add never fired")


class TestRemoveMoveComplexRollback(unittest.TestCase):
    def test_cdt_apply_rollback_restores_complex_l2_and_phase(self):
        st = _grown_cdt_with_removable_vertex()
        _seed_complex_geometry(st)
        before = _edge_state(st)
        tops_before = _tops(st)

        applied = None
        for seed in range(400):
            m = tessera.RemoveMove(st, seed)
            if m.propose() and m.apply():
                applied = m
                break
        self.assertIsNotNone(applied, "no CDT remove move fired")
        self.assertNotEqual(_edge_state(st), before)

        applied.rollback()
        _assert_state_equal(before, _edge_state(st))
        self.assertEqual(_tops(st), tops_before)

    def test_pregeometric_apply_rollback_restores_complex_l2_and_phase(self):
        st = _pregeometric_host()
        _seed_complex_geometry(st)
        before = _edge_state(st)
        tops_before = _tops(st)

        applied = None
        for seed in range(200):
            m = tessera.RemoveMove(st, seed, PRE, False)
            if m.propose() and m.apply():
                applied = m
                break
        self.assertIsNotNone(applied, "pre-geometric remove never fired")

        applied.rollback()
        _assert_state_equal(before, _edge_state(st))
        self.assertEqual(_tops(st), tops_before)

    def test_cdt_rollback_preserves_complex_action_on_lorentzian_host(self):
        # Real signed l^2 (the CDT toroid's native timelike-carrying metric):
        # the rollback must retrace the full complex dual Regge action.
        st = _grown_cdt_with_removable_vertex()
        solver = tessera.ReggeSolver(st, tessera.MatterConfiguration())
        s0 = complex(solver.dualReggeAction())
        self.assertGreater(abs(s0.imag), 1e-6,
                           "fixture is not genuinely Lorentzian")

        applied = None
        for seed in range(400):
            m = tessera.RemoveMove(st, seed)
            if m.propose() and m.apply():
                applied = m
                break
        self.assertIsNotNone(applied, "no CDT remove move fired")
        applied.rollback()

        s1 = complex(
            tessera.ReggeSolver(st, tessera.MatterConfiguration())
            .dualReggeAction())
        self.assertLess(abs(s1.real - s0.real), 1e-6)
        self.assertLess(abs(s1.imag - s0.imag), 1e-6)


if __name__ == "__main__":
    unittest.main()
