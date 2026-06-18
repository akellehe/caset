# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The incremental/local Regge gradient under Pachner moves
(:meth:`ReggeSolver.applyMoveIncremental` / ``rollbackMoveIncremental``) tracks a
from-scratch :meth:`ReggeSolver.actionGradientExact` (and ``dualReggeAction``) to
machine precision after any sequence of moves -- the regression guard for the
resident gradient that lets a combinatorial triangulation search pay
``O(#changed hinges)`` per move instead of ``O(H)``.

The resident gradient is keyed by vertex ID, so every move here is constructed in
the stable-ID regime (``AddMove(..., relabel=False)``); a cosmetic vertex relabel
re-keys edges outside the touched region and is deliberately out of scope (see
``ReggeSolver::applyMoveIncremental`` docs).
"""
import unittest

import tessera

# Analytic-vs-analytic agreement: the only slack is float summation order
# between the from-scratch pass and the maintained resident map.
_TOL = 1e-9


def _make_st(n_simplices=200, seed=0):
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED,
                           tessera.Toroid())
    st.build(n_simplices)
    st.setSeed(seed)
    st.materializeFacets()
    return st


def _grown_st(n_simplices=200, adds=150, seed=1):
    """A CDT lattice grown via ``CDTSimulation.add`` so the rarer moves
    (iflip / remove / shift) have eligible configurations."""
    st = _make_st(n_simplices, seed)
    cdt = tessera.CDTSimulation(st, 2.2, 0.5, 0.6, 0.02, st.getN41())
    for _ in range(adds):
        cdt.add()
    st.materializeFacets()
    return st


def _solver(st):
    return tessera.ReggeSolver(st, tessera.MatterConfiguration())


def _fresh_gradient(st):
    """A from-scratch exact gradient on the current (materialized) complex."""
    st.materializeFacets()
    return [complex(z) for z in _solver(st).actionGradientExact()]


def _fresh_action(st):
    st.materializeFacets()
    return complex(_solver(st).dualReggeAction())


def _incr_gradient(rs):
    return [complex(z) for z in rs.incrementalGradient()]


def _worst(a, b):
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def _edge_keys(st):
    keys = []
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        keys.append((min(a, b), max(a, b)))
    return keys


# Each entry: (label, factory(st, seed) -> move).  AddMove is pinned to
# relabel=False so vertex IDs stay stable across apply()/rollback().
#
# ``reference_invariant`` flags whether a raw apply()+rollback() of this move
# leaves ``actionGradientExact`` byte-for-byte unchanged. shift/flip/iflip/add
# do (~1e-16); RemoveMove does NOT (its rollback restores the top cells and
# edges but not the lower-dimensional hinge geometry to full precision, so the
# from-scratch reference itself drifts by O(1) -- a property of the move's
# rollback, independent of the incremental machinery). The per-step contract
# below (incremental == fresh actionGradientExact at every step) holds for ALL
# moves; only the full round-trip "restore to baseline" claim is move-dependent.
_MOVE_FACTORIES = [
    ("shift", lambda st, s: tessera.ShiftMove(st, s), True),
    ("flip", lambda st, s: tessera.FlipMove(st, s), True),
    ("iflip", lambda st, s: tessera.IFlipMove(st, s), True),
    ("add", lambda st, s: tessera.AddMove(st, s, False), True),
    ("remove", lambda st, s: tessera.RemoveMove(st, s), False),
]


def _propose(factory, st, budget=4000):
    """First successfully-proposing move from ``factory`` over seeds."""
    for seed in range(budget):
        m = factory(st, seed)
        if m.propose():
            return m
    return None


class TestBaseline(unittest.TestCase):
    """resetIncrementalGradient establishes a baseline equal to the
    from-scratch exact gradient / dual action."""

    def test_resident_matches_exact_gradient(self):
        st = _make_st()
        rs = _solver(st)
        rs.resetIncrementalGradient()
        exact = [complex(z) for z in rs.actionGradientExact()]
        incr = _incr_gradient(rs)
        self.assertEqual(len(incr), len(st.getEdgeList().toVector()))
        self.assertEqual(len(incr), len(exact))
        self.assertLess(_worst(exact, incr), _TOL)

    def test_resident_matches_dual_action(self):
        st = _make_st()
        rs = _solver(st)
        rs.resetIncrementalGradient()
        self.assertLess(abs(complex(rs.incrementalAction())
                            - complex(rs.dualReggeAction())), _TOL)

    def test_gradient_order_matches_edge_list(self):
        # Element i of the resident gradient is the edge at index i of
        # getEdgeList() -- the same contract actionGradientExact obeys.
        st = _make_st()
        rs = _solver(st)
        rs.resetIncrementalGradient()
        exact = [complex(z) for z in rs.actionGradientExact()]
        incr = _incr_gradient(rs)
        for i in range(len(exact)):
            self.assertAlmostEqual(exact[i].real, incr[i].real, places=8)
            self.assertAlmostEqual(exact[i].imag, incr[i].imag, places=8)

    def test_reset_is_idempotent(self):
        st = _make_st()
        rs = _solver(st)
        rs.resetIncrementalGradient()
        once = _incr_gradient(rs)
        rs.resetIncrementalGradient()
        twice = _incr_gradient(rs)
        self.assertEqual(_worst(once, twice), 0.0)

    def test_apply_without_baseline_raises(self):
        st = _make_st()
        rs = _solver(st)
        m = _propose(_MOVE_FACTORIES[0][1], st)
        self.assertIsNotNone(m)
        with self.assertRaises(RuntimeError):
            rs.applyMoveIncremental(m)

    def test_rollback_without_baseline_raises(self):
        st = _make_st()
        rs = _solver(st)
        m = _propose(_MOVE_FACTORIES[0][1], st)
        self.assertIsNotNone(m)
        with self.assertRaises(RuntimeError):
            rs.rollbackMoveIncremental(m)


class TestSingleMoveTracking(unittest.TestCase):
    """The core contract (issue #365): after a move's apply -- and again after
    its rollback -- the resident gradient/action equal a from-scratch
    actionGradientExact/dualReggeAction on the *current* complex, to machine
    precision. Holds for every move type."""

    def _run(self, label, factory):
        st = _grown_st()
        rs = _solver(st)
        rs.resetIncrementalGradient()

        m = _propose(factory, st)
        self.assertIsNotNone(m, f"{label}: no eligible move in seed budget")

        rs.applyMoveIncremental(m)
        self.assertLess(_worst(_fresh_gradient(st), _incr_gradient(rs)), _TOL,
                        f"{label}: gradient diverged from exact after apply")
        self.assertLess(abs(complex(rs.incrementalAction()) - _fresh_action(st)),
                        _TOL, f"{label}: action diverged from exact after apply")

        rs.rollbackMoveIncremental(m)
        self.assertLess(_worst(_fresh_gradient(st), _incr_gradient(rs)), _TOL,
                        f"{label}: gradient diverged from exact after rollback")
        self.assertLess(abs(complex(rs.incrementalAction()) - _fresh_action(st)),
                        _TOL, f"{label}: action diverged from exact after rollback")

    def test_each_move_type(self):
        for label, factory, _inv in _MOVE_FACTORIES:
            with self.subTest(move=label):
                self._run(label, factory)


class TestReversibility(unittest.TestCase):
    """For a move whose rollback is geometrically exact, apply()+rollback()
    returns the resident gradient/action to the byte-baseline -- i.e. the delta
    is exactly invertible, not merely tracking. (RemoveMove is excluded: its
    rollback leaves the from-scratch reference itself drifted, see
    ``_MOVE_FACTORIES``.)"""

    def _run(self, label, factory):
        st = _grown_st()
        rs = _solver(st)
        rs.resetIncrementalGradient()
        base_grad = _incr_gradient(rs)
        base_act = complex(rs.incrementalAction())

        m = _propose(factory, st)
        self.assertIsNotNone(m, f"{label}: no eligible move in seed budget")
        rs.applyMoveIncremental(m)
        rs.rollbackMoveIncremental(m)

        self.assertLess(_worst(base_grad, _incr_gradient(rs)), _TOL,
                        f"{label}: gradient not restored to baseline")
        self.assertLess(abs(complex(rs.incrementalAction()) - base_act), _TOL,
                        f"{label}: action not restored to baseline")

    def test_reversible_moves_restore_baseline(self):
        ran = 0
        for label, factory, invariant in _MOVE_FACTORIES:
            if not invariant:
                continue
            with self.subTest(move=label):
                self._run(label, factory)
            ran += 1
        self.assertGreaterEqual(ran, 4)


class TestMoveSequence(unittest.TestCase):
    """A chain of kept reference-invariant moves tracks the fresh gradient at
    every step; rolling the whole chain back (LIFO) returns to the baseline."""

    def test_apply_chain_then_rollback(self):
        st = _grown_st()
        rs = _solver(st)
        rs.resetIncrementalGradient()
        base_grad = _incr_gradient(rs)
        base_act = complex(rs.incrementalAction())

        # Interleave the reference-invariant move types so the full round trip
        # is byte-exact (a remove in the chain would drift the reference itself).
        invariant = [(lbl, fac) for lbl, fac, inv in _MOVE_FACTORIES if inv]
        applied = []
        seed = 0
        while len(applied) < 8 and seed < 8000:
            label, factory = invariant[len(applied) % len(invariant)]
            m = factory(st, seed)
            seed += 1
            if not m.propose():
                continue
            rs.applyMoveIncremental(m)
            applied.append((label, m))
            self.assertLess(_worst(_fresh_gradient(st), _incr_gradient(rs)),
                            _TOL, f"step {len(applied)} ({label}): grad drift")
            self.assertLess(
                abs(complex(rs.incrementalAction()) - _fresh_action(st)),
                _TOL, f"step {len(applied)} ({label}): action drift")

        self.assertGreaterEqual(len(applied), 4,
                                "expected at least 4 accepted moves")

        # Roll the chain back in reverse order; track at each step.
        for label, m in reversed(applied):
            rs.rollbackMoveIncremental(m)
            self.assertLess(_worst(_fresh_gradient(st), _incr_gradient(rs)),
                            _TOL, f"rollback ({label}): grad drift")

        # Full circle: exact baseline restored.
        self.assertLess(_worst(base_grad, _incr_gradient(rs)), _TOL,
                        "gradient not restored after full rollback")
        self.assertLess(abs(complex(rs.incrementalAction()) - base_act), _TOL,
                        "action not restored after full rollback")


class TestLocality(unittest.TestCase):
    """The update is local: a single move perturbs only a bounded number of
    edge gradients, and that number does NOT scale with total mesh size."""

    def _changed_count(self, n_simplices):
        st = _make_st(n_simplices)
        rs = _solver(st)
        rs.resetIncrementalGradient()
        base = dict(zip(_edge_keys(st), _incr_gradient(rs)))

        m = _propose(_MOVE_FACTORIES[0][1], st)  # shift: pure (3,3)
        self.assertIsNotNone(m)
        rs.applyMoveIncremental(m)

        after = dict(zip(_edge_keys(st), _incr_gradient(rs)))
        changed = [k for k in after if abs(after[k] - base.get(k, 0)) > 1e-12]
        return len(base), len(changed)

    def test_changed_edge_count_is_size_independent(self):
        # A 4D (3,3) shift rearranges a handful of cells: the perturbed-edge
        # count is a small constant even as the edge count grows ~4x.
        small_edges, small_changed = self._changed_count(200)
        large_edges, large_changed = self._changed_count(900)

        self.assertGreater(large_edges, 3 * small_edges // 2,
                           "meshes should differ enough to test scaling")
        self.assertLess(small_changed, 80)
        self.assertLess(large_changed, 80)
        # The footprint barely moves while the mesh nearly quadruples.
        self.assertLess(abs(large_changed - small_changed), 25)


if __name__ == "__main__":
    unittest.main()
