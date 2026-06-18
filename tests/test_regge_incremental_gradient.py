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


def _rel_worst(fresh, incr):
    # Relative agreement: a move or perturbation can land on a near-degenerate
    # cell whose dual volume (hence the gradient) is legitimately large; compare
    # against the gradient magnitude rather than an absolute floor.
    scale = max((abs(z) for z in fresh), default=1.0)
    return _worst(fresh, incr) / (1.0 + scale)


def _edge_keys(st):
    keys = []
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        keys.append((min(a, b), max(a, b)))
    return keys


# Each entry: (label, factory(st, seed) -> move).  AddMove is pinned to
# relabel=False so vertex IDs stay stable across apply()/rollback() (a cosmetic
# relabel re-keys edges outside the touched region and is out of scope).
_MOVE_FACTORIES = [
    ("shift", lambda st, s: tessera.ShiftMove(st, s)),
    ("flip", lambda st, s: tessera.FlipMove(st, s)),
    ("iflip", lambda st, s: tessera.IFlipMove(st, s)),
    ("add", lambda st, s: tessera.AddMove(st, s, False)),
    ("remove", lambda st, s: tessera.RemoveMove(st, s)),
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
    precision. Holds for every move type. (This is the real guarantee: the
    resident always equals what a full recompute would give NOW. Note it does
    NOT imply apply+rollback returns to the original baseline -- the from-scratch
    reference itself is not invariant under an arbitrary move's rollback, which
    only restores top cells/edges, not lower-dimensional hinge geometry to full
    precision. The resident tracks that reference faithfully either way.)"""

    def _run(self, label, factory):
        st = _grown_st()
        rs = _solver(st)
        rs.resetIncrementalGradient()

        m = _propose(factory, st)
        self.assertIsNotNone(m, f"{label}: no eligible move in seed budget")

        rs.applyMoveIncremental(m)
        self.assertLess(_rel_worst(_fresh_gradient(st), _incr_gradient(rs)),
                        _TOL, f"{label}: gradient diverged from exact after apply")
        act, fresh_act = complex(rs.incrementalAction()), _fresh_action(st)
        self.assertLess(abs(act - fresh_act), _TOL * (1 + abs(fresh_act)),
                        f"{label}: action diverged from exact after apply")

        rs.rollbackMoveIncremental(m)
        self.assertLess(_rel_worst(_fresh_gradient(st), _incr_gradient(rs)),
                        _TOL, f"{label}: gradient diverged from exact after rollback")
        act, fresh_act = complex(rs.incrementalAction()), _fresh_action(st)
        self.assertLess(abs(act - fresh_act), _TOL * (1 + abs(fresh_act)),
                        f"{label}: action diverged from exact after rollback")

    def test_each_move_type(self):
        for label, factory in _MOVE_FACTORIES:
            with self.subTest(move=label):
                self._run(label, factory)


class TestMoveSequence(unittest.TestCase):
    """A chain of kept moves tracks the fresh gradient at every step, and so
    does rolling the chain back (LIFO). The invariant checked is the contract --
    resident == fresh on the current complex -- not return-to-baseline (see
    TestSingleMoveTracking on why the reference is not rollback-invariant)."""

    def test_apply_chain_then_rollback(self):
        st = _grown_st()
        rs = _solver(st)
        rs.resetIncrementalGradient()

        applied = []
        seed = 0
        while len(applied) < 8 and seed < 8000:
            label, factory = _MOVE_FACTORIES[len(applied) % len(_MOVE_FACTORIES)]
            m = factory(st, seed)
            seed += 1
            if not m.propose():
                continue
            rs.applyMoveIncremental(m)
            applied.append((label, m))
            self.assertLess(_rel_worst(_fresh_gradient(st), _incr_gradient(rs)),
                            _TOL, f"step {len(applied)} ({label}): grad drift")
            act, fa = complex(rs.incrementalAction()), _fresh_action(st)
            self.assertLess(abs(act - fa), _TOL * (1 + abs(fa)),
                            f"step {len(applied)} ({label}): action drift")

        self.assertGreaterEqual(len(applied), 4,
                                "expected at least 4 accepted moves")

        # Roll the chain back in reverse (LIFO); the resident keeps tracking the
        # fresh gradient at every step.
        for label, m in reversed(applied):
            rs.rollbackMoveIncremental(m)
            self.assertLess(_rel_worst(_fresh_gradient(st), _incr_gradient(rs)),
                            _TOL, f"rollback ({label}): grad drift")


class TestLengthChange(unittest.TestCase):
    """``applyLengthChangeIncremental`` is the geometric counterpart of the move
    updates: setting one edge's squared length updates the resident
    gradient/action over only that edge's coface star, matching a from-scratch
    recompute."""

    def test_single_change_tracks_exact(self):
        st = _make_st()
        rs = _solver(st)
        rs.resetIncrementalGradient()
        edges = st.getEdgeList().toVector()
        # Perturb a spread of edges; each must keep the resident in lockstep
        # with a fresh exact recompute on the new geometry.
        for idx in range(0, len(edges), max(1, len(edges) // 12)):
            e = edges[idx]
            old = e.getSquaredLength()
            new = complex(old.real * 1.05, old.imag)  # preserve edge character
            rs.applyLengthChangeIncremental(e, new)
            self.assertLess(_rel_worst(_fresh_gradient(st),
                                            _incr_gradient(rs)), _TOL,
                            f"edge {idx}: gradient diverged after length change")
            self.assertLess(abs(complex(rs.incrementalAction()) -
                                _fresh_action(st)), _TOL * (1 + abs(_fresh_action(st))),
                            f"edge {idx}: action diverged after length change")

    def test_change_then_restore_returns_to_baseline(self):
        st = _make_st()
        rs = _solver(st)
        rs.resetIncrementalGradient()
        base_grad = _incr_gradient(rs)
        base_act = complex(rs.incrementalAction())

        e = st.getEdgeList().toVector()[7]
        old = e.getSquaredLength()
        rs.applyLengthChangeIncremental(e, complex(old.real * 1.1, old.imag))
        rs.applyLengthChangeIncremental(e, old)  # exact restore

        self.assertLess(_worst(base_grad, _incr_gradient(rs)), _TOL,
                        "gradient not restored after length round-trip")
        self.assertLess(abs(complex(rs.incrementalAction()) - base_act), _TOL,
                        "action not restored after length round-trip")

    def test_sequence_of_changes_tracks_exact(self):
        st = _make_st()
        rs = _solver(st)
        rs.resetIncrementalGradient()
        edges = st.getEdgeList().toVector()
        for k, idx in enumerate((3, 11, 11, 40, 3, 60)):  # repeats + overlaps
            e = edges[idx]
            old = e.getSquaredLength()
            rs.applyLengthChangeIncremental(e, complex(old.real * 1.07, old.imag))
            self.assertLess(_rel_worst(_fresh_gradient(st),
                                            _incr_gradient(rs)), _TOL,
                            f"step {k} (edge {idx}): gradient drift")

    def test_change_without_baseline_raises(self):
        st = _make_st()
        rs = _solver(st)
        with self.assertRaises(RuntimeError):
            rs.applyLengthChangeIncremental(st.getEdgeList().toVector()[0], 1.0)

    def test_change_footprint_is_size_independent(self):
        def changed_count(n_simplices):
            st = _make_st(n_simplices)
            rs = _solver(st)
            rs.resetIncrementalGradient()
            base = dict(zip(_edge_keys(st), _incr_gradient(rs)))
            e = st.getEdgeList().toVector()[5]
            old = e.getSquaredLength()
            rs.applyLengthChangeIncremental(e, complex(old.real * 1.1, old.imag))
            after = dict(zip(_edge_keys(st), _incr_gradient(rs)))
            changed = [k for k in after if abs(after[k] - base.get(k, 0)) > 1e-12]
            ek = (min(e.getSource().getId(), e.getTarget().getId()),
                  max(e.getSource().getId(), e.getTarget().getId()))
            self.assertIn(ek, changed, "the changed edge itself must move")
            return len(base), len(changed)

        small_edges, small_changed = changed_count(200)
        large_edges, large_changed = changed_count(900)
        self.assertGreater(large_edges, 3 * small_edges // 2)
        # Only the edge's coface star moves -- a small constant, not O(mesh).
        self.assertLess(small_changed, 60)
        self.assertLess(large_changed, 60)
        self.assertEqual(small_changed, large_changed)


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
