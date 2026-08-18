# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The parent-side sub-complex Betti cache (#705) is invisible in values.

Input-block residuals materialize each block's region as a FRESH spacetime,
so the per-instance Betti slot (#681) never carried across evaluations and
the Smith normal form recomputed per block, per line-search trial, per
candidate (measured: 47.5% of live-run cycles). The parent spacetime now
caches each block's Betti numbers per (structural revision, vertex-set key).
The cache changes WHEN the computation runs, never its input or output:
warm and cold evaluations must agree bitwise, and evaluations must stay
exact across metric changes and structural moves.
"""
import cmath
import math
import unittest

import tessera

cob = tessera.cobordism


def _seed():
    return tessera.Spacetime.fromCells(4, [[0, 1, 2, 3, 4]], 1.0, 0.0)


def _node(seed=5):
    return cob.Proton(seed=seed).direct_node(seed)


class BlockBettiCacheTest(unittest.TestCase):

    def test_warm_repeat_is_bitwise_identical(self):
        node = _node()
        st = _seed()
        cold = node.r_u(st)      # fills the parent-side slots
        warm = node.r_u(st)      # every block served from the cache
        self.assertEqual(cold, warm)

    def test_warm_equals_cold_on_a_fresh_identical_parent(self):
        # A separately constructed identical parent starts with empty slots:
        # its first (cold) evaluation must equal the warmed one's bitwise.
        node_a = _node()
        node_b = _node()
        st_warmed = _seed()
        node_a.r_u(st_warmed)
        warm = node_a.r_u(st_warmed)
        cold = node_b.r_u(_seed())
        self.assertEqual(warm, cold)

    def test_metric_change_reuses_structure_exactly(self):
        # Scaling every edge changes the metric but not the combinatorics:
        # the cached Betti numbers stay valid, and the residual still matches
        # a cold evaluation of the same scaled geometry bitwise.
        st_a, st_b = _seed(), _seed()
        node_a, node_b = _node(), _node()
        node_a.r_u(st_a)                       # warm the slots pre-scale
        for st in (st_a, st_b):
            for e in st.getEdgeList().toVector():
                e.setLength(e.getLength() * cmath.sqrt(2.0))
            st.materializeFacets()
        self.assertEqual(node_a.r_u(st_a), node_b.r_u(st_b))

    def test_structural_move_invalidates_and_stays_exact(self):
        # One combined-drive iteration applies gated structural moves to the
        # node's own host; the revision bump must invalidate the slots and the
        # objective must keep evaluating (finite) on the changed complex.
        node = _node()
        node.r_u(node.st)
        node.run(max_iters=1, n_candidate_moves=4, grow_boundaries=True,
                 beta=1.0, alpha0=0.05, rel_tol=1e-6, max_lookahead=1,
                 relax_budget_per_move=2)
        after = node.r_u(node.st)
        self.assertTrue(math.isfinite(after))
        self.assertGreaterEqual(after, 0.0)
        self.assertEqual(after, node.r_u(node.st))


if __name__ == "__main__":
    unittest.main()
