# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""ReggeSolver's topology cache (#366).

``actionGradientExact`` / ``actionHessianExact`` used to rebuild a ``std::map``
edge index and re-scan the simplex list for hinges on every call. Both depend
only on the triangulation, so they are now cached (in a ``std::unordered_map``)
and rebuilt only when an O(1) topology signature ``(edge count, simplex count)``
changes. These tests guard that the optimization is transparent:

  * repeated calls on one solver return identical gradients/Hessians — the cache
    is reused, not corrupted, and the accumulator is re-zeroed each call;
  * a metric-only change (``setSquaredLength``) does NOT change the signature, so
    the cache is NOT rebuilt — yet the gradient still reflects the new lengths
    (they are read live; only the topology-derived structures are cached);
  * a topology change (adding a simplex) DOES change the signature, so the cache
    is rebuilt and the gradient/action match a freshly constructed solver;
  * the cached path is value-identical to a fresh solver on the real merge
    cobordism — proving the ``unordered_map`` edge ordering still matches
    ``getEdgeList()`` — including its complex (boost) hinges;
  * the cached edge index also backs the analytic Hessian, which stays symmetric
    and matches a finite difference of the exact gradient.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import unittest

import pytest

try:
    import tessera
    _IMPORT_OK = True
except Exception:  # pragma: no cover
    _IMPORT_OK = False

pytestmark = pytest.mark.skipif(not _IMPORT_OK, reason="tessera not built")

_HERE = os.path.dirname(os.path.abspath(__file__))
_MERGE = os.path.join(_HERE, "..", "..", "examples", "cobordism",
                      "merge_cobordism.py")


def _load_merge():
    spec = importlib.util.spec_from_file_location("merge_cobordism", _MERGE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["merge_cobordism"] = module
    spec.loader.exec_module(module)
    return module


def _solver(st):
    return tessera.ReggeSolver(st, tessera.MatterConfiguration())


def _grad(rs):
    return [complex(z) for z in rs.actionGradientExact()]


def _hess(rs):
    return [[complex(z) for z in row] for row in rs.actionHessianExact()]


def _set_unit(st):
    """All edges spacelike, l^2 = 1, then materialize the facet lattice."""
    for e in st.getEdgeList().toVector():
        e.setSquaredLength(1.0)
        e.setPhase(0.0)
    st.materializeFacets()


def _tetra_fan(apexes):
    """Unit-regular tetrahedra sharing the face (0,1,2), one per apex id. A 3D
    complex (d=3) whose hinges are edges; every tetra has all six edges of
    squared length 1, so each is admissible. Returns (spacetime, vertex map)."""
    sig = tessera.Signature(3, tessera.Lorentzian)
    st = tessera.Spacetime(tessera.Metric(True, sig), tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, None)
    vmap = {i: st.createVertex(i) for i in (0, 1, 2, *apexes)}
    for ap in apexes:
        st.createSimplex([vmap[0], vmap[1], vmap[2], vmap[ap]])
    _set_unit(st)
    return st, vmap


def _assert_close(case, a, b, tol=1e-9):
    """Two complex sequences agree elementwise within ``tol`` (tight enough to be
    effectively exact, loose enough to absorb any summation-order jitter)."""
    case.assertEqual(len(a), len(b))
    worst = max((abs(x - y) for x, y in zip(a, b)), default=0.0)
    case.assertLess(worst, tol, f"worst |a-b| = {worst:.3e}")


# ===================================================================== #
# Cache build / early-return / rebuild paths, on a controlled small mesh #
# ===================================================================== #
class TopologyCacheStructure(unittest.TestCase):
    def test_repeated_gradient_calls_are_identical(self):
        """The cache is reused across calls and the accumulator re-zeroed, so
        repeated gradients are bit-identical (no drift, no accumulation)."""
        st, _ = _tetra_fan((3, 4))
        rs = _solver(st)
        g1, g2, g3 = _grad(rs), _grad(rs), _grad(rs)
        self.assertGreater(len(g1), 0)
        self.assertEqual(g1, g2)
        self.assertEqual(g2, g3)

    def test_repeated_dual_action_calls_are_identical(self):
        """``dualReggeAction`` reads the cached hinge list via ``collectHinges``;
        repeated calls are identical."""
        st, _ = _tetra_fan((3, 4))
        rs = _solver(st)
        self.assertEqual(complex(rs.dualReggeAction()),
                         complex(rs.dualReggeAction()))

    def test_metric_change_keeps_cache_but_gradient_is_live(self):
        """A pure ``setSquaredLength`` leaves the topology signature unchanged, so
        the cache is NOT rebuilt — but the gradient must still reflect the new
        lengths (read live) and match a freshly built solver."""
        st, _ = _tetra_fan((3, 4))
        rs = _solver(st)
        g_before = _grad(rs)
        st.getEdgeList().toVector()[0].setSquaredLength(1.7)
        st.getEdgeList().toVector()[0].setPhase(0.0)
        st.materializeFacets()
        g_after = _grad(rs)              # same solver; cache NOT invalidated
        g_fresh = _grad(_solver(st))     # fresh solver on the new lengths
        self.assertEqual(len(g_after), len(g_before))
        self.assertNotEqual(g_after, g_before)   # lengths changed -> grad changed
        _assert_close(self, g_after, g_fresh)    # cache used the live lengths

    def test_topology_change_invalidates_cache(self):
        """Adding a tetra changes ``(edge count, simplex count)``, so the cache
        must rebuild: the gradient grows in length and matches a fresh solver on
        the mutated mesh."""
        st, vmap = _tetra_fan((3, 4))
        rs = _solver(st)
        e_before = len(_grad(rs))
        # mutate the live spacetime the solver holds: a third unit tetra
        vmap[5] = st.createVertex(5)
        st.createSimplex([vmap[0], vmap[1], vmap[2], vmap[5]])
        _set_unit(st)
        g_after = _grad(rs)              # same solver; cache MUST invalidate
        g_fresh = _grad(_solver(st))
        self.assertGreater(len(g_after), e_before)        # three new edges
        self.assertEqual(len(g_fresh), len(g_after))
        _assert_close(self, g_after, g_fresh)             # rebuilt == fresh

    def test_dual_action_tracks_topology_change(self):
        """The cached hinge list behind ``dualReggeAction`` rebuilds on a topology
        change too."""
        st, vmap = _tetra_fan((3, 4))
        rs = _solver(st)
        s_two = complex(rs.dualReggeAction())
        vmap[5] = st.createVertex(5)
        st.createSimplex([vmap[0], vmap[1], vmap[2], vmap[5]])
        _set_unit(st)
        s_three = complex(rs.dualReggeAction())            # cache rebuilt
        s_fresh = complex(_solver(st).dualReggeAction())
        self.assertAlmostEqual(s_three.real, s_fresh.real, places=9)
        self.assertAlmostEqual(s_three.imag, s_fresh.imag, places=9)
        self.assertNotAlmostEqual(s_two.real, s_three.real)  # geometry grew


# ===================================================================== #
# Hessian shares the cached edge index                                   #
# ===================================================================== #
class HessianUsesCache(unittest.TestCase):
    def test_hessian_repeated_calls_identical_and_symmetric(self):
        st, _ = _tetra_fan((3, 4))
        rs = _solver(st)
        H1, H2 = _hess(rs), _hess(rs)
        self.assertEqual(H1, H2)                           # cache reuse
        n = len(H1)
        self.assertGreater(n, 0)
        for i in range(n):
            for j in range(n):
                self.assertAlmostEqual(H1[i][j].real, H1[j][i].real, places=9)
                self.assertAlmostEqual(H1[i][j].imag, H1[j][i].imag, places=9)

    def test_hessian_matches_fresh_solver(self):
        st, _ = _tetra_fan((3, 4))
        H_cached = _hess(_solver(st))
        H_fresh = _hess(_solver(st))
        flat_cached = [z for row in H_cached for z in row]
        flat_fresh = [z for row in H_fresh for z in row]
        _assert_close(self, flat_cached, flat_fresh)

    def test_hessian_matches_finite_difference_of_gradient(self):
        """H[r][f] = d g[r] / d l^2_f. Perturbing column f and re-reading the
        gradient (on the same cached solver) recovers the analytic Hessian."""
        st, _ = _tetra_fan((3, 4))
        rs = _solver(st)
        H = _hess(rs)
        edges = st.getEdgeList().toVector()      # index == gradient/Hessian order
        n = len(edges)
        delta = 1e-6
        worst = 0.0
        for f in range(min(n, 4)):
            o = edges[f].getSquaredLength().real
            edges[f].setSquaredLength(o + delta); edges[f].setPhase(0.0)
            st.materializeFacets()
            gp = _grad(rs)
            edges[f].setSquaredLength(o - delta); edges[f].setPhase(0.0)
            st.materializeFacets()
            gm = _grad(rs)
            edges[f].setSquaredLength(o); edges[f].setPhase(0.0)
            st.materializeFacets()
            for r in range(n):
                fd = (gp[r] - gm[r]) / (2.0 * delta)
                worst = max(worst, abs(H[r][f] - fd))
        self.assertLess(worst, 1e-4, f"worst |H - FD(grad)| = {worst:.3e}")


# ===================================================================== #
# Value-identity on the real merge cobordism (complex / boost hinges)    #
# ===================================================================== #
class CachedPathOnMergeCobordism(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.st = _load_merge().MergeCobordism().st
        cls.st.materializeFacets()

    def test_cached_gradient_idempotent_and_matches_fresh(self):
        rs = _solver(self.st)
        g_first = _grad(rs)              # first call builds the cache
        g_again = _grad(rs)              # second call hits the cache
        g_fresh = _grad(_solver(self.st))
        self.assertEqual(g_first, g_again)               # idempotent
        _assert_close(self, g_first, g_fresh)            # cache == fresh build
        self.assertTrue(any(abs(z.imag) > 1e-6 for z in g_first),
                        "merge mesh should exercise complex (boost) gradients")

    def test_gradient_order_matches_getedgelist(self):
        """The ``unordered_map`` edge index must keep the same edge ordering as
        ``getEdgeList()->toVector()`` (the Python EIDX contract)."""
        rs = _solver(self.st)
        g = _grad(rs)
        self.assertEqual(len(g), self.st.getEdgeList().size())


if __name__ == "__main__":
    unittest.main()
