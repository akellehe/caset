# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""E and B from the temporal/spatial split of the field-strength 2-cochain (#417).

The field strength is a 2-cochain ``F in Omega^2``. Split it by the causal type of
each plaquette: the ELECTRIC part lives on plaquettes carrying a timelike edge (one
temporal leg, the discrete ``F_{0i}``); the MAGNETIC part on purely-spacelike
plaquettes (``F_{ij}``). The source is ``F = dA``, the discrete coboundary of the
carried U(1) connection 1-cochain ``A`` -- the same induced-orientation signed edge
sum the period read-out rides on (``examples/cobordism/proton_observables.py:55-56``).

The seed is the W_ABC junction (``TripartiteRegisterTopology``): a static
(all-spacelike, uniform ``l^2 = 1``) build has every plaquette magnetic; a Lorentzian
flux (``set_lorentzian_worldlines`` -> the cross-layer/forward-time edges timelike)
makes the cross-layer plaquettes electric. The split tracks the causal structure.

This is an observable-only read-out: ``fieldStrengthSplit`` / ``curvatureFromConnection``
classify cells by ``Edge.isTimelike()`` and partition a supplied ``F`` -- they never
mutate the geometry.
"""

import unittest

import numpy as np

import tessera

cob = tessera.cobordism

# Three color-neutral q-qbar pairs (Sigma = 0 each): the carriable W_ABC inputs.
_NEUTRAL_PAIRS = [[1, -1, 0], [1, 0, -1], [0, 1, -1]]
# Fixed RNG seed (no wall-clock, no global seed) -- determinism (G7).
_SEED = 417


def _build(lorentzian=None):
    """Build (no relax) the W_ABC junction; `lorentzian` (a negative l^2) sets the
    cross-layer worldline edges timelike for the flux regime."""
    trt = cob.TripartiteRegisterTopology()
    if lorentzian is not None:
        trt.set_lorentzian_worldlines(lorentzian)
    return cob.TransportCobordism(_NEUTRAL_PAIRS, max_iters=0, seed=0, topology=trt)


def _edge_order(st):
    """(map sorted-edge -> index, n_edges): the degree-1 cell order curvatureFromConnection
    indexes A in (== EigenstateSynthesis(st, 1).cellSimplices())."""
    es1 = cob.EigenstateSynthesis(st, 1)
    idx = {(min(c), max(c)): i for i, c in enumerate(es1.cellSimplices())}
    return idx, es1.order()


def _vertex_ids(st):
    return [c[0] for c in cob.EigenstateSynthesis(st, 0).cellSimplices()]


def _rng_cochain(rng, n):
    return list(rng.standard_normal(n) + 1j * rng.standard_normal(n))


def _edge_isolated_timelike_map(st):
    """sorted-edge (a,b) -> Edge.isTimelike(), read straight off the live EdgeList:
    the independent Python re-classification the C++ split is cross-checked against."""
    out = {}
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        out[(min(a, b), max(a, b))] = e.isTimelike()
    return out


def _python_electric_cells(st, es2):
    """The set of degree-2 cell indices with >= 1 timelike edge, recomputed in pure
    Python -- the cross-check, not a copy of the C++ result."""
    tl = _edge_isolated_timelike_map(st)
    elec = set()
    for i, (a, b, c) in enumerate(es2.cellSimplices()):
        if tl[(a, b)] or tl[(b, c)] or tl[(a, c)]:
            elec.add(i)
    return elec


def _edge_metric_snapshot(st):
    """Every edge's (Re l^2, Im l^2) -- the bit-for-bit geometry an observable-only
    read-out must leave untouched."""
    return [(e.getSquaredLength().real, e.getSquaredLength().imag)
            for e in st.getEdgeList().toVector()]


class FieldStrengthSplitStaticTest(unittest.TestCase):
    """Static (all-spacelike) seed -> E identically zero, B carries everything."""

    def test_static_seed_is_purely_magnetic(self):
        m = _build()
        st = m.cobordism
        es2 = cob.EigenstateSynthesis(st, 2)
        _, n_edges = _edge_order(st)
        rng = np.random.default_rng(_SEED)
        A = _rng_cochain(rng, n_edges)
        F = es2.curvatureFromConnection(A)
        self.assertEqual(len(F), es2.order())

        split = es2.fieldStrengthSplit(F)
        # E is empty: no plaquette has a timelike leg on the static seed.
        self.assertEqual(list(split.electricCells), [])
        # ||E|| is bit-zero (every component a real zero, not within a tolerance).
        self.assertEqual(np.linalg.norm(split.electric), 0.0)
        # B carries the entire structure, supported on all order() cells.
        self.assertEqual(len(split.magneticCells), es2.order())
        self.assertTrue(np.allclose(split.magnetic, F, atol=1e-12))

    def test_static_partition_is_complete(self):
        m = _build()
        es2 = cob.EigenstateSynthesis(m.cobordism, 2)
        _, n_edges = _edge_order(m.cobordism)
        rng = np.random.default_rng(_SEED)
        F = es2.curvatureFromConnection(_rng_cochain(rng, n_edges))
        split = es2.fieldStrengthSplit(F)
        total = np.array(split.electric) + np.array(split.magnetic)
        self.assertTrue(np.allclose(total, F, atol=1e-12))


class FieldStrengthSplitFluxTest(unittest.TestCase):
    """Timelike flux -> ||E|| > 0, the split tracks the causal type."""

    def test_flux_makes_worldlines_timelike(self):
        st = _build(-0.3).cobordism
        n_timelike = sum(1 for v in _edge_isolated_timelike_map(st).values() if v)
        self.assertGreater(n_timelike, 0)

    def test_flux_has_nonzero_electric_tracking_causal_type(self):
        m = _build(-0.3)
        st = m.cobordism
        es2 = cob.EigenstateSynthesis(st, 2)
        _, n_edges = _edge_order(st)
        rng = np.random.default_rng(_SEED)
        F = es2.curvatureFromConnection(_rng_cochain(rng, n_edges))
        split = es2.fieldStrengthSplit(F)

        # ||E|| strictly nonzero, well above round-off.
        self.assertGreater(np.linalg.norm(split.electric), 1e-9)
        # electricCells == the cells with a timelike leg, recomputed independently.
        self.assertEqual(set(split.electricCells), _python_electric_cells(st, es2))
        # disjoint + complete partition of the order() cells.
        e_set, m_set = set(split.electricCells), set(split.magneticCells)
        self.assertEqual(e_set & m_set, set())
        self.assertEqual(e_set | m_set, set(range(es2.order())))

    def test_flux_partition_is_complete(self):
        m = _build(-0.3)
        es2 = cob.EigenstateSynthesis(m.cobordism, 2)
        _, n_edges = _edge_order(m.cobordism)
        rng = np.random.default_rng(_SEED)
        F = es2.curvatureFromConnection(_rng_cochain(rng, n_edges))
        split = es2.fieldStrengthSplit(F)
        total = np.array(split.electric) + np.array(split.magnetic)
        self.assertTrue(np.allclose(total, F, atol=1e-12))


class CurvatureFromConnectionTest(unittest.TestCase):
    """F = dA: the coboundary signed edge sum and its gauge invariance."""

    def test_coboundary_on_a_single_triangle(self):
        # One triangle (0,1,2): F = A(0,1) + A(1,2) - A(0,2), the induced-orientation
        # signed edge sum (drop v_0 -> +(1,2), drop v_1 -> -(0,2), drop v_2 -> +(0,1)).
        st = tessera.Spacetime.fromCells(2, [[0, 1, 2]], 1.0, 0.0)
        es2 = cob.EigenstateSynthesis(st, 2)
        self.assertEqual(es2.order(), 1)
        idx, n_edges = _edge_order(st)
        self.assertEqual(n_edges, 3)
        A = [0.0] * 3
        A[idx[(0, 1)]] = 2.0 + 1.0j
        A[idx[(1, 2)]] = -0.5 + 3.0j
        A[idx[(0, 2)]] = 1.0 - 1.0j
        F = es2.curvatureFromConnection(A)
        expected = A[idx[(0, 1)]] + A[idx[(1, 2)]] - A[idx[(0, 2)]]
        self.assertAlmostEqual(F[0], expected, places=12)

    def test_gauge_invariance(self):
        # A' = A + d chi (the discrete gradient chi(b) - chi(a) on edge (a,b)); since
        # d.d = 0, F = dA is unchanged, so E and B are unchanged.
        m = _build(-0.3)
        st = m.cobordism
        es2 = cob.EigenstateSynthesis(st, 2)
        idx, n_edges = _edge_order(st)
        rng = np.random.default_rng(_SEED)
        A = _rng_cochain(rng, n_edges)

        chi = {v: complex(rng.standard_normal(), rng.standard_normal())
               for v in _vertex_ids(st)}
        A_prime = list(A)
        for (a, b), i in idx.items():
            A_prime[i] = A[i] + (chi[b] - chi[a])  # edges are sorted, a < b

        F = es2.curvatureFromConnection(A)
        F_prime = es2.curvatureFromConnection(A_prime)
        self.assertTrue(np.allclose(F_prime, F, atol=1e-10))

        s, sp = es2.fieldStrengthSplit(F), es2.fieldStrengthSplit(F_prime)
        self.assertTrue(np.allclose(sp.electric, s.electric, atol=1e-10))
        self.assertTrue(np.allclose(sp.magnetic, s.magnetic, atol=1e-10))


class FieldStrengthSplitErrorPathTest(unittest.TestCase):
    """Size / degree guards raise RuntimeError."""

    def test_wrong_F_size_raises(self):
        es2 = cob.EigenstateSynthesis(_build().cobordism, 2)
        with self.assertRaises(RuntimeError):
            es2.fieldStrengthSplit([0.0] * (es2.order() + 1))

    def test_non_degree_2_instance_raises(self):
        st = _build().cobordism
        es1 = cob.EigenstateSynthesis(st, 1)
        with self.assertRaises(RuntimeError):
            es1.fieldStrengthSplit([0.0] * es1.order())
        with self.assertRaises(RuntimeError):
            es1.curvatureFromConnection([0.0] * es1.order())

    def test_wrong_A_size_raises(self):
        st = _build().cobordism
        es2 = cob.EigenstateSynthesis(st, 2)
        _, n_edges = _edge_order(st)
        with self.assertRaises(RuntimeError):
            es2.curvatureFromConnection([0.0] * (n_edges + 1))


class ObservableOnlyRegressionTest(unittest.TestCase):
    """The reader is read-only: it leaves the junction geometry/topology unchanged."""

    def test_reader_does_not_perturb_the_geometry(self):
        m = _build(-0.3)
        st = m.cobordism
        # G4 TOPOLOGY: b1 == 11 and the dual complex is valid on the seed.
        self.assertEqual(list(m.stats.betti_cobordism)[1], 11)
        self.assertTrue(cob.EigenstateSynthesis(st, 1).dualComplexValid()[0])

        before = _edge_metric_snapshot(st)
        es2 = cob.EigenstateSynthesis(st, 2)
        _, n_edges = _edge_order(st)
        rng = np.random.default_rng(_SEED)
        F = es2.curvatureFromConnection(_rng_cochain(rng, n_edges))
        es2.fieldStrengthSplit(F)
        after = _edge_metric_snapshot(st)
        # bit-for-bit identical -- the read-out mutated nothing it should only read.
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
