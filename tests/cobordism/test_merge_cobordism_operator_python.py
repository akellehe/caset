# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The clean C++ MergeCobordism (#388): ``tessera.cobordism.MergeCobordism``.

An emergent merge operator built from input/output qubit states through the
``(T^2 - 3 holes) x S^1`` operator topology, relaxed to a stationary point of
the dual Lorentzian Regge action under the state-pinning residual. These tests
pin the clean base's contract:

  * the topology it builds and the carried operator dimension
    ``ker L1(W - dW) = d^2 - 1`` (= 3 for a qubit),
  * the boundary / bulk / interior structure (all combinatorial, so exact and
    seed-independent),
  * that the relaxation descends from the bare seed to the intrinsic
    ``delta-S = 0`` floor, decomposes as ``beta*||grad S||^2 + r_psi``, and is
    deterministic for a fixed seed.

Distinct from ``test_merge_cobordism_python.py``, which tests the older Python
pair-of-pants example ``examples/cobordism/merge_cobordism.py`` (ker L1 = 2).
"""

import math
import unittest

import tessera

# d = 2 qubit: two input states on the boundary, one output. Built once and
# shared -- the 400-iteration relaxation is the expensive part.
_IN = [[1 + 0j, 0 + 0j], [0 + 0j, 1 + 0j]]
_OUT = [[1 + 0j, 0 + 0j]]
# A short relaxation budget keeps the suite runnable: the structural invariants
# are set after buildSeed (budget-independent), and the descent is steep early
# -- 30 LM steps already drop the residual ~300x from the bare seed. Production
# uses max_iters=400 (the delta-S=0 floor); the full-relax byte-identical
# descent is pinned by the period smoke, not here.
_ITERS = 30
_M = tessera.cobordism.MergeCobordism(_IN, _OUT, max_iters=_ITERS, seed=0)
_S = _M.stats


class TopologyStructureTest(unittest.TestCase):
    """The (T^2-3holes)xS^1 operator topology and its carried dimensions.
    Everything here is combinatorial -- seed-independent and exact."""

    def test_topology_name(self):
        self.assertTrue(_S.topology.startswith("(T^2-3holes)xS^1 operator"))
        self.assertIn("ker L1(W-dW)=3", _S.topology)

    def test_carried_operator_dimension(self):
        # ker L1(W - dW) = d^2 - 1 = 3 for a qubit (the Sigma=0 Choi dim).
        self.assertEqual(_S.ker_l1_bulk, 3)

    def test_bulk_first_betti(self):
        self.assertEqual(_S.b1_bulk, 5)

    def test_interior_vertex_count(self):
        self.assertEqual(_S.interior_vertices, 81)

    def test_boundary_and_bulk_cell_counts(self):
        self.assertEqual(len(_M.boundary), 54)
        self.assertEqual(len(_M.bulk), 483)

    def test_cobordism_spacetime_is_populated(self):
        W = _M.cobordism
        # interior + boundary vertices, all present in W.
        self.assertGreater(W.getVertexList().size(), _S.interior_vertices)
        self.assertGreater(len(W.getSimplices()), 0)

    def test_betti_array_consistent(self):
        b = _S.betti_cobordism
        self.assertGreaterEqual(len(b), 2)
        self.assertGreaterEqual(b[0], 1)        # at least one component
        self.assertEqual(b[1], _S.b1_bulk)      # b1_bulk is read off betti[1]


class RelaxationTest(unittest.TestCase):
    """The dual-Regge relaxation descends to the delta-S = 0 stationary floor."""

    def test_descends_well_below_seed(self):
        # the bare seed starts at r ~ 162; 30 LM steps drop it below 1.
        self.assertLess(_S.residual, 1.0)

    def test_floors_above_zero(self):
        # delta-S = 0 is a stationary point, not a minimum -- r does not vanish.
        self.assertGreater(_S.residual, 0.0)

    def test_residual_decomposes(self):
        # beta = 1: r = beta*||grad S||^2 + r_state.
        self.assertAlmostEqual(
            _S.residual, _S.stat_action_residual + _S.state_residual, places=9)

    def test_state_residual_near_zero(self):
        # The default r_state term is r_U (realizability); the basis input/output
        # states are realizable, so it sits near zero.
        self.assertEqual(_S.state_mode, "r_U")
        self.assertLess(_S.state_residual, 1e-2)

    def test_runs_the_iteration_budget(self):
        self.assertEqual(_S.relax_iterations, _ITERS)

    def test_not_converged_below_epsilon(self):
        # the floor (~0.075) is above the 1e-6 tolerance, so converged is False.
        self.assertFalse(_S.converged)

    def test_dual_action_is_finite(self):
        # The dual Regge action is well-defined (finite) at the relaxed metric.
        # We deliberately do NOT assert Im == 0: the action is complex by
        # construction (Lorentzian / Sorkin) and the imaginary part is real
        # physics (spacelike-hinge boosts). It happens to come out ~0 for this
        # all-spacelike emergent seed, but pinning Im == 0 would bake in a
        # Euclidean expectation and forbid a genuine timelike emergence -- so we
        # require only finiteness, leaving Im free.
        self.assertTrue(math.isfinite(_S.dual_action.real))
        self.assertTrue(math.isfinite(_S.dual_action.imag))


class StatePassThroughTest(unittest.TestCase):
    """Inputs/outputs are preserved verbatim (emergent mode)."""

    def test_input_states_preserved(self):
        self.assertEqual(_M.input_states, _IN)

    def test_output_states_preserved(self):
        self.assertEqual(_M.output_states, _OUT)


class OperatorRecoveryDeferredTest(unittest.TestCase):
    """Operator read-out is deferred to #6; the accessors are empty for now.
    Update this test when ``reshape(ker L1(W-dW)) == U`` lands."""

    def test_operator_u_empty(self):
        self.assertEqual(len(_M.operator_U), 0)

    def test_choi_state_empty(self):
        self.assertEqual(len(_M.choi_state), 0)


class DeterminismTest(unittest.TestCase):
    """A fixed seed gives a bit-identical relaxation (reproducible)."""

    @classmethod
    def setUpClass(cls):
        cls.m2 = tessera.cobordism.MergeCobordism(_IN, _OUT, max_iters=_ITERS, seed=0)

    def test_same_seed_same_residual(self):
        self.assertEqual(self.m2.stats.residual, _S.residual)

    def test_same_seed_same_structure(self):
        self.assertEqual(self.m2.stats.ker_l1_bulk, _S.ker_l1_bulk)
        self.assertEqual(self.m2.stats.interior_vertices, _S.interior_vertices)
        self.assertEqual(len(self.m2.bulk), len(_M.bulk))


class StateResidualModeTest(unittest.TestCase):
    """The selectable r_state term (#377): r_U realizability (default) vs r_psi
    hard period-pin. The topology/structure is the term-independent substrate;
    only the matter term and its descent differ."""

    _MODE = tessera.cobordism.MergeCobordism.StateResidualMode

    @classmethod
    def setUpClass(cls):
        # The default merge (_M) is r_U; build the r_psi counterpart once.
        cls.m_pin = tessera.cobordism.MergeCobordism(
            _IN, _OUT, max_iters=_ITERS, seed=0,
            state_mode=cls._MODE.PeriodPin)

    def test_default_is_realizability(self):
        # No state_mode argument => r_U (the #377 default).
        self.assertEqual(_S.state_mode, "r_U")

    def test_period_pin_is_selected(self):
        self.assertEqual(self.m_pin.stats.state_mode, "r_psi")

    def test_period_pin_descends_to_a_finite_floor(self):
        # r_psi also descends well below the bare seed (~162) and floors > 0.
        s = self.m_pin.stats
        self.assertGreater(s.residual, 0.0)
        self.assertLess(s.residual, 2.0)  # ~300x below the bare seed
        self.assertAlmostEqual(
            s.residual, s.stat_action_residual + s.state_residual, places=9)

    def test_mode_does_not_change_the_topology(self):
        # The term is the matter pin, not the geometry: same carried dim / bulk.
        self.assertEqual(self.m_pin.stats.ker_l1_bulk, _S.ker_l1_bulk)
        self.assertEqual(self.m_pin.stats.interior_vertices, _S.interior_vertices)

    def test_period_pin_is_deterministic(self):
        m2 = tessera.cobordism.MergeCobordism(
            _IN, _OUT, max_iters=_ITERS, seed=0, state_mode=self._MODE.PeriodPin)
        self.assertEqual(m2.stats.residual, self.m_pin.stats.residual)


if __name__ == "__main__":
    unittest.main()
