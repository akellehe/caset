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
    deterministic for a fixed seed,
  * the bidirectional emergence (#376): the primary emergent quantity is the one
    the caller did not supply -- the emergent ``output_state`` (the inputs carried
    through the relaxed geometry, read over the output cycles) is populated in
    both modes, U-only construction is legal, and the operator read-out stays
    deferred (empty).

Distinct from ``test_merge_cobordism_python.py``, which tests the older Python
pair-of-pants example ``examples/cobordism/merge_cobordism.py`` (ker L1 = 2).
"""

import math
import unittest

import numpy as np

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

# U-supplied mode (no output_states): the OUTPUT state is the primary emergent
# quantity. One input |0> and the identity operator, so the U-computed output
# (the pinned target) is |0>.
_U_IDENTITY = [1 + 0j, 0j, 0j, 1 + 0j]
_MU = tessera.cobordism.MergeCobordism(
    [[1 + 0j, 0j]], U=_U_IDENTITY, max_iters=_ITERS, seed=0)

# 2 inputs + U => 2 U-computed outputs => 4 states > 3 holes, so the input/output
# cycle split is indeterminate and the emergent output read is skipped (honest).
# The empty read is iters-independent, so a minimal budget keeps it cheap.
_M_2IN_U = tessera.cobordism.MergeCobordism(
    _IN, U=[0j, 1 + 0j, 1 + 0j, 0j], max_iters=2, seed=0)


def _normalized_phase_fixed(vec):
    """Unit-normalize a complex amplitude vector and fix its global phase by the
    largest component, so vectors equal up to (scale, global phase) compare
    equal -- the emergent output is returned unnormalized and up to a phase."""
    a = np.array(vec, dtype=complex)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return a
    a = a / n
    k = int(np.argmax(np.abs(a)))
    return a * np.conj(a[k]) / abs(a[k])


def _fidelity(emergent, target):
    """|<target|emergent>| with both unit-normalized (emergent phase-fixed)."""
    u = _normalized_phase_fixed(emergent)
    v = np.array(target, dtype=complex)
    return abs(complex(np.vdot(v / np.linalg.norm(v), u)))


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
        # beta = 1: r = beta*||grad S||^2 + r_psi.
        self.assertAlmostEqual(
            _S.residual, _S.stat_action_residual + _S.state_residual, places=9)

    def test_states_are_pinned_hard(self):
        # r_psi is the (hard) state-pinning term; it sits near zero.
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


class OutputEmergenceTest(unittest.TestCase):
    """The emergent output state (#376): the inputs carried through the relaxed
    geometry, read over the output cycles -- the #353 inputs->emergent-output
    flow. Populated in both modes (primary in U-supplied; a consistency read when
    the output was supplied)."""

    def test_output_state_populated(self):
        # one qubit output = d = 2 periods.
        self.assertEqual(len(_M.output_state), 2)

    def test_output_state_finite(self):
        for z in _M.output_state:
            self.assertTrue(math.isfinite(z.real) and math.isfinite(z.imag))

    def test_output_supplied_emergent_matches_supplied(self):
        # output-supplied mode: the emergent output (carry inputs -> read output
        # cycles), normalized and global-phase-fixed, reproduces the supplied |0>.
        # Returned unnormalized (the period scale) and up to a global phase.
        self.assertGreater(_fidelity(_M.output_state, _OUT[0]), 0.9)

    def test_u_supplied_no_output_is_legal(self):
        # U supplied, output_states omitted: construction is legal (output_states
        # defaults to empty and is computed from U) and the output emerges.
        self.assertEqual(len(_MU.output_state), 2)

    def test_u_supplied_output_states_computed_from_U(self):
        # the (plural) outputStates echo is U applied to each input: I|0> = |0>.
        self.assertEqual(_MU.output_states, [[1 + 0j, 0j]])

    def test_u_supplied_output_tracks_input_transport(self):
        # with the operator-as-bulk-constraint deferred, the transport is ~identity
        # on this topology, so the emergent output tracks the carried input |0>.
        self.assertGreater(_fidelity(_MU.output_state, [1 + 0j, 0j]), 0.9)

    def test_indeterminate_split_skips_output(self):
        # 2 inputs + U => 4 states > 3 holes: the input/output cycle split is
        # indeterminate, so the output read is skipped (empty) rather than guessed.
        self.assertEqual(len(_M_2IN_U.output_state), 0)


class OperatorDeferredTest(unittest.TestCase):
    """The operator read-out unvec(ker L1(W-dW)) is deferred (#376): on this
    topology ker L1(W-dW) is a (d^2-1)-dim interior-cochain subspace with no
    basis-independent map to the dxd operator (it needs distinguished interior
    Choi-cycles the topology does not yet supply), so the accessors stay empty
    rather than report a frame-dependent value. Update when the interior-handle
    operator-topology rework lands."""

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

    def test_same_seed_same_output_state(self):
        self.assertEqual(list(self.m2.output_state), list(_M.output_state))


if __name__ == "__main__":
    unittest.main()
