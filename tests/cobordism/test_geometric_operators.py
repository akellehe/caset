# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Acceptance tests for residual-only geometric-operator identification."""

import importlib.util
import os
import sys
import unittest

import numpy as np

import tessera


_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.join(
    _HERE, "..", "..", "examples", "cobordism",
    "geometric_operators.py",
)
_SPEC = importlib.util.spec_from_file_location(
    "geometric_operators", _EXAMPLE)
GO = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = GO
_SPEC.loader.exec_module(GO)

_ORIGINAL_CONVENTION = None


def setUpModule():
    global _ORIGINAL_CONVENTION
    _ORIGINAL_CONVENTION = (
        tessera.cobordism.HodgeLaplacian.defaultWeightConvention())
    tessera.cobordism.HodgeLaplacian.setDefaultWeightConvention(
        tessera.cobordism.HodgeWeightConvention.SquaredContent)


def tearDownModule():
    tessera.cobordism.HodgeLaplacian.setDefaultWeightConvention(
        _ORIGINAL_CONVENTION)


class ExplicitConstraintTest(unittest.TestCase):
    def test_constraints_are_ordered_replaceable_and_clearable(self):
        fill = GO.PeriodCobordism()
        first = fill.pin_state(
            "selected", GO.identity_operator(), np.array([1.0, 0.0]))
        records = fill.node.register_constraints()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["name"], "selected")
        self.assertEqual(
            records[0]["holes"], [sorted(hole) for hole in fill.holes])
        np.testing.assert_allclose(records[0]["target"], first)
        expected = fill.synthesis.residualForPeriods(fill.holes, first)
        self.assertAlmostEqual(
            fill.node.r_u(fill.node.spacetime()), expected, places=14)

        second = fill.pin_state(
            "selected", GO.identity_operator(), np.array([0.0, 1.0]))
        records = fill.node.register_constraints()
        self.assertEqual(len(records), 1)
        np.testing.assert_allclose(records[0]["target"], second)

        fill.node.clear_register_constraints()
        self.assertEqual(fill.node.register_constraints(), [])
        self.assertEqual(fill.node.r_u(fill.node.spacetime()), 0.0)

    def test_constraint_validation_and_flags(self):
        fill = GO.PeriodCobordism()
        self.assertFalse(fill.node.einstein_hilbert_enabled)
        self.assertTrue(fill.node.real_squared_lengths_only)
        legacy = tessera.cobordism.MultiCobordism(
            fill.spacetime, [], [], [1])
        self.assertTrue(legacy.einstein_hilbert_enabled)
        self.assertFalse(legacy.real_squared_lengths_only)

        with self.assertRaises(ValueError):
            fill.node.declare_register_constraint(
                "", 1, fill.holes, [0j] * len(fill.holes))
        with self.assertRaises(ValueError):
            fill.node.declare_register_constraint(
                "bad", 1, fill.holes, [0j])
        with self.assertRaises(ValueError):
            fill.node.declare_register_constraint(
                "bad", 1, [[0, 1]], [0j])


class HistoricalFixedBoundarySpectralTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = GO.historical_spectral_experiment(
            epsilon=1e-10,
            restarts=1,
            max_growth=4,
            seed=0,
            max_iterations=200,
        )

    def test_rayleigh_residual_converges_with_boundary_fixed(self):
        phase = self.result["cases"]["charge_preserving_phase"]
        self.assertTrue(phase["converged"])
        self.assertLess(phase["residual"], 1e-10)
        self.assertAlmostEqual(
            phase["residual"], phase["residual_cross_check"], places=20)
        self.assertLess(phase["eigenvector_defect"], 1e-5)
        self.assertGreater(phase["growth_steps"], 0)
        self.assertTrue(phase["boundary_preserved"])
        self.assertEqual(phase["boundary_drift"], 0.0)

    def test_pinned_choi_ray_promotes_and_applies_to_held_out_states(self):
        for case in self.result["cases"].values():
            self.assertGreater(case["support_choi_overlap"], 1.0 - 1e-12)
            self.assertLess(case["operator_error"], 1e-12)
            self.assertLess(case["held_out_error_max"], 1e-12)

    def test_charge_conservation_is_not_part_of_historical_solver(self):
        phase = self.result["cases"]["charge_preserving_phase"]
        changing = self.result["cases"]["charge_changing_x"]
        self.assertLess(phase["charge_commutator_error"], 1e-12)
        self.assertGreater(changing["charge_commutator_error"], 1.0)
        self.assertLess(changing["residual"], 1e-10)

    def test_support_cells_are_explicit_and_validated(self):
        fixture = GO.SpectralChoiCobordism()
        with self.assertRaises(ValueError):
            fixture.node.relax_fixed_boundary_eigenstate(
                0, [[99]], [1.0 + 0.0j], restarts=1, max_growth=0)
        with self.assertRaises(ValueError):
            fixture.node.relax_fixed_boundary_eigenstate(
                0, [[0], [0]], [1.0 + 0.0j, 0.0j],
                restarts=1, max_growth=0)


class ResidualOnlyRelaxationTest(unittest.TestCase):
    def test_objective_is_only_r_u(self):
        fill = GO.PeriodCobordism()
        fill.pin_basis(GO.generic_charge_preserving_operator())
        snapshot = fill.snapshot()
        self.assertEqual(snapshot["regge_stationarity"], 0.0)
        self.assertEqual(snapshot["objective"], snapshot["r_u"])
        self.assertEqual(
            snapshot["register_residual_term"], snapshot["r_u"])

    def test_full_boundary_is_fixed_while_bulk_descends(self):
        fill = GO.PeriodCobordism()
        fill.pin_basis(GO.generic_charge_preserving_operator())
        boundary_before = fill.boundary_snapshot()
        free_before = fill.free_squared_lengths().copy()
        gap_before = fill.hard_period_gap()
        result = fill.relax(iterations=1, alpha=0.05)
        free_after = fill.free_squared_lengths()

        self.assertEqual(result["accepted_steps"], 1)
        self.assertLess(
            result["final"]["r_u"], result["initial"]["r_u"])
        self.assertEqual(fill.boundary_snapshot(), boundary_before)
        self.assertEqual(fill.boundary_drift(), 0.0)
        self.assertGreater(
            float(np.max(np.abs(free_after - free_before))), 1e-8)
        self.assertEqual(
            float(np.max(np.abs(free_after.imag), initial=0.0)), 0.0)
        self.assertAlmostEqual(
            fill.hard_period_gap(), gap_before, places=10)


class BoundaryTransportIdentifiabilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.identity = GO.PeriodCobordism()
        cls.identity.pin_basis(GO.identity_operator())
        cls.cycle = GO.PeriodCobordism(twist=GO._GAMMA)
        cls.cycle.pin_basis(GO.cycle_operator())

    def test_one_pair_has_tiny_residual_but_not_an_operator(self):
        reflection = GO.lift_sector_operator(
            np.diag([1.0, -1.0]))
        fill = GO.PeriodCobordism()
        selected = np.array([1.0, 0.0], dtype=complex)
        fill.pin_state("selected", reflection, selected)
        self.assertLess(
            fill.snapshot()["r_u"], GO._TINY_PERIOD_RESIDUAL)

        transport = fill.read_transport()
        unseen = np.array([0.0, 1.0], dtype=complex)
        self.assertGreater(
            np.linalg.norm(transport.apply(unseen) + unseen), 1.0)
        readout = fill.node.geometric_operator(2)
        self.assertFalse(readout.identifiable)
        self.assertIn("no interior 1-cells", readout.obstruction)

    def test_complete_identity_and_mapping_class_generalize(self):
        identity = self.identity.read_transport().transport
        cycle = self.cycle.read_transport().transport
        expected_cycle = GO.logical_operator(GO.cycle_operator())
        np.testing.assert_allclose(identity, np.eye(2), atol=1e-12)
        np.testing.assert_allclose(
            cycle, expected_cycle, atol=1e-12)

        state = GO._unit(np.array([0.31 + 0.2j, -0.7 + 0.1j]))
        np.testing.assert_allclose(
            self.identity.read_transport().apply(state),
            state,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            self.cycle.read_transport().apply(state),
            expected_cycle @ state,
            atol=1e-12,
        )

    def test_charge_conservation_is_not_sufficient(self):
        target = GO.generic_charge_preserving_operator()
        fill = GO.PeriodCobordism()
        fill.pin_basis(target)
        self.assertLess(GO.charge_commutator_error(target), 1e-12)
        self.assertGreater(fill.hard_period_gap(), 1e-2)
        self.assertGreater(
            np.linalg.norm(
                fill.read_transport().transport
                - GO.logical_operator(target)),
            1e-2,
        )

    def test_charge_leak_is_detected(self):
        target = GO.charge_leaking_operator()
        np.testing.assert_allclose(
            target.conj().T @ target, np.eye(3), atol=1e-12)
        self.assertGreater(
            GO.charge_commutator_error(target), 1e-2)
        fill = GO.PeriodCobordism()
        fill.pin_basis(target)
        self.assertGreater(fill.hard_period_gap(), 1e-2)


class BulkChoiPromotionTest(unittest.TestCase):
    @staticmethod
    def square_cycle():
        base = [[0, 1], [1, 2], [2, 3], [0, 3]]
        spacetime = tessera.Spacetime.fromCells(
            2, tessera.Spacetime.prismCells(base, 2, {}),
            1.0, 0.0)
        spacetime.materializeFacets()
        node = tessera.cobordism.MultiCobordism(
            spacetime, [], [], [1],
            einstein_hilbert=False,
            real_squared_lengths_only=True,
        )
        return spacetime, node

    def test_rank_one_frame_promotes_with_locked_choi_convention(self):
        result = GO.square_cycle_choi_control()
        self.assertTrue(result["identifiable"])
        self.assertEqual(result["kernel_dimension"], 1)
        self.assertEqual(result["frame_rank"], 1)
        self.assertLess(result["unitarity_error"], 1e-12)
        self.assertLess(result["choi_convention_error"], 1e-12)
        self.assertLess(result["choi_duality_error"], 1e-12)
        self.assertLess(result["charge_commutator_error"], 1e-12)
        self.assertLess(result["metric_perturbation_error"], 1e-12)
        self.assertGreater(result["frame_permutation_error"], 1.0)

    def test_metric_kernel_and_combinatorial_default_are_available(self):
        spacetime, node = self.square_cycle()
        synthesis = tessera.cobordism.EigenstateSynthesis(
            spacetime, 1)
        cells = synthesis.bulkMinusBoundaryCells()
        combinatorial = synthesis.bulkMinusBoundaryHarmonicMatrix()
        explicit_combinatorial = (
            synthesis.bulkMinusBoundaryHarmonicMatrix(metric=False))
        np.testing.assert_array_equal(combinatorial, explicit_combinatorial)
        metric = synthesis.bulkMinusBoundaryHarmonicMatrix(
            metric=True)
        self.assertEqual(len(cells), 4)
        self.assertEqual(len(combinatorial), 4)
        self.assertEqual(len(metric), 4)

        missing_frame = node.geometric_operator(3)
        self.assertFalse(missing_frame.identifiable)
        self.assertIn("ordered d^2 Choi frame", missing_frame.obstruction)
        repeated = node.geometric_operator(2, [cells[0]] * 4)
        self.assertFalse(repeated.identifiable)
        self.assertIn("repeats", repeated.obstruction)

    def test_multidimensional_frame_is_an_obstruction(self):
        two_cycles = [
            [0, 1], [1, 2], [0, 2],
            [3, 4], [4, 5], [3, 5],
        ]
        spacetime = tessera.Spacetime.fromCells(
            2,
            tessera.Spacetime.prismCells(two_cycles, 2, {}),
            1.0,
            0.0,
        )
        spacetime.materializeFacets()
        node = tessera.cobordism.MultiCobordism(
            spacetime, [], [], [1],
            einstein_hilbert=False,
            real_squared_lengths_only=True,
        )
        synthesis = tessera.cobordism.EigenstateSynthesis(
            spacetime, 1)
        readout = node.geometric_operator(
            2, synthesis.bulkMinusBoundaryCells()[:4])
        self.assertFalse(readout.identifiable)
        self.assertEqual(readout.kernel_dimension, 2)
        self.assertEqual(readout.frame_rank, 2)
        self.assertIn("no unique Choi ray", readout.obstruction)


class CommandLineTest(unittest.TestCase):
    def test_live_flag_is_exposed(self):
        args = GO.build_parser().parse_args(
            ["--live", "--no-write"])
        self.assertTrue(args.live)
        self.assertTrue(args.no_write)


if __name__ == "__main__":
    unittest.main()
