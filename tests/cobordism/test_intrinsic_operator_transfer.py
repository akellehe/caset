# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Intrinsic operator recovery from a frozen relaxed cobordism (#899).

The experiment is valid only if the target disappears before readout. These
tests therefore pin four independent properties:

* the two boundary restrictions of the frozen harmonic space reconstruct the
  identity and the known mapping-class cycle without a target argument;
* a complete charge-sector basis catches an operator that a selected input
  cannot distinguish from the identity;
* a generic charge-preserving unitary can remain obstructed on the fixed
  topology, while a charge-leaking unitary is rejected by the boundary charge
  constraint;
* the exact gradient descends the historical stationarity-plus-register
  objective without moving the boundary, but a lower scale-dependent residual
  is not confused with realization when the hard period gap and intrinsic
  operator remain unchanged.
"""

import importlib.util
import os
import sys
import unittest

import numpy as np

import tessera


_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.join(
    _HERE, "..", "..", "examples", "cobordism",
    "intrinsic_operator_transfer.py",
)
_SPEC = importlib.util.spec_from_file_location(
    "intrinsic_operator_transfer", _EXAMPLE)
IOT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = IOT
_SPEC.loader.exec_module(IOT)

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


class FrozenOperatorReadoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.identity_fill = IOT.PeriodCobordism()
        cls.identity = cls.identity_fill.read_operator()
        cls.cycle_fill = IOT.PeriodCobordism(twist=IOT._GAMMA)
        cls.cycle = cls.cycle_fill.read_operator()

    def test_identity_is_reconstructed_without_a_target(self):
        np.testing.assert_allclose(
            self.identity.transport, np.eye(2), atol=1e-12)
        self.assertFalse(hasattr(self.identity, "target"))
        state = IOT._unit(np.array([0.31 + 0.2j, -0.7 + 0.1j]))
        np.testing.assert_allclose(
            self.identity.apply(state), state, atol=1e-12)

    def test_mapping_class_cycle_is_reconstructed(self):
        expected = (
            IOT._SECTOR_BASIS @ IOT.cycle_operator()
            @ IOT._SECTOR_BASIS.conj().T
        )
        np.testing.assert_allclose(
            self.cycle.transport, expected, atol=1e-12)
        state = IOT._unit(np.array([0.2 - 0.4j, 0.7 + 0.1j]))
        np.testing.assert_allclose(
            self.cycle.apply(state), expected @ state, atol=1e-12)

    def test_amplitude_operator_is_the_derived_gram_corrected_map(self):
        expected = np.linalg.solve(
            self.identity.transport.conj().T, self.identity.gram)
        np.testing.assert_allclose(
            self.identity.amplitude_operator, expected, atol=1e-13)
        source = IOT._unit(np.array([1.0, 0.2j]))
        effect = IOT._unit(np.array([0.3, -0.4j]))
        self.assertAlmostEqual(
            self.identity.amplitude(effect, source),
            np.vdot(effect, expected @ source),
            places=13,
        )

    def test_gram_uses_the_live_signed_hodge_pairing(self):
        forms = []
        for basis_state in np.eye(2, dtype=complex):
            coefficients = np.linalg.solve(
                self.identity.input_restriction.T, basis_state)
            forms.append(coefficients @ self.identity.harmonics)
        expected = np.array([
            [np.vdot(left, self.identity.weights * right)
             for right in forms]
            for left in forms
        ])
        euclidean = np.array([
            [np.vdot(left, right) for right in forms]
            for left in forms
        ])
        np.testing.assert_allclose(
            self.identity.raw_gram, expected, atol=1e-13)
        self.assertGreater(np.linalg.norm(expected - euclidean), 1e-2)

    def test_causal_staircase_chart_is_not_isometric(self):
        # Transport succeeds, but H3 additionally needs G=I. The causal
        # staircase deliberately exposes that independent unmet condition.
        self.assertLess(
            np.linalg.norm(self.identity.transport - np.eye(2)), 1e-12)
        self.assertGreater(
            np.linalg.norm(self.identity.gram - np.eye(2)), 1e-2)
        self.assertGreater(
            np.linalg.norm(
                self.identity.trace_normalized_gram - np.eye(2)),
            1e-2,
        )
        self.assertGreater(
            np.linalg.norm(self.identity.amplitude_operator - np.eye(2)),
            1e-2,
        )


class CompleteBasisPinTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fill = IOT.PeriodCobordism()

    def test_one_selected_input_does_not_identify_an_operator(self):
        reflection = IOT.lift_sector_operator(np.diag([1.0, -1.0]))
        selected = self.fill.state_residual(
            reflection, np.array([1.0, 0.0]), hard=True)
        complete = self.fill.operator_residual(reflection, hard=True)
        self.assertLess(selected, 1e-12)
        self.assertGreater(complete, 1e-2)

    def test_charge_conservation_is_not_sufficient_on_fixed_topology(self):
        target = IOT.generic_charge_preserving_operator()
        self.assertLess(IOT.charge_commutator_error(target), 1e-12)
        self.assertLess(IOT.sector_charge_leak(target), 1e-12)
        self.assertGreater(
            self.fill.operator_residual(target, hard=True), 1e-2)
        self.assertGreater(
            np.linalg.norm(
                self.fill.read_operator().transport
                - IOT._SECTOR_BASIS @ target
                @ IOT._SECTOR_BASIS.conj().T),
            1e-2,
        )

    def test_charge_leaking_unitary_is_rejected(self):
        target = IOT.charge_leaking_operator()
        np.testing.assert_allclose(
            target.conj().T @ target, np.eye(3), atol=1e-12)
        self.assertGreater(IOT.charge_commutator_error(target), 1e-2)
        self.assertGreater(IOT.sector_charge_leak(target), 1e-2)
        self.assertGreater(
            self.fill.operator_residual(target, hard=True), 1e-2)


class FixedBoundaryRelaxationTest(unittest.TestCase):
    def test_complete_objective_gradient_matches_finite_difference(self):
        fill = IOT.PeriodCobordism()
        relaxer = IOT.FixedBoundaryRelaxation(
            fill, IOT.generic_charge_preserving_operator(), gamma=1.0)
        self.assertLess(
            relaxer.finite_difference_error(indices=(0, 1)), 1e-4)

    def test_relaxation_descends_without_moving_boundary_or_operator(self):
        fill = IOT.PeriodCobordism()
        target = IOT.generic_charge_preserving_operator()
        transfer_before = fill.read_operator().transport.copy()
        relaxer = IOT.FixedBoundaryRelaxation(fill, target, gamma=1.0)
        result = relaxer.relax(max_iterations=2)
        transfer_after = fill.read_operator().transport

        self.assertGreater(result["accepted_steps"], 0)
        self.assertLess(
            result["final"]["total"], result["initial"]["total"])
        self.assertLess(
            result["final"]["register_residual"],
            result["initial"]["register_residual"],
        )
        self.assertAlmostEqual(
            result["final"]["hard_period_gap"],
            result["initial"]["hard_period_gap"],
            places=10,
        )
        np.testing.assert_allclose(
            transfer_after, transfer_before, atol=1e-11)
        self.assertEqual(result["boundary_drift"], 0.0)
        self.assertTrue(result["admissible"])
        self.assertFalse(result["stationary"])
        self.assertFalse(result["operator_realized"])
        self.assertFalse(result["converged"])

    def test_new_inputs_generalize_after_positive_relaxation(self):
        fill = IOT.PeriodCobordism()
        target = IOT.identity_operator()
        relaxation = IOT.FixedBoundaryRelaxation(
            fill, target, gamma=1.0)
        relaxation.relax(max_iterations=2)
        metrics = IOT.operator_metrics(fill, target, held_out=20)
        self.assertLess(metrics["transport_operator_error"], 1e-10)
        self.assertLess(metrics["held_out_transport_error_max"], 1e-10)
        self.assertEqual(metrics["boundary_drift"], 0.0)


if __name__ == "__main__":
    unittest.main()
