# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Intrinsic operator recovery from a relaxed simplicial cobordism.

This experiment closes the central loophole in the retired gate-realizability
experiments: a target output is used only while fitting the geometry.  After
relaxation the target is discarded and the operator is reconstructed from the
two boundary restrictions of the live harmonic space.

The boundary register is the charge-zero plane

    V = ker(1, 1, 1)  subset C^3.

For a graph-like fill W, let A and B be the restrictions of ker L_1(W) to an
orthonormal basis of V on the input and output boundaries.  When both
restrictions are full rank, the geometry determines

    T_W = B^T (A^T)^-1.

The complete-basis pin for a candidate U is the sum of the existing
``residualForPeriods`` values for (e_j, U e_j), not a test on one selected
state.  Relaxation minimizes

    Phi = ||dS_Regge / d l^2_free||^2 + Gamma r_U

over only the timelike interior squared lengths; all boundary lengths remain
bit-identical.  The hard period gap is reported separately because it has the
same exact zero set as r_U but cannot be lowered by rescaling a leaked state.

After freezing W, unseen inputs are attached with ``FrozenPeriodOperator.apply``.
That object has no target-operator field.  It also reconstructs the
anchor-normalized input Gram G and the amplitude operator

    G_ij = s1 h_i^dagger W_1 h_j,    A_W = (T_W^dagger)^-1 G,

so transport, chart isometry, unitarity, and the H3 amplitude claim are measured
independently.

Run:

    python examples/cobordism/intrinsic_operator_transfer.py
    python examples/cobordism/intrinsic_operator_transfer.py --max-iterations 20
"""

from __future__ import annotations

import argparse
import cmath
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import tessera


cob = tessera.cobordism

_TOL = 1e-9
_CHARGE = np.ones(3, dtype=complex) / math.sqrt(3.0)
_CHARGE_PROJECTOR = np.outer(_CHARGE, _CHARGE.conj())
_SECTOR_BASIS = np.array(
    [
        [1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0), 0.0],
        [1.0 / math.sqrt(6.0), 1.0 / math.sqrt(6.0),
         -2.0 / math.sqrt(6.0)],
    ],
    dtype=complex,
)

# Standard icosahedron with three mutually vertex-disjoint faces removed.  The
# resulting pair-of-pants surface has b_1 = 2 and carries the two-dimensional
# charge-zero register.
_ICOSAHEDRON_FACES = [
    [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 1],
    [1, 5, 10], [1, 10, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3],
    [3, 7, 8], [3, 8, 4], [4, 8, 9], [4, 9, 5], [5, 9, 10],
    [6, 10, 11], [7, 6, 11], [8, 7, 11], [9, 8, 11], [10, 9, 11],
]
_WINDOWS = [[0, 1, 2], [3, 7, 8], [4, 9, 5]]
_WINDOW_SET = {tuple(sorted(window)) for window in _WINDOWS}
_REGISTER_FACES = [
    face for face in _ICOSAHEDRON_FACES
    if tuple(sorted(face)) not in _WINDOW_SET
]

# The order-three automorphism of the holed icosahedron used by the historical
# mapping-class positive control.
_GAMMA = {
    0: 3, 1: 7, 2: 8, 3: 4, 4: 0, 5: 2,
    6: 11, 7: 9, 8: 5, 9: 1, 10: 6, 11: 10,
}


def _edge_key(edge):
    a = int(edge.getSource().getId())
    b = int(edge.getTarget().getId())
    return min(a, b), max(a, b)


def _matrix_payload(matrix):
    """JSON-safe complex matrix as ``[[[real, imag], ...], ...]``."""
    array = np.asarray(matrix, dtype=complex)
    return [[[float(value.real), float(value.imag)] for value in row]
            for row in array]


def _unit(vector):
    vector = np.asarray(vector, dtype=complex)
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        raise ValueError("zero vector has no normalized state")
    return vector / norm


def lift_sector_operator(logical):
    """Lift a 2x2 operator on V to C^3 while fixing the charge direction."""
    logical = np.asarray(logical, dtype=complex)
    if logical.shape != (2, 2):
        raise ValueError("logical operator must be 2x2")
    return _CHARGE_PROJECTOR + _SECTOR_BASIS.conj().T @ logical @ _SECTOR_BASIS


def identity_operator():
    return np.eye(3, dtype=complex)


def cycle_operator():
    """The known raw-period three-cycle induced by the gamma twist."""
    return np.array([[0.0, 0.0, 1.0],
                     [1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0]], dtype=complex)


def generic_charge_preserving_operator(theta=0.37, phase=0.41):
    """A continuous SU(2) rotation on V, outside the prism mapping classes."""
    c = math.cos(theta)
    s = math.sin(theta)
    logical = np.array(
        [[c, -cmath.exp(-1j * phase) * s],
         [cmath.exp(1j * phase) * s, c]],
        dtype=complex,
    )
    return lift_sector_operator(logical)


def charge_leaking_operator(theta=0.43):
    """A unitary negative control that rotates V into the charge direction."""
    basis = np.column_stack((_CHARGE, _SECTOR_BASIS.conj().T))
    c = math.cos(theta)
    s = math.sin(theta)
    in_basis = np.array([[c, -s, 0.0],
                         [s, c, 0.0],
                         [0.0, 0.0, 1.0]], dtype=complex)
    return basis @ in_basis @ basis.conj().T


def charge_commutator_error(operator):
    operator = np.asarray(operator, dtype=complex)
    return float(np.linalg.norm(
        operator @ _CHARGE_PROJECTOR - _CHARGE_PROJECTOR @ operator))


def sector_charge_leak(operator):
    """Charge amplitude created from any input in V (Frobenius norm)."""
    operator = np.asarray(operator, dtype=complex)
    return float(np.linalg.norm(
        _CHARGE.conj()[None, :] @ operator @ _SECTOR_BASIS.conj().T))


class FrozenPeriodOperator:
    """Target-independent operator read from one frozen live geometry."""

    def __init__(self, periods, harmonics, weights, sign_in, sign_out,
                 tol=_TOL):
        self.periods = np.asarray(periods, dtype=complex)
        self.harmonics = np.asarray(harmonics, dtype=complex)
        self.weights = np.asarray(weights, dtype=complex)
        self.sign_in = np.asarray(sign_in, dtype=complex)
        self.sign_out = np.asarray(sign_out, dtype=complex)
        if self.periods.ndim != 2 or self.periods.shape[1] != 6:
            raise ValueError("period matrix must have six boundary columns")
        if self.harmonics.shape[0] != self.periods.shape[0]:
            raise ValueError("period and harmonic row counts differ")
        if self.weights.shape != (self.harmonics.shape[1],):
            raise ValueError("Hodge weights do not match harmonic width")
        if not np.all(np.isfinite(self.weights)):
            raise ValueError("Hodge weights must be finite")
        if np.max(np.abs(self.weights.imag), initial=0.0) > tol:
            raise ValueError(
                "the experiment requires real signed Hodge weights")
        if self.periods.shape[0] != 2:
            raise ValueError(
                "a unique two-dimensional operator requires dim ker L_1 = 2")

        raw_in = self.periods[:, :3]
        raw_out = self.periods[:, 3:]
        self.input_restriction = (
            raw_in * self.sign_in[None, :]) @ _SECTOR_BASIS.conj().T
        self.output_restriction = (
            raw_out * self.sign_out[None, :]) @ _SECTOR_BASIS.conj().T
        self.input_rank = int(np.linalg.matrix_rank(
            self.input_restriction, tol=tol))
        self.output_rank = int(np.linalg.matrix_rank(
            self.output_restriction, tol=tol))
        if self.input_rank != 2 or self.output_rank != 2:
            raise ValueError("a boundary restriction is rank-deficient")

        # If c are harmonic coefficients, x=A^T c and y=B^T c.  Eliminating c
        # gives y=B^T(A^T)^-1 x.
        self.transport = np.linalg.solve(
            self.input_restriction, self.output_restriction).T

        input_forms = []
        for basis_state in np.eye(2, dtype=complex):
            coefficients = np.linalg.solve(
                self.input_restriction.T, basis_state)
            input_forms.append(coefficients @ self.harmonics)
        def pair(left, right):
            return np.vdot(left, self.weights * right)
        anchor_norm = complex(pair(input_forms[0], input_forms[0]))
        if abs(anchor_norm.imag) > tol or anchor_norm.real <= tol:
            raise ValueError("input anchor has zero norm")
        self.anchor_scale = 1.0 / anchor_norm.real
        self.raw_gram = np.array(
            [[pair(left, right) for right in input_forms]
             for left in input_forms],
            dtype=complex,
        )
        self.gram = self.anchor_scale * self.raw_gram
        trace = complex(np.trace(self.raw_gram))
        if abs(trace.imag) > tol or trace.real <= tol:
            raise ValueError("input Gram has non-positive trace")
        self.trace_normalized_gram = 2.0 * self.raw_gram / trace.real
        self.amplitude_operator = np.linalg.solve(
            self.transport.conj().T, self.gram)

    def apply(self, logical_input):
        """Attach a new V-state after fitting and return the intrinsic output."""
        logical_input = np.asarray(logical_input, dtype=complex)
        if logical_input.shape != (2,):
            raise ValueError("logical input must have two components")
        return self.transport @ logical_input

    def amplitude(self, logical_output, logical_input):
        logical_output = np.asarray(logical_output, dtype=complex)
        return complex(np.vdot(
            logical_output, self.amplitude_operator @ logical_input))

    def diagnostics(self):
        identity = np.eye(2, dtype=complex)
        return {
            "transport": _matrix_payload(self.transport),
            "gram": _matrix_payload(self.gram),
            "amplitude_operator": _matrix_payload(self.amplitude_operator),
            "input_rank": self.input_rank,
            "output_rank": self.output_rank,
            "gram_error": float(np.linalg.norm(self.gram - identity)),
            "gram_shape_error": float(np.linalg.norm(
                self.trace_normalized_gram - identity)),
            "gram_condition_number": float(np.linalg.cond(self.raw_gram)),
            "transport_unitarity_error": float(np.linalg.norm(
                self.transport.conj().T @ self.transport - identity)),
            "amplitude_unitarity_error": float(np.linalg.norm(
                self.amplitude_operator.conj().T
                @ self.amplitude_operator - identity)),
        }


class PeriodCobordism:
    """Causal prism between two three-holed charge-register boundaries."""

    def __init__(self, twist=None):
        cells = tessera.Spacetime.prismCells(
            _REGISTER_FACES, 1, twist or {})
        self.spacetime = tessera.Spacetime.fromCells(3, cells, 1.0, 0.0)

        # The two layers use ids 0..11 and 12..23.  A cross-layer edge is
        # timelike; every free interior edge of this prism is cross-layer.
        for edge in self.spacetime.getEdgeList().toVector():
            a, b = _edge_key(edge)
            cross_layer = (a < 12) != (b < 12)
            edge.setLength(1j if cross_layer else 1.0)
            edge.setPhase(0.0)
        self.spacetime.materializeFacets()

        self.synthesis = cob.EigenstateSynthesis(self.spacetime, 1)
        self.holes_in = [list(window) for window in _WINDOWS]
        self.holes_out = [[vertex + 12 for vertex in window]
                          for window in _WINDOWS]
        self.holes = self.holes_in + self.holes_out
        faces_in = [list(face) for face in _REGISTER_FACES]
        faces_out = [[vertex + 12 for vertex in face]
                     for face in _REGISTER_FACES]
        self.sign_in = np.asarray(
            cob.ChainComplex.endSignCovector(faces_in, self.holes_in),
            dtype=complex,
        )
        self.sign_out = np.asarray(
            cob.ChainComplex.endSignCovector(faces_out, self.holes_out),
            dtype=complex,
        )
        valid, reason = self.synthesis.dualComplexValid()
        if not valid:
            raise RuntimeError(f"invalid dual complex: {reason}")

        self.edge_map = {
            _edge_key(edge): edge
            for edge in self.spacetime.getEdgeList().toVector()
        }
        self.boundary_edges = [tuple(map(int, edge))
                               for edge in self.synthesis.boundaryEdges()]
        self.interior_edges = [tuple(map(int, edge))
                               for edge in self.synthesis.interiorEdges()]
        self.boundary_initial = self.boundary_snapshot()

    def boundary_snapshot(self):
        return {
            edge: complex(self.edge_map[edge].getLength()) ** 2
            for edge in self.boundary_edges
        }

    def boundary_drift(self):
        current = self.boundary_snapshot()
        return float(max(
            (abs(current[edge] - value)
             for edge, value in self.boundary_initial.items()),
            default=0.0,
        ))

    def set_interior_weights(self, squared_lengths):
        values = np.asarray(squared_lengths, dtype=float)
        if values.shape != (len(self.interior_edges),):
            raise ValueError("wrong number of interior squared lengths")
        self.synthesis.setInteriorWeights(values.tolist())
        self.spacetime.materializeFacets()

    def interior_weights(self):
        return np.asarray(self.synthesis.interiorWeights(), dtype=complex).real

    def geometry_admissible(self, tol=1e-8):
        """Every tetrahedron has one timelike and two spacelike Gram modes."""
        for simplex in self.spacetime.getTopSimplices():
            gram = np.asarray(simplex.gramMatrix(), dtype=complex).reshape(3, 3)
            if not np.all(np.isfinite(gram)) or np.max(np.abs(gram.imag)) > tol:
                return False
            eigenvalues = np.linalg.eigvalsh(gram.real)
            if np.count_nonzero(eigenvalues < -tol) != 1:
                return False
            if np.count_nonzero(eigenvalues > tol) != 2:
                return False
        return True

    def _raw_pair(self, raw_input, raw_output):
        raw_input = np.asarray(raw_input, dtype=complex)
        raw_output = np.asarray(raw_output, dtype=complex)
        return np.concatenate((self.sign_in * raw_input,
                               self.sign_out * raw_output))

    def _basis_pairs(self, operator):
        operator = np.asarray(operator, dtype=complex)
        if operator.shape != (3, 3):
            raise ValueError("operator must be 3x3 on the raw period space")
        for logical_input in np.eye(2, dtype=complex):
            raw_input = _SECTOR_BASIS.conj().T @ logical_input
            yield self._raw_pair(raw_input, operator @ raw_input)

    def state_residual(self, operator, logical_input, hard=False):
        logical_input = np.asarray(logical_input, dtype=complex)
        raw_input = _SECTOR_BASIS.conj().T @ logical_input
        pair = self._raw_pair(raw_input, operator @ raw_input)
        method = (self.synthesis.periodGapForPeriods if hard
                  else self.synthesis.residualForPeriods)
        return float(method(self.holes, [complex(value) for value in pair]))

    def operator_residual(self, operator, hard=False):
        method = (self.synthesis.periodGapForPeriods if hard
                  else self.synthesis.residualForPeriods)
        return float(sum(
            method(self.holes, [complex(value) for value in pair])
            for pair in self._basis_pairs(operator)
        ))

    def operator_residual_gradient(self, operator):
        gradients = [
            np.asarray(self.synthesis.residualForPeriodsGradient(
                self.holes, [complex(value) for value in pair]), dtype=complex)
            for pair in self._basis_pairs(operator)
        ]
        return np.sum(gradients, axis=0)

    def read_operator(self):
        flat_periods = np.asarray(
            self.synthesis.cyclePeriods(self.holes), dtype=complex)
        periods = flat_periods.reshape(-1, len(self.holes))
        cells = self.synthesis.cellSimplices()
        hodge = cob.HodgeLaplacian(self.spacetime)
        flat_harmonics = np.asarray(
            hodge.harmonicMatrix(1, _TOL, True),
            dtype=complex,
        )
        harmonics = flat_harmonics.reshape(-1, len(cells))
        weights = np.asarray(hodge.weights(1), dtype=complex)
        return FrozenPeriodOperator(
            periods, harmonics, weights, self.sign_in, self.sign_out)


@dataclass(frozen=True)
class ObjectiveValue:
    total: float
    stationarity: float
    stationarity_norm: float
    register_residual: float
    hard_period_gap: float
    action: complex

    def payload(self):
        return {
            "total": self.total,
            "stationarity": self.stationarity,
            "stationarity_norm": self.stationarity_norm,
            "register_residual": self.register_residual,
            "hard_period_gap": self.hard_period_gap,
            "action": [float(self.action.real), float(self.action.imag)],
        }


class FixedBoundaryRelaxation:
    """Exact-gradient relaxation of Phi on the free timelike edges."""

    def __init__(self, fill, target, gamma=1.0):
        self.fill = fill
        self.target = np.asarray(target, dtype=complex)
        self.gamma = float(gamma)
        if self.gamma < 0.0:
            raise ValueError("gamma must be non-negative")
        self.free_edges = list(fill.interior_edges)
        if not self.free_edges:
            raise ValueError("fill has no free interior edges")

        action_edges = [
            _edge_key(edge)
            for edge in fill.spacetime.getEdgeList().toVector()
        ]
        action_index = {edge: index for index, edge in enumerate(action_edges)}
        period_edges = [tuple(map(int, cell))
                        for cell in fill.synthesis.cellSimplices()]
        period_index = {edge: index for index, edge in enumerate(period_edges)}
        self.action_indices = np.array(
            [action_index[edge] for edge in self.free_edges], dtype=int)
        self.period_indices = np.array(
            [period_index[edge] for edge in self.free_edges], dtype=int)

        self.initial_weights = fill.interior_weights()
        if not np.all(self.initial_weights < 0.0):
            raise ValueError("the experiment expects free timelike edges")
        magnitudes = np.abs(self.initial_weights)
        self.lower = -2.5 * magnitudes
        self.upper = -0.25 * magnitudes

    def _action_state(self, with_hessian=False):
        solver = tessera.ReggeSolver(
            self.fill.spacetime, tessera.MatterConfiguration())
        gradient = np.asarray(solver.actionGradientExact(), dtype=complex)
        hessian = None
        if with_hessian:
            hessian = np.asarray(solver.actionHessianExact(), dtype=complex)
        return solver, gradient, hessian

    def value(self):
        solver, action_gradient, _ = self._action_state(False)
        free_gradient = action_gradient[self.action_indices]
        stationarity = float(np.vdot(free_gradient, free_gradient).real)
        register_residual = self.fill.operator_residual(self.target)
        hard_gap = self.fill.operator_residual(self.target, hard=True)
        return ObjectiveValue(
            total=stationarity + self.gamma * register_residual,
            stationarity=stationarity,
            stationarity_norm=float(np.linalg.norm(free_gradient)),
            register_residual=register_residual,
            hard_period_gap=hard_gap,
            action=complex(solver.dualReggeAction()),
        )

    def value_and_gradient(self):
        solver, action_gradient, action_hessian = self._action_state(True)
        free_gradient = action_gradient[self.action_indices]
        free_hessian = action_hessian[np.ix_(
            self.action_indices, self.action_indices)]
        stationarity_gradient = 2.0 * np.real(
            free_hessian.conj().T @ free_gradient)

        residual_gradient = self.fill.operator_residual_gradient(self.target)
        register_gradient = residual_gradient[self.period_indices].real
        register_residual = self.fill.operator_residual(self.target)
        hard_gap = self.fill.operator_residual(self.target, hard=True)
        stationarity = float(np.vdot(free_gradient, free_gradient).real)
        value = ObjectiveValue(
            total=stationarity + self.gamma * register_residual,
            stationarity=stationarity,
            stationarity_norm=float(np.linalg.norm(free_gradient)),
            register_residual=register_residual,
            hard_period_gap=hard_gap,
            action=complex(solver.dualReggeAction()),
        )
        gradient = stationarity_gradient + self.gamma * register_gradient
        return value, np.asarray(gradient, dtype=float)

    def finite_difference_error(self, indices=(0, 1), step=1e-6):
        """Cross-check the complete Phi gradient on selected free variables."""
        base = self.fill.interior_weights()
        _value, analytic = self.value_and_gradient()
        worst = 0.0
        for index in indices:
            if index >= len(base):
                continue
            up = base.copy()
            down = base.copy()
            up[index] += step
            down[index] -= step
            self.fill.set_interior_weights(up)
            value_up = self.value().total
            self.fill.set_interior_weights(down)
            value_down = self.value().total
            finite_difference = (value_up - value_down) / (2.0 * step)
            worst = max(worst, abs(finite_difference - analytic[index]))
        self.fill.set_interior_weights(base)
        return float(worst)

    def relax(self, max_iterations=12, initial_step=0.05,
              gradient_tolerance=1e-8, operator_tolerance=1e-9):
        """Backtracking descent with bounds and Lorentzian signature gating."""
        weights = self.fill.interior_weights()
        boundary_before = self.fill.boundary_snapshot()
        initial = self.value()
        trace = [initial.payload()]
        termination = "iteration_budget"
        accepted_steps = 0

        for _iteration in range(int(max_iterations)):
            current, gradient = self.value_and_gradient()
            if not np.all(np.isfinite(gradient)):
                termination = "nonfinite_gradient"
                break
            if (current.stationarity_norm <= gradient_tolerance
                    and current.hard_period_gap <= operator_tolerance):
                termination = "converged"
                break
            gradient_scale = float(np.max(np.abs(gradient)))
            if gradient_scale <= gradient_tolerance:
                termination = "objective_stationary"
                break

            direction = -gradient / gradient_scale
            directional_derivative = float(np.dot(gradient, direction))
            accepted = False
            step = float(initial_step)
            for _backtrack in range(18):
                candidate = np.clip(
                    weights + step * direction, self.lower, self.upper)
                if np.array_equal(candidate, weights):
                    step *= 0.5
                    continue
                self.fill.set_interior_weights(candidate)
                if not self.fill.geometry_admissible():
                    step *= 0.5
                    continue
                candidate_value = self.value()
                armijo = current.total + 1e-4 * step * directional_derivative
                if (math.isfinite(candidate_value.total)
                        and candidate_value.total <= armijo):
                    weights = candidate
                    trace.append(candidate_value.payload())
                    accepted_steps += 1
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                self.fill.set_interior_weights(weights)
                termination = "line_search_stalled"
                break

        self.fill.set_interior_weights(weights)
        final = self.value()
        stationary = bool(final.stationarity_norm <= gradient_tolerance)
        operator_realized = bool(
            final.hard_period_gap <= operator_tolerance)
        converged = stationary and operator_realized
        if converged:
            termination = "converged"
        boundary_after = self.fill.boundary_snapshot()
        boundary_drift = max(
            (abs(boundary_after[edge] - value)
             for edge, value in boundary_before.items()),
            default=0.0,
        )
        return {
            "initial": initial.payload(),
            "final": final.payload(),
            "termination": termination,
            "converged": converged,
            "stationary": stationary,
            "operator_realized": operator_realized,
            "accepted_steps": accepted_steps,
            "boundary_drift": float(boundary_drift),
            "admissible": bool(self.fill.geometry_admissible()),
            "initial_weights": self.initial_weights.tolist(),
            "final_weights": weights.tolist(),
            "lower_bounds": self.lower.tolist(),
            "upper_bounds": self.upper.tolist(),
            "trace": trace,
        }


def operator_metrics(fill, target, seed=20260826, held_out=16):
    """Compare a target only after target-free reconstruction from frozen W."""
    target = np.asarray(target, dtype=complex)
    frozen = fill.read_operator()
    target_logical = _SECTOR_BASIS @ target @ _SECTOR_BASIS.conj().T
    rng = np.random.default_rng(seed)
    transport_errors = []
    amplitude_errors = []
    for _ in range(int(held_out)):
        logical_input = _unit(
            rng.normal(size=2) + 1j * rng.normal(size=2))
        logical_output = _unit(
            rng.normal(size=2) + 1j * rng.normal(size=2))
        raw_input = _SECTOR_BASIS.conj().T @ logical_input
        target_raw_output = target @ raw_input
        predicted_raw_output = (
            _SECTOR_BASIS.conj().T @ frozen.apply(logical_input))
        transport_errors.append(float(np.linalg.norm(
            predicted_raw_output - target_raw_output)))
        expected_amplitude = complex(np.vdot(
            _SECTOR_BASIS.conj().T @ logical_output,
            target_raw_output,
        ))
        amplitude_errors.append(abs(
            frozen.amplitude(logical_output, logical_input)
            - expected_amplitude))

    result = frozen.diagnostics()
    result.update({
        "target_logical": _matrix_payload(target_logical),
        "charge_commutator_error": charge_commutator_error(target),
        "target_sector_charge_leak": sector_charge_leak(target),
        "register_residual": fill.operator_residual(target),
        "hard_period_gap": fill.operator_residual(target, hard=True),
        "transport_operator_error": float(np.linalg.norm(
            frozen.transport - target_logical)),
        "amplitude_operator_error": float(np.linalg.norm(
            frozen.amplitude_operator - target_logical)),
        "held_out_transport_error_max": max(transport_errors, default=0.0),
        "held_out_transport_error_mean": (
            float(np.mean(transport_errors)) if transport_errors else 0.0),
        "held_out_amplitude_error_max": max(amplitude_errors, default=0.0),
        "boundary_drift": fill.boundary_drift(),
    })
    return result


def run_experiment(max_iterations=12, gamma=1.0, seed=20260826):
    """Run the positive controls and the two decisive negative controls."""
    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be at least one")
    if float(gamma) < 0.0:
        raise ValueError("gamma must be non-negative")
    original_convention = cob.HodgeLaplacian.defaultWeightConvention()
    cob.HodgeLaplacian.setDefaultWeightConvention(
        cob.HodgeWeightConvention.SquaredContent)
    try:
        cases = {}
        definitions = [
            ("identity", None, identity_operator(), True),
            ("mapping_class_cycle", _GAMMA, cycle_operator(), True),
            ("generic_charge_preserving", None,
             generic_charge_preserving_operator(), True),
            ("charge_leaking", None, charge_leaking_operator(), False),
        ]
        for offset, (name, twist, target, should_relax) in enumerate(definitions):
            fill = PeriodCobordism(twist=twist)
            before = operator_metrics(fill, target, seed + 101 * offset)
            relaxation = None
            gradient_error = None
            if should_relax:
                relaxer = FixedBoundaryRelaxation(fill, target, gamma=gamma)
                gradient_error = relaxer.finite_difference_error(indices=(0,))
                relaxation = relaxer.relax(max_iterations=max_iterations)
            after = operator_metrics(fill, target, seed + 101 * offset)
            cases[name] = {
                "before": before,
                "after": after,
                "relaxation": relaxation,
                "objective_gradient_fd_error": gradient_error,
            }

        # One-state pinning false positive: this unitary fixes e_0 exactly but
        # negates e_1.  The identity fill accepts the selected state and rejects
        # the complete basis.
        counter_fill = PeriodCobordism()
        reflection = lift_sector_operator(np.diag([1.0, -1.0]))
        one_state = counter_fill.state_residual(
            reflection, np.array([1.0, 0.0]), hard=True)
        complete_basis = counter_fill.operator_residual(reflection, hard=True)

        positive_identity = cases["identity"]["after"]
        positive_cycle = cases["mapping_class_cycle"]["after"]
        generic = cases["generic_charge_preserving"]["after"]
        leaking = cases["charge_leaking"]["after"]
        checks = {
            "identity_recovered": (
                positive_identity["transport_operator_error"] < 1e-8
                and positive_identity["held_out_transport_error_max"] < 1e-8),
            "mapping_class_recovered": (
                positive_cycle["transport_operator_error"] < 1e-8
                and positive_cycle["held_out_transport_error_max"] < 1e-8),
            "generic_target_is_charge_preserving": (
                generic["charge_commutator_error"] < 1e-10),
            "generic_target_not_realized": (
                generic["hard_period_gap"] > 1e-3
                and generic["transport_operator_error"] > 1e-3),
            "lower_residual_is_not_misreported_as_realization": (
                cases["generic_charge_preserving"]["after"]["register_residual"]
                < cases["generic_charge_preserving"]["before"]["register_residual"]
                and abs(
                    cases["generic_charge_preserving"]["after"]["hard_period_gap"]
                    - cases["generic_charge_preserving"]["before"]["hard_period_gap"]
                ) < 1e-9
                and abs(
                    cases["generic_charge_preserving"]["after"][
                        "transport_operator_error"]
                    - cases["generic_charge_preserving"]["before"][
                        "transport_operator_error"]
                ) < 1e-9),
            "charge_leak_rejected": (
                leaking["target_sector_charge_leak"] > 1e-3
                and leaking["hard_period_gap"] > 1e-3),
            "h3_amplitude_not_certified_on_causal_chart": (
                positive_identity["gram_shape_error"] > 1e-3
                and positive_identity["amplitude_operator_error"] > 1e-3),
            "complete_basis_detects_single_state_false_positive": (
                one_state < 1e-10 and complete_basis > 1e-3),
            "all_relaxations_hold_boundaries": all(
                case["relaxation"] is None
                or case["relaxation"]["boundary_drift"] == 0.0
                for case in cases.values()),
            "all_relaxations_remain_admissible": all(
                case["relaxation"] is None
                or case["relaxation"]["admissible"]
                for case in cases.values()),
        }
        checks_passed = bool(all(checks.values()))
        hypothesis_supported = bool(
            generic["hard_period_gap"] < 1e-9
            and generic["transport_operator_error"] < 1e-9
            and positive_identity["amplitude_operator_error"] < 1e-9
            and positive_cycle["amplitude_operator_error"] < 1e-9
        )
        return {
            "schema": 1,
            "objective": "Phi = ||dS/dl2_free||^2 + Gamma * sum_basis r_U",
            "gamma": float(gamma),
            "max_iterations": int(max_iterations),
            "seed": int(seed),
            "cases": cases,
            "single_state_counterexample": {
                "selected_state_hard_gap": one_state,
                "complete_basis_hard_gap": complete_basis,
            },
            "checks": checks,
            "checks_passed": checks_passed,
            "hypothesis_supported": hypothesis_supported,
            "scientific_verdict": "not_supported_on_fixed_prism_topology",
            "conclusion": (
                "The frozen cobordism predicts unseen states for the identity "
                "and mapping-class transport controls. However, metric "
                "relaxation of this fixed topology does not realize a generic "
                "charge-preserving unitary, and the causal chart is not "
                "isometric, so its Gram-corrected amplitude operator does not "
                "satisfy H3 even for the transport controls. Charge conservation "
                "is necessary here, not sufficient."
            ),
        }
    finally:
        cob.HodgeLaplacian.setDefaultWeightConvention(original_convention)


def _print_summary(result):
    print("Intrinsic operator transfer from a frozen relaxed cobordism")
    print("  complete basis: two independent states in ker(1,1,1)")
    for name, case in result["cases"].items():
        after = case["after"]
        relaxation = case["relaxation"]
        line = (
            f"  {name:28} gap={after['hard_period_gap']:.3e} "
            f"||T-U||={after['transport_operator_error']:.3e} "
            f"||A-U||={after['amplitude_operator_error']:.3e} "
            f"held-out={after['held_out_transport_error_max']:.3e} "
            f"charge-leak={after['target_sector_charge_leak']:.3e}"
        )
        if relaxation is not None:
            line += (
                f" steps={relaxation['accepted_steps']} "
                f"stop={relaxation['termination']}"
            )
        print(line)
    counter = result["single_state_counterexample"]
    print(
        "  one-state counterexample: "
        f"selected gap={counter['selected_state_hard_gap']:.3e}, "
        f"complete-basis gap={counter['complete_basis_hard_gap']:.3e}"
    )
    print(f"\n  Conclusion: {result['conclusion']}")
    print("  Checks: " + ("PASS" if result["checks_passed"] else "FAIL"))
    print("  Proposed hypothesis: " + (
        "SUPPORTED" if result["hypothesis_supported"] else "NOT SUPPORTED"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/cobordism/intrinsic_operator_transfer.json"),
    )
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    result = run_experiment(
        max_iterations=args.max_iterations,
        gamma=args.gamma,
        seed=args.seed,
    )
    _print_summary(result)
    if not args.no_write:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"  Raw result: {args.out}")
    raise SystemExit(0 if result["checks_passed"] else 1)


if __name__ == "__main__":
    main()
