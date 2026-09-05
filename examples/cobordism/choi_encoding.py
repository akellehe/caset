# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The Choi-encoding experiment (#936).

Two prepared boundary circles A and B are the inputs; the whole complex
(bulk plus boundary) is the output. An operator U on the direct sum of the two
boundary registers is chosen, its outputs Θ_j = U(ψ_j ⊕ φ_j) are computed
algebraically for a spanning set of input pairs, and the bulk is synthesized
with the boundary geometry and amplitudes fixed so that the whole complex
carries every witness as an eigenstate at one common eigenvalue whose
whole-complex readout is Θ_j. The bulk is then frozen and held-out inputs
(including attachment permutations of the boundary cells) are read by the
Poincaré–Steklov extension and compared with their algebraic outputs.

Representation. The degree-0 operator of `EigenstateSynthesis` is the U(1)
connection Laplacian L = D − A on vertices (edge weights and phases), which is
not a Hodge operator: the process-wide metric source does not enter it, so
this experiment reads the same under diagonal weights and the Whitney pencil
(both recorded). The isolated unit 3-circle's connection Laplacian has
eigenvalues {0, 3, 3}; the two ω-modes span the eigenspace at 3 and are the
boundary qubit of each circle. The whole-complex readout is the coordinate vector of the
witness on the seed bulk's interior vertices in a fixed orthonormal frame
(rows of the unitary discrete Fourier transform); the readout constraints are
imposed exactly by `MultiCobordism.relax_whole_complex_readout_targets`, so
the fit residual measures only whether the whole complex carries the witnesses
as common-eigenvalue eigenstates.

Operator classes (the whitepaper's quasi-free boundary):
  1. one-particle operators U ∈ U(4) on H_A ⊕ H_B: encodable by a linear bulk;
  2. the fermionic Fock lift Γ(U)(ψ ∧ φ): read from the same frozen bulk as
     determinants of one-particle reads (Cauchy–Binet), with the Pfaffian
     decomposability witness for targets no quasi-free lift reaches;
  3. tensor-product operators on ψ ⊗ φ (CNOT, and the identity on ⊗ as the
     control) over the product spanning set {e_a ⊕ e_b}, which is linearly
     dependent in the direct sum: a linear extension forces
     Σ_j c_j Ψ_j = 0 for the dependency c, so the readout targets must satisfy
     Σ_j c_j Θ_j = 0, which every tensor-product operator violates.

Run:  python examples/cobordism/choi_encoding.py --output ~/cobordism-runs/choi-encoding/run.json
"""
from __future__ import annotations

import argparse
import cmath
import itertools
import json
import math
from pathlib import Path

import numpy as np

import tessera
from tessera import cobordism as cob

_OMEGA = cmath.exp(2j * math.pi / 3.0)
# The ω-modes of the unit 3-circle: rows are the logical basis vectors of the
# boundary qubit written as amplitudes on the three ordered vertices.
_MODES = np.array(
    [[1.0, _OMEGA, _OMEGA * _OMEGA], [1.0, _OMEGA * _OMEGA, _OMEGA]],
    dtype=complex,
) / math.sqrt(3.0)
_BASE_CIRCLE = [[0, 1], [1, 2], [0, 2]]
# Dependency of the product spanning set {e_a ⊕ e_b} in the direct sum, in the
# order (0,0), (0,1), (1,0), (1,1):  (e0⊕e0) − (e0⊕e1) − (e1⊕e0) + (e1⊕e1) = 0.
_PRODUCT_PAIRS = [(0, 0), (0, 1), (1, 0), (1, 1)]
_PRODUCT_DEPENDENCY = np.array([1.0, -1.0, -1.0, 1.0], dtype=complex)


def _unit(vector):
    vector = np.asarray(vector, dtype=complex)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("zero vector has no normalized state")
    return vector / norm


def _matrix_payload(matrix):
    return [[[float(v.real), float(v.imag)] for v in row]
            for row in np.asarray(matrix, dtype=complex)]


def _vector_payload(vector):
    return [[float(v.real), float(v.imag)] for v in np.asarray(vector, dtype=complex)]


def _edge_key(edge):
    a, b = edge.getSource().getId(), edge.getTarget().getId()
    return (min(a, b), max(a, b))


def haar_unitary(dimension, rng):
    z = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))[None, :]


def amplitudes_from_logical(logical):
    """Boundary amplitudes on the three ordered vertices of one circle."""
    return _MODES.T @ np.asarray(logical, dtype=complex)


def logical_from_amplitudes(amplitudes):
    """Logical coordinates and the norm of the part outside the qubit."""
    amplitudes = np.asarray(amplitudes, dtype=complex)
    coefficients = _MODES.conj() @ amplitudes
    leak = float(np.linalg.norm(amplitudes - _MODES.T @ coefficients))
    return coefficients, leak


def cnot_operator():
    """CNOT on |ab⟩ = |a⟩ ⊗ |b⟩ with index 2a + b (control a, target b)."""
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


def product_obstruction(tensor_operator):
    """‖Σ_j c_j U|a_j b_j⟩‖ over the product spanning set: zero is necessary
    for a linear bulk to carry the four product witnesses."""
    total = np.zeros(4, dtype=complex)
    for coefficient, (a, b) in zip(_PRODUCT_DEPENDENCY, _PRODUCT_PAIRS):
        total += coefficient * np.asarray(tensor_operator, dtype=complex)[:, 2 * a + b]
    return float(np.linalg.norm(total))


def wedge(u, v):
    """Coordinates of u ∧ v on the ordered pairs a < b."""
    u, v = np.asarray(u, dtype=complex), np.asarray(v, dtype=complex)
    return np.array([u[a] * v[b] - u[b] * v[a]
                     for a, b in itertools.combinations(range(len(u)), 2)])


def pfaffian_4(bivector):
    """Pf of the antisymmetric 4×4 matrix of a bivector in Λ²C⁴: zero iff
    the bivector is decomposable (Slater rank one)."""
    w = {pair: value for pair, value in zip(itertools.combinations(range(4), 2), bivector)}
    return complex(w[(0, 1)] * w[(2, 3)] - w[(0, 2)] * w[(1, 3)] + w[(0, 3)] * w[(1, 2)])


class ChoiEncodingCobordism:
    """Two prepared unit 3-circles (degree 0) joined by an annular bulk of
    `layers` prism layers; the seed bulk's interior vertices carry the readout
    frame. Every boundary edge is pinned facet by facet; bulk edge weights and
    connection phases are the free geometry."""

    def __init__(self, layers=3, frame_size=4, metric_source=None):
        cells = tessera.Spacetime.prismCells(_BASE_CIRCLE, layers, {})
        self.spacetime = tessera.Spacetime.fromCells(2, cells, 1.0, 0.0)
        for edge in self.spacetime.getEdgeList().toVector():
            edge.setLength(1.0)
            edge.setPhase(0.0)
        self.spacetime.materializeFacets()
        top = 3 * layers
        self.vertices_a = [0, 1, 2]
        self.vertices_b = [top, top + 1, top + 2]
        self.interior = [v for v in range(3 * (layers + 1))
                         if v not in self.vertices_a and v not in self.vertices_b]
        if frame_size > len(self.interior):
            raise ValueError("the readout frame cannot exceed the seed interior")
        self.metric_source = (metric_source if metric_source is not None
                              else cob.HodgeLaplacian.defaultMetricSource())
        self.node = cob.MultiCobordism(
            host=self.spacetime, input_targets=[], output_targets=[], degrees=[0],
            einstein_hilbert=False, metric_source=self.metric_source)
        self.node.declare_pinned_region("A", set(self.vertices_a))
        self.node.declare_pinned_region("B", set(self.vertices_b))
        self.cells_a = [[v] for v in self.vertices_a]
        self.cells_b = [[v] for v in self.vertices_b]
        n = len(self.interior)
        dft = np.array([[cmath.exp(-2j * math.pi * r * i / n) for i in range(n)]
                        for r in range(n)]) / math.sqrt(n)
        self.frame = dft[:frame_size]
        # readout_r(Ψ) = Σ_i frame[r, i] Ψ(interior[i]) : the frame coordinates.
        self.readouts = [[([v], complex(self.frame[r, i])) for i, v in enumerate(self.interior)]
                         for r in range(frame_size)]
        self.pinned_edges = sorted(
            _edge_key(e) for e in self.spacetime.getEdgeList().toVector()
            if self.node.edge_is_pinned(*_edge_key(e)))
        self.boundary_initial = self.boundary_snapshot()

    def boundary_snapshot(self):
        edges = {_edge_key(e): e for e in self.node.spacetime().getEdgeList().toVector()}
        return {key: (complex(edges[key].getLength()) ** 2, complex(edges[key].getPhase()))
                for key in self.pinned_edges}

    def boundary_drift(self):
        current = self.boundary_snapshot()
        return float(max((max(abs(current[k][0] - v[0]), abs(current[k][1] - v[1]))
                          for k, v in self.boundary_initial.items()), default=0.0))

    def fit(self, joint_inputs, targets, epsilon=1e-16, boundary_epsilon=1e-12,
            restarts=4, max_growth=8, seed=0, max_iterations=400):
        """Witness j: joint logical input `joint_inputs[j]` ∈ C⁴ (ψ_j ⊕ φ_j)
        and readout target `targets[j]` ∈ C^frame."""
        states_a = [amplitudes_from_logical(x[:2]).tolist() for x in joint_inputs]
        states_b = [amplitudes_from_logical(x[2:]).tolist() for x in joint_inputs]
        return self.node.relax_whole_complex_readout_targets(
            degree=0, region_a="A", cells_a=self.cells_a, states_a=states_a,
            region_b="B", cells_b=self.cells_b, states_b=states_b,
            readouts=self.readouts,
            targets=[[complex(v) for v in t] for t in targets],
            common_eigenvalue=True, epsilon=float(epsilon),
            boundary_epsilon=float(boundary_epsilon), restarts=int(restarts),
            max_growth=int(max_growth), seed=int(seed),
            max_iterations=int(max_iterations))

    def dense_operator(self):
        """The live operator the fit scored, as a dense matrix in canonical
        cell order (columns from unit cochains)."""
        synthesis = cob.EigenstateSynthesis(self.node.spacetime(), 0, self.metric_source)
        n = synthesis.order()
        matrix = np.zeros((n, n), dtype=complex)
        for i in range(n):
            unit = [0j] * n
            unit[i] = 1.0 + 0j
            matrix[:, i] = synthesis.apply(unit)
        cells = [tuple(int(v) for v in c) for c in synthesis.cellSimplices()]
        return matrix, cells

    def read(self, amplitudes_a, amplitudes_b, eigenvalue):
        """Poincaré–Steklov read on the frozen bulk: the whole-complex state
        with the given boundary amplitudes at the common eigenvalue, its
        readout, its eigen-residual, and the Dirichlet gap
        σ_min(L_II − λ) (zero would mean a bulk mode invisible from ∂W)."""
        L, cells = self.dense_operator()
        index = {c: i for i, c in enumerate(cells)}
        ia = [index[(v,)] for v in self.vertices_a]
        ib = [index[(v,)] for v in self.vertices_b]
        ii = [index[(v,)] for v in self.interior]
        boundary = ia + ib
        free = [i for i in range(len(cells)) if i not in boundary]
        psi = np.zeros(len(cells), dtype=complex)
        psi[ia] = amplitudes_a
        psi[ib] = amplitudes_b
        lii = L[np.ix_(free, free)] - eigenvalue * np.eye(len(free))
        lib = L[np.ix_(free, boundary)]
        gap = float(np.linalg.svd(lii, compute_uv=False).min()) if free else float("inf")
        psi[free] = -np.linalg.lstsq(lii, lib @ psi[boundary], rcond=None)[0]
        residual = float(np.linalg.norm(L @ psi - eigenvalue * psi) / np.linalg.norm(psi))
        return {"readout": self.frame @ psi[ii], "state": psi,
                "residual": residual, "dirichlet_gap": gap}

    def read_logical(self, joint_logical, eigenvalue):
        x = np.asarray(joint_logical, dtype=complex)
        return self.read(amplitudes_from_logical(x[:2]), amplitudes_from_logical(x[2:]), eigenvalue)


def _fit_payload(fill, result):
    states = np.asarray(result.states, dtype=complex)
    return {
        "converged": bool(result.converged),
        "residual": float(result.residual),
        "residual_trace": [float(v) for v in result.residual_trace],
        "state_residuals": [float(v) for v in result.state_residuals],
        "state_eigenvalues": [float(v) for v in result.state_eigenvalues],
        "common_eigenvalue": float(result.eigenvalue),
        "common_eigenvalue_spread": float(np.max(
            np.abs(np.asarray(result.state_eigenvalues) - result.eigenvalue), initial=0.0)),
        "growth_steps": int(result.growth_steps),
        "free_edge_count": int(result.free_edge_count),
        "auxiliary_coordinate_count": int(result.auxiliary_cell_count),
        "readout_rank": int(result.readout_rank),
        "readout_deviation": float(result.readout_deviation),
        "boundary_residuals_a": [float(v) for v in result.boundary_residuals_a],
        "boundary_residuals_b": [float(v) for v in result.boundary_residuals_b],
        "boundary_drift": fill.boundary_drift(),
        "witness_count": int(states.shape[0]),
        "cell_count": int(states.shape[1]) if states.ndim == 2 else 0,
    }


def embed(joint, channel):
    """Joint logical vector on `channel` modes per side (ψ ⊕ φ ∈ C^{2·channel})
    as a full joint vector in C⁴ = (mode 0, mode 1 | mode 0, mode 1)."""
    joint = np.asarray(joint, dtype=complex)
    full = np.zeros(4, dtype=complex)
    full[:channel] = joint[:channel]
    full[2:2 + channel] = joint[channel:]
    return full


def restrict(full, channel):
    full = np.asarray(full, dtype=complex)
    return np.concatenate([full[:channel], full[2:2 + channel]])


def one_particle_encoding(fill, operator, seed, held_out_count, channel, **budget):
    """Class 1 on `channel` modes per side: a spanning set of 2·channel generic
    joint inputs, then held-out and attachment-permuted reads on the frozen
    bulk. With one mode per side only the rotations of a circle preserve the
    channel (a reflection swaps the ω-modes), so the attachment permutations
    range over C₃ × C₃; with both modes they range over S₃ × S₃."""
    rng = np.random.default_rng(seed)
    dimension = 2 * channel
    spanning = haar_unitary(dimension, rng)  # columns: joint inputs ψ_j ⊕ φ_j
    inputs = [spanning[:, j] for j in range(dimension)]
    targets = [operator @ x for x in inputs]
    result = fill.fit([embed(x, channel) for x in inputs], targets, seed=seed, **budget)
    payload = {"fit": _fit_payload(fill, result), "target_operator": _matrix_payload(operator),
               "channel_modes_per_side": int(channel)}
    lam = float(result.eigenvalue)

    # The witnesses ARE the extension of their own boundary values.
    states = np.asarray(result.states, dtype=complex)
    witness_errors = []
    for j, x in enumerate(inputs):
        read = fill.read_logical(embed(x, channel), lam)
        witness_errors.append(float(np.abs(read["state"] - states[j]).max()))
    payload["witness_extension_error_max"] = max(witness_errors)

    # Recovered operator from basis reads, held-out reads, and permutations.
    recovered = np.column_stack([fill.read_logical(embed(np.eye(dimension)[:, k], channel), lam)["readout"]
                                 for k in range(dimension)])
    payload["recovered_operator"] = _matrix_payload(recovered)
    payload["operator_error"] = float(np.linalg.norm(recovered - operator))
    errors, residuals, gaps = [], [], []
    for _ in range(int(held_out_count)):
        x = _unit(rng.normal(size=dimension) + 1j * rng.normal(size=dimension))
        read = fill.read_logical(embed(x, channel), lam)
        errors.append(float(np.linalg.norm(read["readout"] - operator @ x)))
        residuals.append(read["residual"])
        gaps.append(read["dirichlet_gap"])
    payload["held_out_error_max"] = max(errors)
    payload["held_out_error_mean"] = float(np.mean(errors))
    payload["held_out_residual_max"] = max(residuals)
    payload["dirichlet_gap_min"] = min(gaps)

    x = embed(_unit(rng.normal(size=dimension) + 1j * rng.normal(size=dimension)), channel)
    amp_a, amp_b = amplitudes_from_logical(x[:2]), amplitudes_from_logical(x[2:])
    attachments = (list(itertools.permutations(range(3))) if channel == 2
                   else [(0, 1, 2), (1, 2, 0), (2, 0, 1)])
    permutation_errors, leaks = [], []
    for sigma_a in attachments:
        for sigma_b in attachments:
            pa, pb = amp_a[list(sigma_a)], amp_b[list(sigma_b)]
            la, leak_a = logical_from_amplitudes(pa)
            lb, leak_b = logical_from_amplitudes(pb)
            full = np.concatenate([la, lb])
            leaks.append(max(leak_a, leak_b, float(np.linalg.norm(full - embed(restrict(full, channel), channel)))))
            expected = operator @ restrict(full, channel)
            read = fill.read(pa, pb, lam)
            permutation_errors.append(float(np.linalg.norm(read["readout"] - expected)))
    payload["attachment_permutation_count"] = len(permutation_errors)
    payload["attachment_permutation_error_max"] = max(permutation_errors)
    payload["attachment_permutation_channel_leak_max"] = max(leaks)
    return payload, recovered, lam


def fock_lift(fill, operator, recovered, eigenvalue, seed, trials, channel):
    """Class 2: two-particle amplitudes (one particle on A, one on B) read
    from the frozen bulk as determinants of one-particle reads, against
    Γ(U)(ψ ∧ φ). With two modes per side the Pfaffian witness records that
    every read output is decomposable while a Bell-type target is not."""
    rng = np.random.default_rng(seed + 7)
    errors = []
    for _ in range(int(trials)):
        psi = _unit(rng.normal(size=channel) + 1j * rng.normal(size=channel))
        phi = _unit(rng.normal(size=channel) + 1j * rng.normal(size=channel))
        x = np.concatenate([psi, np.zeros(channel)])   # one particle on A
        y = np.concatenate([np.zeros(channel), phi])   # one particle on B
        u = fill.read_logical(embed(x, channel), eigenvalue)["readout"]
        v = fill.read_logical(embed(y, channel), eigenvalue)["readout"]
        errors.append(float(np.linalg.norm(wedge(u, v) - wedge(operator @ x, operator @ y))))
    payload = {"two_particle_error_max": max(errors), "trials": int(trials),
               "channel_modes_per_side": int(channel)}
    if channel == 2:
        bell = np.zeros(6, dtype=complex)
        bell[0] = bell[5] = 1.0 / math.sqrt(2.0)  # (e0∧e1 + e2∧e3)/√2
        payload["decomposable_output_pfaffian_max"] = max(
            abs(pfaffian_4(wedge(recovered @ _unit(rng.normal(size=4) + 1j * rng.normal(size=4)),
                                 recovered @ _unit(rng.normal(size=4) + 1j * rng.normal(size=4)))))
            for _ in range(int(trials)))
        payload["bell_target_pfaffian"] = abs(pfaffian_4(bell))
        payload["bell_best_decomposable_overlap"] = 1.0 / math.sqrt(2.0)
    return payload


def tensor_product_encoding(fill_factory, tensor_operator, seed, **budget):
    """Class 3: the product spanning set with a tensor-product operator's
    outputs in the readout frame."""
    inputs = [np.concatenate([np.eye(2)[:, a], np.eye(2)[:, b]]) for a, b in _PRODUCT_PAIRS]
    targets = [np.asarray(tensor_operator, dtype=complex)[:, 2 * a + b] for a, b in _PRODUCT_PAIRS]
    fill = fill_factory()
    result = fill.fit(inputs, targets, seed=seed, **budget)
    payload = {"fit": _fit_payload(fill, result),
               "product_obstruction": product_obstruction(tensor_operator)}
    states = np.asarray(result.states, dtype=complex)
    # The joint inputs have norm √2; the fit scaled each witness by 1/√2.
    dependent = (_PRODUCT_DEPENDENCY[:, None] * states).sum(axis=0)
    L, cells = fill.dense_operator()
    lam = float(result.eigenvalue)
    payload["dependency_witness_norm"] = float(np.linalg.norm(dependent))
    payload["dependency_witness_residual"] = float(
        np.linalg.norm(L @ dependent - lam * dependent) / max(np.linalg.norm(dependent), 1e-300))
    payload["dependency_witness_readout"] = _vector_payload(fill.frame @ dependent[
        [cells.index((v,)) for v in fill.interior]])
    payload["dirichlet_gap"] = fill.read_logical(_unit(np.ones(4)), lam)["dirichlet_gap"]
    return payload


def run_experiment(seed=0, held_out_count=16, layers=3, qubit_layers=3, skip_controls=False,
                   skip_qubits=False, **budget):
    rng = np.random.default_rng(seed + 20260904)
    operator_2 = haar_unitary(2, rng)   # one mode per side: a beam splitter between A and B
    operator = haar_unitary(4, rng)     # both modes per side: a generic U(4)
    record = {"method": {
        "boundary": "two prepared unit 3-circles, degree 0; the ω-modes at eigenvalue 3 are each circle's qubit",
        "bulk": f"annular prism, {layers} layers (one mode per side) and {qubit_layers} layers (qubits); "
                "seed interior vertices carry the readout frame",
        "readout": "coordinates on the seed interior vertices in the first 2·channel rows of the unitary DFT",
        "fit": "relax_whole_complex_readout_targets: boundary geometry and amplitudes fixed, readouts exact, common-eigenvalue Rayleigh residual",
        "read": "Poincaré–Steklov extension on the frozen bulk at the common eigenvalue",
        "seed": int(seed), **{k: (float(v) if isinstance(v, float) else int(v)) for k, v in budget.items()},
    }}
    # Class 1 with one mode per side (a two-dimensional direct sum).
    fill_2 = ChoiEncodingCobordism(layers=layers, frame_size=2)
    class1_2, recovered_2, lam_2 = one_particle_encoding(fill_2, operator_2, seed, held_out_count, 1, **budget)
    record["class_1_one_mode_per_side"] = class1_2
    record["class_2_fock_lift_one_mode_per_side"] = fock_lift(fill_2, operator_2, recovered_2, lam_2, seed,
                                                              held_out_count, 1)
    if skip_qubits:
        record["checks"] = _checks(record, [("one_mode_per_side", class1_2)], None)
        return record
    # Class 1 with both modes per side (qubits; a four-dimensional direct sum).
    factory = lambda: ChoiEncodingCobordism(layers=qubit_layers, frame_size=4)  # noqa: E731
    fill = factory()
    class1, recovered, lam = one_particle_encoding(fill, operator, seed, held_out_count, 2, **budget)
    record["class_1_qubits"] = class1
    record["class_2_fock_lift_qubits"] = fock_lift(fill, operator, recovered, lam, seed, held_out_count, 2)
    record["class_3_cnot"] = tensor_product_encoding(factory, cnot_operator(), seed, **budget)
    if not skip_controls:
        record["class_3_identity_control"] = tensor_product_encoding(factory, np.eye(4), seed, **budget)
        # A one-particle operator on the SAME product spanning set is consistent
        # (its targets respect the dependency), isolating the tensor structure.
        consistent = {"fit": None}
        fill_c = factory()
        inputs = [np.concatenate([np.eye(2)[:, a], np.eye(2)[:, b]]) for a, b in _PRODUCT_PAIRS]
        result = fill_c.fit(inputs, [operator @ x for x in inputs], seed=seed, **budget)
        consistent["fit"] = _fit_payload(fill_c, result)
        consistent["product_obstruction"] = float(np.linalg.norm(
            sum(c * (operator @ x) for c, x in zip(_PRODUCT_DEPENDENCY, inputs))))
        record["class_3_one_particle_control"] = consistent
    record["checks"] = _checks(record, [("one_mode_per_side", class1_2), ("qubits", class1)],
                               None if skip_controls else "class_3_one_particle_control")
    return record


def _checks(record, class1_payloads, control_key):
    checks = {}
    for label, payload in class1_payloads:
        checks[f"class_1_{label}_fit_converged"] = payload["fit"]["converged"]
        checks[f"class_1_{label}_readouts_exact"] = payload["fit"]["readout_deviation"] < 1e-12
        checks[f"class_1_{label}_boundary_bit_identical"] = payload["fit"]["boundary_drift"] == 0.0
        checks[f"class_1_{label}_held_out_reads_match"] = payload["held_out_error_max"] < 1e-6
        checks[f"class_1_{label}_attachment_permutations_match"] = payload["attachment_permutation_error_max"] < 1e-6
        checks[f"class_2_{label}_two_particle_amplitudes_match"] = (
            record[f"class_2_fock_lift_{label}"]["two_particle_error_max"] < 1e-6)
    if "class_3_cnot" in record:
        checks["class_3_cnot_obstructed_algebraically"] = record["class_3_cnot"]["product_obstruction"] > 1.0
        checks["class_3_cnot_fit_floors"] = not record["class_3_cnot"]["fit"]["converged"]
    if control_key is not None:
        checks["class_3_one_particle_control_converged"] = record[control_key]["fit"]["converged"]
    return checks


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", type=int, default=3, help="prism layers for the one-mode-per-side fits")
    parser.add_argument("--qubit-layers", type=int, default=3, help="prism layers for the qubit (four-witness) fits")
    parser.add_argument("--held-out-count", type=int, default=16)
    parser.add_argument("--epsilon", type=float, default=1e-16)
    parser.add_argument("--boundary-epsilon", type=float, default=1e-12)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--max-growth", type=int, default=8)
    parser.add_argument("--max-iterations", type=int, default=400)
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument("--skip-qubits", action="store_true",
                        help="run only the one-mode-per-side track (classes 1 and 2)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    record = run_experiment(
        seed=args.seed, held_out_count=args.held_out_count, layers=args.layers,
        qubit_layers=args.qubit_layers, skip_controls=args.skip_controls,
        skip_qubits=args.skip_qubits, epsilon=args.epsilon,
        boundary_epsilon=args.boundary_epsilon, restarts=args.restarts,
        max_growth=args.max_growth, max_iterations=args.max_iterations)
    text = json.dumps(record, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(json.dumps(record["checks"], indent=2))
    for key in ("class_1_one_mode_per_side", "class_1_qubits", "class_3_cnot"):
        if key not in record:
            continue
        fit = record[key]["fit"]
        print(f"{key}: converged={fit['converged']} residual={fit['residual']:.3e} "
              f"eigenvalue={fit['common_eigenvalue']:.4f} growth={fit['growth_steps']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
