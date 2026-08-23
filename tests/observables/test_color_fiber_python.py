# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Exact fixtures for the three-edge SU(3) color kernel (issue #767):
ColorFiber (constant algebra, normalizers, certificates, sector reads) and
ColorAnchor (the calibrated weighted oriented-triangle anchoring kernel).

Acceptance coverage (ticket #767):

* F3^dag F3 = I and |det F3| = 1;
* Gell-Mann matrices Hermitian/traceless with Tr(lambda_a lambda_b) =
  2 delta_ab;
* [E_ij, E_kl] = delta_jk E_il - delta_il E_kj on the 3x3 matrix units AND
  the full 8x8 Fock bilinears;
* det(gC) = det(C) for random CERTIFIED g in SU(3);
* the singlet wedge vanishes for duplicate color modes and reaches unit
  Gram determinant for an orthonormal triad;
* literal-triangle and extended-atlas anchor fixtures pass, an abstract
  unanchored rank-three band fails, the calibrated score never exceeds one
  (to round-off) and reaches one exactly on the concentrated oracle, and
  post-hoc weight selection is rejected;
* every read is invariant under oriented edge relabeling and in-band SU(3)
  frame changes.

Exactness bar: algebraic identities are compared at double round-off
(~1e-15); expected values are built from sqrt(3) and the cube root of
unity ALGEBRAICALLY, with floating representations compared only at the
final boundary.  Anything looser (matrix products, eigen-modulus paths) is
labeled with its honest tolerance.

The references are INDEPENDENT NumPy constructions (exp-based DFT, a
hardcoded Gell-Mann table, dense Jordan-Wigner kron chains as in
tests/quantum/test_graded_fock_python.py, and a standalone NumPy anchor
evaluator) — never re-derivations through the bindings under test.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

import tessera
from tessera.quantum import ExteriorAlgebra

ColorFiber = tessera.ColorFiber
ColorAnchor = tessera.ColorAnchor
OrientedTriangle = tessera.OrientedTriangle

SQRT3 = math.sqrt(3.0)
# Algebraic omega = (-1 + i sqrt(3)) / 2 — the same closed form the kernel
# documents; the EXP-based value below is the independent cross-check.
OMEGA_ALG = complex(-0.5, SQRT3 / 2.0)
OMEGA_EXP = np.exp(2j * np.pi / 3.0)


# ─── independent references ────────────────────────────────────────────────

def dense(coo):
    """(rows, cols, values, n) COO tuple -> dense complex ndarray."""
    rows, cols, vals, n = coo
    out = np.zeros((n, n), dtype=complex)
    for r, c, v in zip(rows, cols, vals):
        out[r, c] += v
    return out


_S_MINUS = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
_Z = np.diag([1.0, -1.0]).astype(complex)


def jw_annihilation(mode: int, n_modes: int) -> np.ndarray:
    """Independent dense Jordan-Wigner a_mode on the n(b) = sum b_i 2^i
    basis (same reference construction as test_graded_fock_python.py)."""
    op = np.eye(1, dtype=complex)
    for m in range(n_modes - 1, -1, -1):
        if m > mode:
            factor = np.eye(2, dtype=complex)
        elif m == mode:
            factor = _S_MINUS
        else:
            factor = _Z
        op = np.kron(op, factor)
    return op


def jw_creation(mode: int, n_modes: int) -> np.ndarray:
    return jw_annihilation(mode, n_modes).conj().T


# The standard Gell-Mann table, hardcoded independently of the C++ path.
GELL_MANN_REF = {
    1: np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex),
    2: np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=complex),
    3: np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=complex),
    4: np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=complex),
    5: np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=complex),
    6: np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex),
    7: np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=complex),
    8: np.diag([1, 1, -2]).astype(complex) / SQRT3,
}


def random_su3(rng) -> np.ndarray:
    """A Haar-ish random SU(3) element, CERTIFIED before use."""
    z = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    q, r = np.linalg.qr(z)
    q = q @ np.diag(r.diagonal() / np.abs(r.diagonal()))
    q = q / np.linalg.det(q) ** (1.0 / 3.0)
    # Certification is part of the fixture: reject a bad sample loudly.
    assert np.max(np.abs(q.conj().T @ q - np.eye(3))) <= 1e-12
    assert abs(np.linalg.det(q) - 1.0) <= 1e-12
    return q


def ref_anchor_terms(frame, edge_weights, triangles):
    """Standalone NumPy evaluation of |det(|W_tau|^{1/2} R_tau Phi)|^2 and
    det phases for a DIAGONAL edge-weight vector."""
    absw = np.abs(np.asarray(edge_weights, dtype=float))
    terms, phases = [], []
    for edges, signs in triangles:
        rows = np.array(
            [s * np.asarray(frame)[e, :] for e, s in zip(edges, signs)])
        a = np.diag(np.sqrt(absw[list(edges)])) @ rows
        d = np.linalg.det(a)
        terms.append(abs(d) ** 2)
        phases.append(np.angle(d) if d != 0 else np.nan)
    return np.array(terms), np.array(phases)


def ref_profile(frame, edge_weights, triangles, weights):
    """Standalone NumPy score / participation ratio / phase coherence."""
    terms, phases = ref_anchor_terms(frame, edge_weights, triangles)
    w = np.asarray(weights, dtype=float)
    score = float(np.sum(w * terms))
    sum_t, sum_t2 = float(np.sum(terms)), float(np.sum(terms**2))
    pr = (sum_t * sum_t / sum_t2) if sum_t2 > 0 else 0.0
    u = w * terms
    mask = terms > 0
    if np.sum(u[mask]) > 0:
        coherence = abs(np.sum(u[mask] * np.exp(1j * phases[mask]))) / float(
            np.sum(u[mask]))
    else:
        coherence = np.nan
    return score, terms, pr, phases, coherence


def orthonormal_band(rng, n_edges, edge_weights):
    """A random rank-three |W|-orthonormal band over n_edges edges."""
    frame = rng.normal(size=(n_edges, 3)) + 1j * rng.normal(size=(n_edges, 3))
    return ColorAnchor.orthonormalizeFrame(frame, np.asarray(edge_weights))


# ─── the exact Fourier frame from omega ────────────────────────────────────

class TestFourierFrame(unittest.TestCase):
    """F3 built from the cube root of unity: unitarity, |det| = 1, and the
    identification of (1, omega, omega^2)/sqrt(3) with its cyclic triad."""

    def test_omega_is_the_algebraic_cube_root(self) -> None:
        w = ColorFiber.omega()
        self.assertEqual(w, OMEGA_ALG)
        # Cross-check against the independent exp-based value at the final
        # floating boundary.
        self.assertLessEqual(abs(w - OMEGA_EXP), 1e-15)

    def test_omega_cubic_identities(self) -> None:
        w = ColorFiber.omega()
        # 1 + omega + conj(omega) cancels EXACTLY with the algebraic
        # components (conj(omega) is the algebraic omega^2).
        self.assertEqual(1.0 + w + np.conj(w), 0.0)
        self.assertLessEqual(abs(w**3 - 1.0), 1e-15)
        self.assertLessEqual(abs(w**2 - np.conj(w)), 1e-15)

    def test_f3_dagger_f3_is_identity(self) -> None:
        f = ColorFiber.fourierFrame()
        self.assertLessEqual(
            np.max(np.abs(f.conj().T @ f - np.eye(3))), 1e-15)

    def test_f3_det_modulus_one(self) -> None:
        f = ColorFiber.fourierFrame()
        self.assertLessEqual(abs(abs(np.linalg.det(f)) - 1.0), 1e-15)

    def test_f3_matches_independent_exp_dft(self) -> None:
        f = ColorFiber.fourierFrame()
        j, k = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")
        ref = OMEGA_EXP ** (j * k) / SQRT3
        self.assertLessEqual(np.max(np.abs(f - ref)), 1e-15)

    def test_omega_phase_state_is_one_basis_vector(self) -> None:
        v = ColorFiber.omegaPhaseState()
        ref = np.array([1.0, OMEGA_ALG, np.conj(OMEGA_ALG)]) / SQRT3
        self.assertLessEqual(np.max(np.abs(v - ref)), 1e-15)
        f = ColorFiber.fourierFrame()
        self.assertTrue(np.array_equal(v, f[:, 1]))
        self.assertTrue(
            np.array_equal(v, ColorFiber.fourierBasisVector(1)))

    def test_cyclic_triad_is_orthonormal(self) -> None:
        cols = [ColorFiber.fourierBasisVector(k) for k in range(3)]
        for a in range(3):
            for b in range(3):
                inner = np.vdot(cols[a], cols[b])
                self.assertLessEqual(
                    abs(inner - (1.0 if a == b else 0.0)), 1e-15)

    def test_cyclic_triad_generated_by_pointwise_z3_powers(self) -> None:
        # v_k are Z3 characters: sqrt(3) * (v1 ∘ v1) = v2 and
        # sqrt(3) * (v1 ∘ v2) = v0 — the cyclic orbit of the omega pattern.
        v0 = ColorFiber.fourierBasisVector(0)
        v1 = ColorFiber.fourierBasisVector(1)
        v2 = ColorFiber.fourierBasisVector(2)
        self.assertLessEqual(np.max(np.abs(SQRT3 * v1 * v1 - v2)), 1e-15)
        self.assertLessEqual(np.max(np.abs(SQRT3 * v1 * v2 - v0)), 1e-15)

    def test_fourier_basis_vector_range_errors(self) -> None:
        with self.assertRaises(ValueError):
            ColorFiber.fourierBasisVector(3)
        with self.assertRaises(ValueError):
            ColorFiber.fourierBasisVector(-1)

    def test_constants_regenerate_identically(self) -> None:
        # The constant algebra is a pure function: two generations agree
        # bitwise (no state, no cache to drift — cold recomputation IS the
        # production path).
        self.assertTrue(
            np.array_equal(ColorFiber.fourierFrame(),
                           ColorFiber.fourierFrame()))
        for a in range(1, 9):
            self.assertTrue(
                np.array_equal(ColorFiber.gellMann(a),
                               ColorFiber.gellMann(a)))


# ─── Gell-Mann generators on the one-occupation sector ─────────────────────

class TestGellMann(unittest.TestCase):
    def test_hermitian_exact(self) -> None:
        for a in range(1, 9):
            la = ColorFiber.gellMann(a)
            self.assertEqual(np.max(np.abs(la - la.conj().T)), 0.0)

    def test_traceless_exact(self) -> None:
        for a in range(1, 9):
            self.assertEqual(np.trace(ColorFiber.gellMann(a)), 0.0)

    def test_trace_orthonormalization_two_delta(self) -> None:
        for a in range(1, 9):
            for b in range(1, 9):
                tr = np.trace(ColorFiber.gellMann(a) @ ColorFiber.gellMann(b))
                expected = 2.0 if a == b else 0.0
                self.assertLessEqual(abs(tr - expected), 1e-15,
                                     msg=f"Tr(l{a} l{b})")

    def test_matches_independent_table(self) -> None:
        for a in range(1, 9):
            self.assertLessEqual(
                np.max(np.abs(ColorFiber.gellMann(a) - GELL_MANN_REF[a])),
                1e-15, msg=f"lambda_{a}")

    def test_cartan_generators_h1_h2(self) -> None:
        u = ColorFiber.matrixUnit
        self.assertTrue(
            np.array_equal(ColorFiber.gellMann(3), u(0, 0) - u(1, 1)))
        h2 = (u(0, 0) + u(1, 1) - 2.0 * u(2, 2)) / SQRT3
        self.assertLessEqual(
            np.max(np.abs(ColorFiber.gellMann(8) - h2)), 1e-15)

    def test_index_errors(self) -> None:
        for bad in (0, 9, -1):
            with self.assertRaises(ValueError):
                ColorFiber.gellMann(bad)


# ─── E_ij bilinears and the gl(3) commutator identity ──────────────────────

class TestBilinears(unittest.TestCase):
    def test_gl3_commutators_on_matrix_units_exact(self) -> None:
        u = ColorFiber.matrixUnit
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for L in range(3):
                        lhs = u(i, j) @ u(k, L) - u(k, L) @ u(i, j)
                        rhs = np.zeros((3, 3), dtype=complex)
                        if j == k:
                            rhs += u(i, L)
                        if i == L:
                            rhs -= u(k, j)
                        self.assertEqual(np.max(np.abs(lhs - rhs)), 0.0,
                                         msg=f"[E{i}{j}, E{k}{L}]")

    def test_gl3_commutators_on_fock_bilinears_exact(self) -> None:
        e = {(i, j): ColorFiber.hoppingMatrix(i, j)
             for i in range(3) for j in range(3)}
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for L in range(3):
                        lhs = e[i, j] @ e[k, L] - e[k, L] @ e[i, j]
                        rhs = np.zeros((8, 8), dtype=complex)
                        if j == k:
                            rhs += e[i, L]
                        if i == L:
                            rhs -= e[k, j]
                        self.assertEqual(np.max(np.abs(lhs - rhs)), 0.0,
                                         msg=f"[E{i}{j}, E{k}{L}] (Fock)")

    def test_creation_annihilation_match_jordan_wigner(self) -> None:
        for i in range(3):
            self.assertEqual(
                np.max(np.abs(ColorFiber.creationMatrix(i) -
                              jw_creation(i, 3))), 0.0)
            self.assertEqual(
                np.max(np.abs(ColorFiber.annihilationMatrix(i) -
                              jw_annihilation(i, 3))), 0.0)

    def test_car_anticommutators(self) -> None:
        for i in range(3):
            for j in range(3):
                ai = ColorFiber.annihilationMatrix(i)
                cj = ColorFiber.creationMatrix(j)
                anti = ai @ cj + cj @ ai
                expected = np.eye(8) if i == j else np.zeros((8, 8))
                self.assertEqual(np.max(np.abs(anti - expected)), 0.0)

    def test_hopping_matches_jw_product(self) -> None:
        for i in range(3):
            for j in range(3):
                ref = jw_creation(i, 3) @ jw_annihilation(j, 3)
                self.assertEqual(
                    np.max(np.abs(ColorFiber.hoppingMatrix(i, j) - ref)),
                    0.0)

    def test_triplet_basis_indices(self) -> None:
        self.assertEqual(tuple(ColorFiber.tripletBasisIndices()), (1, 2, 4))

    def test_restrict_hopping_is_matrix_unit(self) -> None:
        for i in range(3):
            for j in range(3):
                got = ColorFiber.restrictToTriplet(
                    ColorFiber.hoppingMatrix(i, j))
                self.assertEqual(
                    np.max(np.abs(got - ColorFiber.matrixUnit(i, j))), 0.0)

    def test_restrict_dgamma_is_identity_map(self) -> None:
        rng = np.random.default_rng(11)
        mats = [ColorFiber.gellMann(a) for a in range(1, 9)]
        mats.append(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
        for m in mats:
            got = ColorFiber.restrictToTriplet(ColorFiber.dGamma(m))
            self.assertLessEqual(np.max(np.abs(got - m)), 1e-15)

    def test_dgamma_annihilates_vacuum_sector(self) -> None:
        rng = np.random.default_rng(12)
        m = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        dg = ColorFiber.dGamma(m)
        vac = np.zeros(8, dtype=complex)
        vac[0] = 1.0
        self.assertEqual(np.max(np.abs(dg @ vac)), 0.0)

    def test_shape_and_index_errors(self) -> None:
        with self.assertRaises(ValueError):
            ColorFiber.restrictToTriplet(np.eye(3, dtype=complex))
        with self.assertRaises(ValueError):
            ColorFiber.matrixUnit(3, 0)
        with self.assertRaises(ValueError):
            ColorFiber.creationMatrix(3)


# ─── the N = 0,1,2,3 sector projectors ─────────────────────────────────────

class TestSectorProjectors(unittest.TestCase):
    def popcount_projector(self, n: int) -> np.ndarray:
        return np.diag([1.0 if bin(b).count("1") == n else 0.0
                        for b in range(8)]).astype(complex)

    def test_match_independent_popcount_masks(self) -> None:
        for n in range(4):
            self.assertEqual(
                np.max(np.abs(ColorFiber.sectorProjector(n) -
                              self.popcount_projector(n))), 0.0)

    def test_idempotent_orthogonal_complete(self) -> None:
        projectors = [ColorFiber.sectorProjector(n) for n in range(4)]
        total = np.zeros((8, 8), dtype=complex)
        for a, p in enumerate(projectors):
            self.assertEqual(np.max(np.abs(p @ p - p)), 0.0)
            for b in range(a):
                self.assertEqual(
                    np.max(np.abs(p @ projectors[b])), 0.0)
            total += p
        self.assertEqual(np.max(np.abs(total - np.eye(8))), 0.0)

    def test_sector_dimensions_1_3_3_1(self) -> None:
        for n, dim in zip(range(4), (1, 3, 3, 1)):
            self.assertEqual(np.trace(ColorFiber.sectorProjector(n)),
                             complex(dim))

    def test_named_projectors_are_the_sectors(self) -> None:
        self.assertTrue(np.array_equal(ColorFiber.vacuumProjector(),
                                       ColorFiber.sectorProjector(0)))
        self.assertTrue(np.array_equal(ColorFiber.tripletProjector(),
                                       ColorFiber.sectorProjector(1)))
        self.assertTrue(np.array_equal(ColorFiber.antiTripletProjector(),
                                       ColorFiber.sectorProjector(2)))
        self.assertTrue(np.array_equal(ColorFiber.singletProjector(),
                                       ColorFiber.sectorProjector(3)))

    def test_sector_above_top_is_zero(self) -> None:
        self.assertEqual(np.max(np.abs(ColorFiber.sectorProjector(4))), 0.0)

    def test_fermion_parity_pattern(self) -> None:
        # 1 ⊕ 3 ⊕ 3̄ ⊕ 1 parities: even, odd, even, odd.
        parity = np.diag([(-1.0) ** bin(b).count("1") for b in range(8)])
        for n, sign in zip(range(4), (+1.0, -1.0, +1.0, -1.0)):
            p = ColorFiber.sectorProjector(n)
            self.assertEqual(np.max(np.abs(parity @ p - sign * p)), 0.0)

    def test_delegation_matches_quantum_primitive(self) -> None:
        # The color API layers interpretation over the #766 primitive — the
        # matrices must be the SAME object content.
        alg = ExteriorAlgebra(3)
        for n in range(4):
            self.assertEqual(
                np.max(np.abs(ColorFiber.sectorProjector(n) -
                              dense(alg.sectorProjectorCOO(n)))), 0.0)


# ─── the traceless adjoint-octet projector ─────────────────────────────────

class TestAdjointOctet(unittest.TestCase):
    def test_matches_independent_construction(self) -> None:
        vec_i = np.eye(3, dtype=complex).reshape(9, order="F")
        ref = np.eye(9) - np.outer(vec_i, vec_i.conj()) / 3.0
        self.assertLessEqual(
            np.max(np.abs(ColorFiber.adjointOctetProjector() - ref)), 1e-15)

    def test_projector_algebra(self) -> None:
        p8 = ColorFiber.adjointOctetProjector()
        self.assertLessEqual(np.max(np.abs(p8 - p8.conj().T)), 1e-15)
        self.assertLessEqual(np.max(np.abs(p8 @ p8 - p8)), 1e-15)
        self.assertLessEqual(abs(np.trace(p8) - 8.0), 1e-15)

    def test_fixes_gellmann_kills_identity(self) -> None:
        p8 = ColorFiber.adjointOctetProjector()
        for a in range(1, 9):
            v = ColorFiber.gellMann(a).reshape(9, order="F")
            self.assertLessEqual(np.max(np.abs(p8 @ v - v)), 1e-15)
        vec_i = np.eye(3, dtype=complex).reshape(9, order="F")
        self.assertLessEqual(np.max(np.abs(p8 @ vec_i)), 1e-15)

    def test_vec_convention_is_column_major(self) -> None:
        rng = np.random.default_rng(21)
        m = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        p8 = ColorFiber.adjointOctetProjector()
        got = p8 @ m.reshape(9, order="F")
        want = ColorFiber.tracelessPart(m).reshape(9, order="F")
        self.assertLessEqual(np.max(np.abs(got - want)), 1e-15)

    def test_octet_read_frobenius_split(self) -> None:
        rng = np.random.default_rng(22)
        m = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        read = ColorFiber.octetRead(m)
        frob = np.linalg.norm(m, "fro") ** 2
        self.assertLessEqual(abs(read.octet + read.singlet - frob),
                             1e-13 * frob)
        self.assertLessEqual(
            abs(read.singlet - abs(np.trace(m)) ** 2 / 3.0), 1e-13 * frob)

    def test_octet_read_on_generators_and_identity(self) -> None:
        for a in range(1, 9):
            read = ColorFiber.octetRead(ColorFiber.gellMann(a))
            self.assertLessEqual(abs(read.octet - 2.0), 1e-14)
            self.assertLessEqual(read.singlet, 1e-15)
        read = ColorFiber.octetRead(np.eye(3, dtype=complex))
        self.assertEqual(read.octet, 0.0)
        self.assertEqual(read.singlet, 3.0)


# ─── perimeter vs Hilbert normalization; the color vector ──────────────────

class TestNormalizers(unittest.TestCase):
    def test_color_vector_is_unit_and_parallel(self) -> None:
        rng = np.random.default_rng(31)
        z = rng.normal(size=3) + 1j * rng.normal(size=3)
        c = ColorFiber.colorVector(z)
        self.assertLessEqual(abs(np.vdot(c, c).real - 1.0), 1e-15)
        self.assertLessEqual(np.max(np.abs(c * np.linalg.norm(z) - z)),
                             1e-15 * np.linalg.norm(z))
        self.assertTrue(np.array_equal(c, ColorFiber.hilbertNormalized(z)))

    def test_omega_pattern_color_vector_is_fourier_basis_vector(self) -> None:
        # Unit-modulus squared lengths with the omega phases: the color
        # vector IS the identified Fourier basis vector.
        z = np.array([1.0, OMEGA_ALG, np.conj(OMEGA_ALG)])
        c = ColorFiber.colorVector(z)
        self.assertLessEqual(
            np.max(np.abs(c - ColorFiber.omegaPhaseState())), 1e-15)

    def test_perimeter_independent_reference(self) -> None:
        rng = np.random.default_rng(32)
        z = rng.normal(size=3) + 1j * rng.normal(size=3)
        ref = float(np.sum(np.sqrt(np.abs(z))))
        self.assertLessEqual(abs(ColorFiber.perimeter(z) - ref), 1e-15 * ref)
        # Algebraic fixture: |z_i| = 1 each -> perimeter exactly 3.
        z3 = np.array([1.0, OMEGA_ALG, np.conj(OMEGA_ALG)])
        self.assertLessEqual(abs(ColorFiber.perimeter(z3) - 3.0), 1e-15)

    def test_perimeter_normalized_has_unit_perimeter(self) -> None:
        rng = np.random.default_rng(33)
        z = 3.0 * (rng.normal(size=3) + 1j * rng.normal(size=3))
        zn = ColorFiber.perimeterNormalized(z)
        self.assertLessEqual(abs(ColorFiber.perimeter(zn) - 1.0), 1e-14)

    def test_perimeter_and_hilbert_are_distinct_apis(self) -> None:
        # The L1 scale gauge is NOT the L2 state normalization: on a
        # generic triangle they disagree, and the perimeter-normalized
        # vector is not a unit Hilbert vector.
        z = np.array([2.0 + 0.5j, -0.25 + 1.0j, 0.75 - 0.3j])
        zp = ColorFiber.perimeterNormalized(z)
        zh = ColorFiber.hilbertNormalized(z)
        self.assertGreater(np.max(np.abs(zp - zh)), 1e-3)
        self.assertGreater(abs(np.linalg.norm(zp) - 1.0), 1e-3)
        self.assertLessEqual(abs(np.linalg.norm(zh) - 1.0), 1e-15)

    def test_scale_gauge_covariance(self) -> None:
        rng = np.random.default_rng(34)
        z = rng.normal(size=3) + 1j * rng.normal(size=3)
        s = 2.75
        zp1 = ColorFiber.perimeterNormalized(z)
        zp2 = ColorFiber.perimeterNormalized(s * s * z)
        self.assertLessEqual(np.max(np.abs(zp1 - zp2)), 1e-14)

    def test_zero_inputs_raise(self) -> None:
        zero = np.zeros(3, dtype=complex)
        with self.assertRaises(ValueError):
            ColorFiber.colorVector(zero)
        with self.assertRaises(ValueError):
            ColorFiber.hilbertNormalized(zero)
        with self.assertRaises(ValueError):
            ColorFiber.perimeterNormalized(zero)


# ─── det(C) / det(C†C) certificates ────────────────────────────────────────

class TestWedgeCertificates(unittest.TestCase):
    def test_orthonormal_triad_reaches_unit_gram(self) -> None:
        f = ColorFiber.fourierFrame()
        self.assertLessEqual(abs(ColorFiber.singletGram(f) - 1.0), 1e-15)
        rng = np.random.default_rng(41)
        for _ in range(5):
            g = random_su3(rng)
            self.assertLessEqual(abs(ColorFiber.singletGram(g) - 1.0),
                                 1e-13)

    def test_duplicate_color_modes_vanish(self) -> None:
        # Honest precision label: the det-based certificate cancels a
        # duplicate mode at double ROUND-OFF (Eigen's 3x3 determinant
        # expands along the first column, so the cancellation is not
        # bitwise); the EXACT zero is the exterior-algebra wedge below.
        rng = np.random.default_rng(42)
        a = rng.normal(size=3) + 1j * rng.normal(size=3)
        b = rng.normal(size=3) + 1j * rng.normal(size=3)
        scale = float(np.linalg.norm(a) ** 2 * np.linalg.norm(b))
        for cols in ((a, a, b), (a, b, a), (b, a, a)):
            self.assertLessEqual(
                abs(ColorFiber.colorWedgeColumns(*cols)), 1e-15 * scale)
        c = np.column_stack([a, a, b])
        self.assertLessEqual(abs(ColorFiber.colorWedge(c)), 1e-15 * scale)
        self.assertLessEqual(ColorFiber.singletGram(c),
                             (1e-15 * scale) ** 2)

    def test_duplicate_color_modes_wedge_to_exact_zero_in_fock(self) -> None:
        # The #766 primitive carries the EXACT Pauli identity for duplicate
        # COMPLETE modes (its own documented exactness domain): the wedge is
        # bitwise zero, and every color-sector read of it is exactly zero.
        alg = ExteriorAlgebra(3)
        basis = np.eye(3, dtype=complex)
        psi = alg.wedge([basis[0], basis[0], basis[2]])
        self.assertEqual(np.max(np.abs(psi)), 0.0)
        w = ColorFiber.sectorWeights(np.asarray(psi))
        self.assertEqual((w.vacuum, w.quark, w.anti_triplet, w.singlet),
                         (0.0, 0.0, 0.0, 0.0))
        # A repeated GENERAL complex color column cancels at double
        # round-off (documented in the #766 suite: FMA contraction), in
        # agreement with the det-based certificate above.
        rng = np.random.default_rng(42)
        a = rng.normal(size=3) + 1j * rng.normal(size=3)
        b = rng.normal(size=3) + 1j * rng.normal(size=3)
        psi2 = np.asarray(alg.wedge([a, a, b]))
        self.assertLessEqual(np.max(np.abs(psi2)), 1e-14)

    def test_wedge_antisymmetry_exact(self) -> None:
        rng = np.random.default_rng(43)
        a = rng.normal(size=3) + 1j * rng.normal(size=3)
        b = rng.normal(size=3) + 1j * rng.normal(size=3)
        c = rng.normal(size=3) + 1j * rng.normal(size=3)
        self.assertEqual(ColorFiber.colorWedgeColumns(a, b, c),
                         -ColorFiber.colorWedgeColumns(b, a, c))

    def test_det_gc_equals_det_c_for_certified_su3(self) -> None:
        rng = np.random.default_rng(44)
        c = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        det_c = ColorFiber.colorWedge(c)
        for _ in range(20):
            g = random_su3(rng)
            self.assertTrue(ColorFiber.isSpecialUnitary(g, 1e-12))
            self.assertLessEqual(
                abs(ColorFiber.colorWedge(g @ c) - det_c),
                1e-12 * max(1.0, abs(det_c)))
            # And the Gram certificate is invariant too.
            self.assertLessEqual(
                abs(ColorFiber.singletGram(g @ c) -
                    ColorFiber.singletGram(c)),
                1e-12 * max(1.0, abs(det_c)) ** 2)

    def test_singlet_gram_is_abs_wedge_squared(self) -> None:
        rng = np.random.default_rng(45)
        c = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        self.assertLessEqual(
            abs(ColorFiber.singletGram(c) - abs(ColorFiber.colorWedge(c))**2),
            1e-13 * max(1.0, abs(ColorFiber.colorWedge(c)) ** 2))

    def test_wedge_matches_exterior_algebra_top_sector(self) -> None:
        # Cross-representation: the SAME certificate through the Fock wedge
        # |psi> = a†(c1) a†(c2) a†(c3) |vac>:  ||psi||^2 = det(C†C) and psi
        # is a pure top-wedge (N=3) state.
        rng = np.random.default_rng(46)
        c = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        creation = [sum(c[i, col] * ColorFiber.creationMatrix(i)
                        for i in range(3)) for col in range(3)]
        vac = np.zeros(8, dtype=complex)
        vac[0] = 1.0
        psi = creation[0] @ (creation[1] @ (creation[2] @ vac))
        norm2 = float(np.vdot(psi, psi).real)
        self.assertLessEqual(abs(norm2 - ColorFiber.singletGram(c)),
                             1e-13 * max(1.0, norm2))
        weights = ColorFiber.sectorWeights(psi)
        self.assertEqual(weights.vacuum, 0.0)
        self.assertEqual(weights.quark, 0.0)
        self.assertEqual(weights.anti_triplet, 0.0)
        self.assertLessEqual(abs(weights.singlet - norm2), 1e-15 * norm2)

    def test_is_special_unitary_negative_controls(self) -> None:
        self.assertFalse(
            ColorFiber.isSpecialUnitary(2.0 * np.eye(3, dtype=complex)))
        # Unitary but det = omega != 1.
        u = np.diag([OMEGA_ALG, 1.0, 1.0])
        self.assertFalse(ColorFiber.isSpecialUnitary(u))
        self.assertTrue(ColorFiber.isSpecialUnitary(np.eye(3, dtype=complex)))


# ─── sector reads (weights only, no classification) ────────────────────────

class TestSectorReads(unittest.TestCase):
    def test_basis_state_weights_exact(self) -> None:
        for b in range(8):
            psi = np.zeros(8, dtype=complex)
            psi[b] = 1.0
            w = ColorFiber.sectorWeights(psi)
            got = (w.vacuum, w.quark, w.anti_triplet, w.singlet)
            expected = [0.0, 0.0, 0.0, 0.0]
            expected[bin(b).count("1")] = 1.0
            self.assertEqual(got, tuple(expected))

    def test_random_state_matches_masks_and_sums(self) -> None:
        rng = np.random.default_rng(51)
        psi = rng.normal(size=8) + 1j * rng.normal(size=8)
        w = ColorFiber.sectorWeights(psi)
        for n, field in zip(range(4),
                            (w.vacuum, w.quark, w.anti_triplet, w.singlet)):
            mask = [bin(b).count("1") == n for b in range(8)]
            self.assertLessEqual(
                abs(field - float(np.sum(np.abs(psi[mask]) ** 2))), 1e-15)
        total = w.vacuum + w.quark + w.anti_triplet + w.singlet
        self.assertLessEqual(abs(total - float(np.vdot(psi, psi).real)),
                             1e-13)

    def test_one_particle_state_reads_quark_sector(self) -> None:
        rng = np.random.default_rng(52)
        v = rng.normal(size=3) + 1j * rng.normal(size=3)
        vac = np.zeros(8, dtype=complex)
        vac[0] = 1.0
        psi = sum(v[i] * ColorFiber.creationMatrix(i)
                  for i in range(3)) @ vac
        w = ColorFiber.sectorWeights(psi)
        self.assertEqual(w.vacuum, 0.0)
        self.assertEqual(w.anti_triplet, 0.0)
        self.assertEqual(w.singlet, 0.0)
        self.assertLessEqual(
            abs(w.quark - float(np.linalg.norm(v) ** 2)), 1e-14)

    def test_diquark_pair_reads_anti_triplet_sector(self) -> None:
        # a_0† a_1† |vac> is a pure two-occupation (anti-triplet) read.
        vac = np.zeros(8, dtype=complex)
        vac[0] = 1.0
        psi = (ColorFiber.creationMatrix(0) @
               (ColorFiber.creationMatrix(1) @ vac))
        w = ColorFiber.sectorWeights(psi)
        self.assertEqual((w.vacuum, w.quark, w.anti_triplet, w.singlet),
                         (0.0, 0.0, 1.0, 0.0))

    def test_weights_invariant_under_mode_relabeling(self) -> None:
        # Oriented edge relabeling acts by the SIGNED permutation unitary of
        # the #766 primitives; every occupation read is invariant.
        rng = np.random.default_rng(53)
        psi = rng.normal(size=8) + 1j * rng.normal(size=8)
        alg = ExteriorAlgebra(3)
        for perm in ([1, 2, 0], [2, 1, 0], [0, 2, 1]):
            u = dense(alg.modePermutationMatrixCOO(perm))
            w0 = ColorFiber.sectorWeights(psi)
            w1 = ColorFiber.sectorWeights(u @ psi)
            for f0, f1 in zip(
                    (w0.vacuum, w0.quark, w0.anti_triplet, w0.singlet),
                    (w1.vacuum, w1.quark, w1.anti_triplet, w1.singlet)):
                self.assertLessEqual(abs(f0 - f1), 1e-15)

    def test_size_error(self) -> None:
        with self.assertRaises(ValueError):
            ColorFiber.sectorWeights(np.zeros(4, dtype=complex))


# ─── the calibrated anchor: oracle, atlas, calibration ─────────────────────

class TestAnchorOracle(unittest.TestCase):
    def test_literal_triangle_oracle_reaches_one(self) -> None:
        rng = np.random.default_rng(61)
        w = np.array([2.0, 0.5, 1.25])
        phi = orthonormal_band(rng, 3, w)
        anchor = ColorAnchor([OrientedTriangle([0, 1, 2], [1, 1, 1])])
        p = anchor.evaluate(phi, w)
        self.assertLessEqual(abs(p.score - 1.0), 1e-13)
        self.assertLessEqual(abs(p.max_term - 1.0), 1e-13)
        self.assertEqual(p.max_term_index, 0)
        self.assertLessEqual(abs(p.participation_ratio - 1.0), 1e-13)
        self.assertTrue(p.positive_regime)
        self.assertEqual(p.krein_signatures, [[3, 0, 0]])
        self.assertLessEqual(p.calibration_margin, 1e-12)
        self.assertLessEqual(p.frame_gram_residual, 1e-12)
        self.assertEqual(p.weighting_id, "uniform")
        self.assertEqual(p.weights, [1.0])
        # The attached #764 certificate (shared vocabulary, no bare read):
        # closed-form given the verified |W|-orthonormal premise on a
        # decoupled diagonal weight.
        cob = tessera.cobordism
        cert = p.certificate
        self.assertEqual(cert.grade, cob.CertificateGrade.StructureExact)
        self.assertEqual(cert.domain, cob.CertificateDomain.Static)
        self.assertEqual(cert.regime,
                         cob.CertificateRegime.PositiveSemidefinite)
        self.assertTrue(cert.holds())
        self.assertLessEqual(cert.residual, 1e-12)
        self.assertEqual(cert.tolerance, 1e-9)
        # Unmeasured quantities are NaN, never zero (#764 convention).
        self.assertTrue(math.isnan(cert.conditioning))
        self.assertTrue(math.isnan(cert.denseReferenceError))

    def test_oracle_exact_algebraic_fixture_f3(self) -> None:
        # Phi = F3 on three unit-weight edges: A_tau IS F3, |det A|^2 = 1.
        f = ColorFiber.fourierFrame()
        w = np.ones(3)
        tri = OrientedTriangle([0, 1, 2], [1, 1, 1])
        a_tau = ColorAnchor.anchorMatrix(f, w, tri)
        self.assertLessEqual(np.max(np.abs(a_tau - f)), 1e-15)
        anchor = ColorAnchor([tri])
        p = anchor.evaluate(f, w)
        # "Reaches one exactly" at the machine-precision bar: unit weights
        # make A_tau = F3 bitwise, so the score is 1 to double round-off.
        self.assertLessEqual(abs(p.score - 1.0), 1e-15)
        # The determinant phase equals arg det F3 (algebraic value -i for
        # this DFT: det F3 = (omega^2 - omega)(...)/3^{3/2} — compare to
        # the independently computed reference).
        ref_phase = float(np.angle(np.linalg.det(np.asarray(f))))
        self.assertLessEqual(abs(np.exp(1j * p.det_phases[0]) -
                                 np.exp(1j * ref_phase)), 1e-13)

    def test_calibrated_score_never_exceeds_one(self) -> None:
        # Property test over random in-domain fixtures: exact-arithmetic
        # bound score <= 1; floating evaluation may exceed it only by
        # round-off (final-boundary comparison at 1e-13).
        rng = np.random.default_rng(62)
        n_edges = 12
        for trial in range(30):
            w = rng.uniform(0.2, 3.0, size=n_edges)
            phi = orthonormal_band(rng, n_edges, w)
            tris = []
            for _ in range(rng.integers(1, 7)):
                edges = rng.choice(n_edges, size=3, replace=False)
                signs = rng.choice([-1, 1], size=3)
                tris.append(OrientedTriangle(
                    [int(e) for e in edges], [int(s) for s in signs]))
            anchor = ColorAnchor(tris)
            p = anchor.evaluate(phi, w)
            self.assertGreaterEqual(p.score, -1e-14, msg=f"trial {trial}")
            self.assertLessEqual(p.score, 1.0 + 1e-13, msg=f"trial {trial}")
            for t in p.terms:
                self.assertLessEqual(t, 1.0 + 1e-13)
                self.assertGreaterEqual(t, 0.0)
            self.assertLessEqual(p.calibration_margin, 1e-12,
                                 msg="diagonal weights are decoupled")

    def test_extended_atlas_matches_independent_reference(self) -> None:
        # The production case: an extended fiber anchored by an atlas of
        # overlapping oriented triangles.  Full profile against the
        # standalone NumPy evaluator.
        rng = np.random.default_rng(63)
        n_edges = 6
        w = rng.uniform(0.5, 2.0, size=n_edges)
        phi = orthonormal_band(rng, n_edges, w)
        tri_spec = [((0, 1, 2), (1, 1, 1)), ((2, 3, 4), (1, -1, 1)),
                    ((0, 3, 5), (-1, 1, 1)), ((1, 4, 5), (1, 1, -1))]
        tris = [OrientedTriangle(list(e), list(s)) for e, s in tri_spec]
        conv = [0.4, 0.3, 0.2, 0.1]
        anchor = ColorAnchor(tris, conv)
        p = anchor.evaluate(phi, w)

        score, terms, pr, phases, coherence = ref_profile(
            phi, w, tri_spec, conv)
        self.assertLessEqual(abs(p.score - score), 1e-12)
        self.assertLessEqual(np.max(np.abs(np.array(p.terms) - terms)),
                             1e-12)
        self.assertLessEqual(abs(p.participation_ratio - pr), 1e-10)
        self.assertLessEqual(abs(p.phase_coherence - coherence), 1e-10)
        self.assertLessEqual(
            abs(p.phase_dispersion - (1.0 - coherence)), 1e-10)
        for got, want, t in zip(p.det_phases, phases, terms):
            if t > 1e-20:
                self.assertLessEqual(
                    abs(np.exp(1j * got) - np.exp(1j * want)), 1e-9)
        self.assertLessEqual(abs(p.max_term - float(np.max(terms))), 1e-12)
        self.assertEqual(p.max_term_index, int(np.argmax(terms)))
        self.assertEqual(p.weighting_id, "declared")
        self.assertEqual(p.weights, conv)
        # A genuinely extended read: several triangles participate and the
        # score sits strictly inside the calibrated interval.
        self.assertGreater(p.participation_ratio, 1.0)
        self.assertGreater(p.score, 0.0)
        self.assertLess(p.score, 1.0)

    def test_anchor_matrix_matches_reference(self) -> None:
        rng = np.random.default_rng(64)
        n_edges = 5
        w = rng.uniform(0.1, 2.0, size=n_edges)
        phi = rng.normal(size=(n_edges, 3)) + 1j * rng.normal(
            size=(n_edges, 3))
        tri = OrientedTriangle([4, 0, 2], [-1, 1, -1])
        a_tau = ColorAnchor.anchorMatrix(phi, w, tri)
        rows = np.array([-1 * phi[4, :], +1 * phi[0, :], -1 * phi[2, :]])
        ref = np.diag(np.sqrt(w[[4, 0, 2]])) @ rows
        self.assertLessEqual(np.max(np.abs(a_tau - ref)), 1e-15)

    def test_orthonormalize_frame_enters_domain(self) -> None:
        rng = np.random.default_rng(65)
        w = rng.uniform(0.2, 4.0, size=7)
        phi = orthonormal_band(rng, 7, w)
        gram = phi.conj().T @ np.diag(np.abs(w)) @ phi
        self.assertLessEqual(np.max(np.abs(gram - np.eye(3))), 1e-13)

    def test_orthonormalize_rejects_rank_deficient(self) -> None:
        frame = np.zeros((4, 3), dtype=complex)
        frame[:, 0] = [1.0, 0.0, 0.0, 0.0]
        frame[:, 1] = [0.0, 1.0, 0.0, 0.0]
        frame[:, 2] = [1.0, 1.0, 0.0, 0.0]  # dependent
        with self.assertRaises(ValueError):
            ColorAnchor.orthonormalizeFrame(frame, np.ones(4))


class TestAnchorNegativeControls(unittest.TestCase):
    def test_unanchored_band_off_the_faces_fails(self) -> None:
        # An abstract rank-three band supported AWAY from every declared
        # anchoring face: the anchor read is exactly zero — the band FAILS
        # the anchor.  The undefined phase datum is NaN, never zero.
        rng = np.random.default_rng(71)
        w = np.ones(9)
        frame = np.zeros((9, 3), dtype=complex)
        block = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        frame[6:9, :] = block  # support only on edges {6, 7, 8}
        phi = ColorAnchor.orthonormalizeFrame(frame, w)
        anchor = ColorAnchor([OrientedTriangle([0, 1, 2], [1, 1, 1]),
                              OrientedTriangle([3, 4, 5], [1, 1, 1])])
        p = anchor.evaluate(phi, w)
        self.assertEqual(p.score, 0.0)
        self.assertEqual(list(p.terms), [0.0, 0.0])
        self.assertEqual(p.participation_ratio, 0.0)
        self.assertEqual(p.max_term, 0.0)
        for phase in p.det_phases:
            self.assertTrue(math.isnan(phase))
        self.assertTrue(math.isnan(p.phase_coherence))
        self.assertTrue(math.isnan(p.phase_dispersion))

    def test_degenerate_band_on_the_faces_fails(self) -> None:
        # Support ON the faces but with no alternating volume (two equal
        # frame rows on the triangle): |det A_tau|^2 collapses.
        rng = np.random.default_rng(72)
        w = np.full(4, 0.7)
        frame = rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3))
        frame[1, :] = frame[0, :]  # duplicate one-chain rows on the face
        phi = ColorAnchor.orthonormalizeFrame(frame, w)
        # Right-multiplication preserves the duplicated rows exactly.
        self.assertTrue(np.array_equal(phi[0, :], phi[1, :]))
        anchor = ColorAnchor([OrientedTriangle([0, 1, 2], [1, 1, 1])])
        p = anchor.evaluate(phi, w)
        self.assertLessEqual(p.score, 1e-28)

    def test_score_ordering_oracle_extended_unanchored(self) -> None:
        rng = np.random.default_rng(73)
        # Oracle.
        w3 = np.ones(3)
        oracle = ColorAnchor([OrientedTriangle([0, 1, 2], [1, 1, 1])])
        s_oracle = oracle.evaluate(orthonormal_band(rng, 3, w3), w3).score
        # Extended.
        w6 = np.ones(6)
        ext = ColorAnchor([OrientedTriangle([0, 1, 2], [1, 1, 1]),
                           OrientedTriangle([2, 3, 4], [1, 1, 1]),
                           OrientedTriangle([0, 4, 5], [1, 1, 1])])
        s_ext = ext.evaluate(orthonormal_band(rng, 6, w6), w6).score
        # Unanchored.
        w9 = np.ones(9)
        frame = np.zeros((9, 3), dtype=complex)
        frame[6:9, :] = np.eye(3)
        un = ColorAnchor([OrientedTriangle([0, 1, 2], [1, 1, 1])])
        s_un = un.evaluate(ColorAnchor.orthonormalizeFrame(frame, w9),
                           w9).score
        self.assertLessEqual(abs(s_oracle - 1.0), 1e-13)
        self.assertGreater(s_oracle, s_ext)
        self.assertGreater(s_ext, s_un)
        self.assertEqual(s_un, 0.0)

    def test_unnormalized_frame_rejected(self) -> None:
        # The calibration identity is undefined outside the
        # |W|-orthonormal domain: a raw frame must be rejected, not scored.
        rng = np.random.default_rng(74)
        frame = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        anchor = ColorAnchor([OrientedTriangle([0, 1, 2], [1, 1, 1])])
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            anchor.evaluate(frame, np.ones(3))

    def test_empty_atlas_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ColorAnchor([])

    def test_bad_triangles_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ColorAnchor([OrientedTriangle([0, 0, 1], [1, 1, 1])])
        with self.assertRaises(ValueError):
            ColorAnchor([OrientedTriangle([0, 1, 2], [1, 2, 1])])

    def test_shape_and_range_errors(self) -> None:
        anchor = ColorAnchor([OrientedTriangle([0, 1, 5], [1, 1, 1])])
        phi = ColorAnchor.orthonormalizeFrame(
            np.eye(3, dtype=complex), np.ones(3))
        with self.assertRaises(ValueError):  # edge 5 out of range
            anchor.evaluate(phi, np.ones(3))
        anchor2 = ColorAnchor([OrientedTriangle([0, 1, 2], [1, 1, 1])])
        with self.assertRaises(ValueError):  # weights length mismatch
            anchor2.evaluate(phi, np.ones(4))


class TestAnchorDeclaredWeighting(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(81)
        w = np.ones(6)
        phi = orthonormal_band(rng, 6, w)
        tris = [OrientedTriangle([0, 1, 2], [1, 1, 1]),
                OrientedTriangle([3, 4, 5], [1, 1, 1])]
        return phi, w, tris

    def test_post_hoc_weight_selection_rejected(self) -> None:
        phi, w, tris = self.fixture()
        anchor = ColorAnchor(tris)
        self.assertFalse(anchor.sealed())
        anchor.evaluate(phi, w)
        self.assertTrue(anchor.sealed())
        with self.assertRaisesRegex(RuntimeError, "post-hoc"):
            anchor.declareWeights([1.0, 0.0])

    def test_failed_evaluate_still_seals(self) -> None:
        # Even a REJECTED read has examined the data: the weighting seals.
        phi, w, tris = self.fixture()
        anchor = ColorAnchor(tris)
        with self.assertRaises(ValueError):
            anchor.evaluate(np.asarray(phi) * 2.0, w)  # not orthonormal
        self.assertTrue(anchor.sealed())
        with self.assertRaisesRegex(RuntimeError, "post-hoc"):
            anchor.declareWeights([1.0, 0.0])

    def test_declaration_before_data_is_allowed(self) -> None:
        phi, w, tris = self.fixture()
        anchor = ColorAnchor(tris)
        self.assertEqual(anchor.weightingId(), "uniform")
        self.assertEqual(anchor.weights(), [0.5, 0.5])
        anchor.declareWeights([0.75, 0.25])
        self.assertEqual(anchor.weightingId(), "declared")
        p = anchor.evaluate(phi, w)
        self.assertEqual(p.weighting_id, "declared")
        self.assertEqual(p.weights, [0.75, 0.25])
        self.assertLessEqual(
            abs(p.score - (0.75 * p.terms[0] + 0.25 * p.terms[1])), 1e-15)

    def test_non_convex_weightings_rejected(self) -> None:
        _, _, tris = self.fixture()
        with self.assertRaises(ValueError):
            ColorAnchor(tris, [0.5, 0.6])  # sum != 1
        with self.assertRaises(ValueError):
            ColorAnchor(tris, [1.5, -0.5])  # negative
        with self.assertRaises(ValueError):
            ColorAnchor(tris, [1.0])  # wrong length
        anchor = ColorAnchor(tris)
        with self.assertRaises(ValueError):
            anchor.declareWeights([0.2, 0.2])

    def test_uniform_weighting_is_the_default_declaration(self) -> None:
        phi, w, tris = self.fixture()
        anchor = ColorAnchor(tris)
        p = anchor.evaluate(phi, w)
        self.assertEqual(p.weighting_id, "uniform")
        self.assertEqual(p.weights, [0.5, 0.5])
        self.assertLessEqual(
            abs(p.score - 0.5 * (p.terms[0] + p.terms[1])), 1e-15)


class TestAnchorInvariances(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(91)
        n_edges = 6
        w = rng.uniform(0.5, 2.0, size=n_edges)
        phi = orthonormal_band(rng, n_edges, w)
        tri_spec = [((0, 1, 2), (1, 1, 1)), ((2, 3, 4), (1, -1, 1)),
                    ((0, 3, 5), (-1, 1, 1))]
        tris = [OrientedTriangle(list(e), list(s)) for e, s in tri_spec]
        return rng, np.asarray(phi), w, tris

    def profile(self, phi, w, tris, weights=None):
        anchor = ColorAnchor(tris) if weights is None else ColorAnchor(
            tris, weights)
        return anchor.evaluate(phi, w)

    def test_in_band_su3_frame_change_invariant(self) -> None:
        rng, phi, w, tris = self.fixture()
        p0 = self.profile(phi, w, tris)
        for _ in range(5):
            g = random_su3(rng)
            p1 = self.profile(phi @ g, w, tris)
            self.assertLessEqual(abs(p1.score - p0.score), 1e-12)
            self.assertLessEqual(
                np.max(np.abs(np.array(p1.terms) - np.array(p0.terms))),
                1e-12)
            self.assertLessEqual(
                abs(p1.participation_ratio - p0.participation_ratio), 1e-9)
            self.assertLessEqual(
                abs(p1.phase_coherence - p0.phase_coherence), 1e-9)
            # det g = 1: every determinant PHASE is itself invariant.
            for a, b in zip(p1.det_phases, p0.det_phases):
                self.assertLessEqual(
                    abs(np.exp(1j * a) - np.exp(1j * b)), 1e-9)

    def test_full_u3_change_shifts_all_phases_by_det(self) -> None:
        rng, phi, w, tris = self.fixture()
        p0 = self.profile(phi, w, tris)
        theta = 0.813
        g = np.exp(1j * theta / 3.0) * random_su3(rng)  # det g = e^{i theta}
        p1 = self.profile(phi @ g, w, tris)
        self.assertLessEqual(abs(p1.score - p0.score), 1e-12)
        for a, b in zip(p1.det_phases, p0.det_phases):
            self.assertLessEqual(
                abs(np.exp(1j * (a - b - theta)) - 1.0), 1e-9)
        self.assertLessEqual(abs(p1.phase_coherence - p0.phase_coherence),
                             1e-9)

    def test_oriented_edge_relabeling_exact_invariance(self) -> None:
        _, phi, w, tris = self.fixture()
        p0 = self.profile(phi, w, tris)
        perm = [3, 5, 0, 1, 4, 2]  # old edge e -> new row perm[e]
        n = len(perm)
        phi_p = np.zeros_like(phi)
        w_p = np.zeros_like(w)
        for e in range(n):
            phi_p[perm[e], :] = phi[e, :]
            w_p[perm[e]] = w[e]
        tris_p = [OrientedTriangle([perm[e] for e in t.edges],
                                   list(t.signs)) for t in tris]
        p1 = self.profile(phi_p, w_p, tris_p)
        # Pure reindexing: the same numbers flow through the same
        # operations — bitwise equality, not just tolerance.
        self.assertEqual(p1.score, p0.score)
        self.assertEqual(list(p1.terms), list(p0.terms))
        self.assertEqual(list(p1.det_phases), list(p0.det_phases))
        self.assertEqual(p1.phase_coherence, p0.phase_coherence)
        self.assertEqual(p1.krein_signatures, p0.krein_signatures)

    def test_stored_orientation_reversal_exact_invariance(self) -> None:
        # Reversing a stored edge orientation negates its frame row and
        # flips the incidence sign in every touching triangle descriptor:
        # the effective oriented boundary is unchanged.
        _, phi, w, tris = self.fixture()
        p0 = self.profile(phi, w, tris)
        for flip_edge in range(phi.shape[0]):
            phi_f = phi.copy()
            phi_f[flip_edge, :] = -phi_f[flip_edge, :]
            tris_f = []
            for t in tris:
                signs = list(t.signs)
                for k, e in enumerate(t.edges):
                    if e == flip_edge:
                        signs[k] = -signs[k]
                tris_f.append(OrientedTriangle(list(t.edges), signs))
            p1 = self.profile(phi_f, w, tris_f)
            self.assertEqual(p1.score, p0.score)
            self.assertEqual(list(p1.terms), list(p0.terms))
            self.assertEqual(list(p1.det_phases), list(p0.det_phases))

    def test_cyclic_triangle_rotation_is_even(self) -> None:
        # The orientation fixes the boundary ordering up to a CYCLIC (even)
        # permutation: det A_tau itself is invariant.
        _, phi, w, tris = self.fixture()
        t = tris[0]
        base = ColorAnchor.anchorMatrix(phi, w, t)
        rot = OrientedTriangle([t.edges[1], t.edges[2], t.edges[0]],
                               [t.signs[1], t.signs[2], t.signs[0]])
        rotated = ColorAnchor.anchorMatrix(phi, w, rot)
        d0, d1 = np.linalg.det(base), np.linalg.det(rotated)
        self.assertLessEqual(abs(d1 - d0), 1e-14 * max(1.0, abs(d0)))

    def test_odd_permutation_flips_the_determinant_only(self) -> None:
        # An odd reordering is the OPPOSITE orientation: det negates
        # (phase shifts by pi), |det|^2 unchanged.
        _, phi, w, tris = self.fixture()
        t = tris[0]
        base = ColorAnchor.anchorMatrix(phi, w, t)
        swap = OrientedTriangle([t.edges[1], t.edges[0], t.edges[2]],
                                [t.signs[1], t.signs[0], t.signs[2]])
        swapped = ColorAnchor.anchorMatrix(phi, w, swap)
        d0, d1 = np.linalg.det(base), np.linalg.det(swapped)
        self.assertLessEqual(abs(d1 + d0), 1e-14 * max(1.0, abs(d0)))
        self.assertLessEqual(abs(abs(d1) ** 2 - abs(d0) ** 2),
                             1e-13 * max(1.0, abs(d0) ** 2))


class TestAnchorSignedAndMatrixWeights(unittest.TestCase):
    def test_signed_sector_krein_reported_separately(self) -> None:
        # One timelike-signed edge weight: the score still restricts with
        # |W_tau|^{1/2} (identical to the |w| run), and the restricted
        # block's Krein signature is reported separately per triangle.
        rng = np.random.default_rng(101)
        w_signed = np.array([1.5, -0.8, 1.1, 0.9, 1.3, 0.6])
        phi = orthonormal_band(rng, 6, w_signed)
        tris = [OrientedTriangle([0, 1, 2], [1, 1, 1]),
                OrientedTriangle([3, 4, 5], [1, 1, 1])]
        p_signed = ColorAnchor(tris).evaluate(phi, w_signed)
        p_abs = ColorAnchor(tris).evaluate(phi, np.abs(w_signed))
        self.assertEqual(p_signed.score, p_abs.score)
        self.assertEqual(list(p_signed.terms), list(p_abs.terms))
        self.assertFalse(p_signed.positive_regime)
        self.assertTrue(p_abs.positive_regime)
        self.assertEqual(p_signed.krein_signatures, [[2, 0, 1], [3, 0, 0]])
        self.assertEqual(p_abs.krein_signatures, [[3, 0, 0], [3, 0, 0]])
        cob = tessera.cobordism
        self.assertEqual(p_signed.certificate.regime,
                         cob.CertificateRegime.HermitianIndefinite)
        self.assertEqual(p_abs.certificate.regime,
                         cob.CertificateRegime.PositiveSemidefinite)
        self.assertTrue(p_signed.certificate.holds())

    def test_zero_weight_reports_zero_mode(self) -> None:
        rng = np.random.default_rng(102)
        w = np.array([1.0, 0.0, 1.0, 1.0])
        frame = np.zeros((4, 3), dtype=complex)
        frame[[0, 2, 3], :] = rng.normal(size=(3, 3)) + 1j * rng.normal(
            size=(3, 3))
        phi = ColorAnchor.orthonormalizeFrame(frame, w)
        p = ColorAnchor([OrientedTriangle([0, 1, 2], [1, 1, 1])]).evaluate(
            phi, w)
        self.assertEqual(p.krein_signatures, [[2, 1, 0]])
        self.assertFalse(p.positive_regime)

    def test_matrix_weight_diagonal_agrees_with_vector_path(self) -> None:
        rng = np.random.default_rng(103)
        w = rng.uniform(0.5, 2.0, size=5)
        phi = orthonormal_band(rng, 5, w)
        tris = [OrientedTriangle([0, 1, 2], [1, -1, 1]),
                OrientedTriangle([1, 3, 4], [-1, 1, 1])]
        p_vec = ColorAnchor(tris).evaluate(phi, w)
        p_mat = ColorAnchor(tris).evaluateMatrix(
            phi, np.diag(w).astype(complex))
        self.assertLessEqual(abs(p_vec.score - p_mat.score), 1e-12)
        self.assertLessEqual(
            np.max(np.abs(np.array(p_vec.terms) - np.array(p_mat.terms))),
            1e-12)
        self.assertEqual(p_vec.krein_signatures, p_mat.krein_signatures)
        # Grades name the claim class honestly: closed-form structure-exact
        # on the diagonal path, certified-numerical on the eigen-modulus
        # matrix path.
        cob = tessera.cobordism
        self.assertEqual(p_vec.certificate.grade,
                         cob.CertificateGrade.StructureExact)
        self.assertEqual(p_mat.certificate.grade,
                         cob.CertificateGrade.CertifiedNumerical)
        self.assertTrue(p_mat.certificate.holds())

    def test_matrix_weight_requires_hermitian(self) -> None:
        rng = np.random.default_rng(104)
        phi = orthonormal_band(rng, 4, np.ones(4))
        bad = np.eye(4, dtype=complex)
        bad[0, 1] = 1.0  # not Hermitian
        anchor = ColorAnchor([OrientedTriangle([0, 1, 2], [1, 1, 1])])
        with self.assertRaisesRegex(ValueError, "Hermitian"):
            anchor.evaluateMatrix(phi, bad)

    def test_coupled_matrix_weight_checked_not_assumed(self) -> None:
        # A coupled Hermitian weight: the profile is still an exact
        # evaluation of A_tau = |W_tau|^{1/2} R_tau Phi (cross-checked in
        # NumPy), and the <= 1 calibration is CHECKED via the reported
        # margin rather than assumed.
        rng = np.random.default_rng(105)
        n = 5
        base = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        weight = base @ base.conj().T + 0.5 * np.eye(n)  # Hermitian PD
        frame = rng.normal(size=(n, 3)) + 1j * rng.normal(size=(n, 3))
        phi = ColorAnchor.orthonormalizeFrameMatrix(frame, weight)
        tri_spec = [((0, 1, 2), (1, 1, 1)), ((2, 3, 4), (1, -1, 1))]
        tris = [OrientedTriangle(list(e), list(s)) for e, s in tri_spec]
        p = ColorAnchor(tris).evaluateMatrix(phi, weight)

        # Independent NumPy reference for the matrix path.
        terms_ref = []
        for edges, signs in tri_spec:
            s = np.diag(signs).astype(complex)
            block = s @ weight[np.ix_(list(edges), list(edges))] @ s
            lam, u = np.linalg.eigh(block)
            sqrt_mod = u @ np.diag(np.sqrt(np.abs(lam))) @ u.conj().T
            rows = np.array(
                [sg * phi[e, :] for e, sg in zip(edges, signs)])
            terms_ref.append(abs(np.linalg.det(sqrt_mod @ rows)) ** 2)
        self.assertLessEqual(
            np.max(np.abs(np.array(p.terms) - np.array(terms_ref))), 1e-11)
        self.assertLessEqual(
            abs(p.score - 0.5 * float(np.sum(terms_ref))), 1e-11)
        self.assertTrue(np.isfinite(p.calibration_margin))
        self.assertTrue(p.positive_regime)
        # The frame was |W|-orthonormalized, so the domain certificate holds.
        self.assertLessEqual(p.frame_gram_residual, 1e-9)


class TestConstantAlgebraSelfCheck(unittest.TestCase):
    def test_verify_constant_algebra_at_round_off(self) -> None:
        # The startup-check contract (debug builds run this automatically;
        # every build can call it): the WHOLE constant algebra re-derives
        # within double round-off.
        self.assertLessEqual(ColorFiber.verifyConstantAlgebra(), 1e-12)

    def test_constant_algebra_certificate(self) -> None:
        # The same claim in the shared #764 vocabulary: AlgebraicallyExact,
        # holds, with the measured residual and the startup tolerance; the
        # re-derivation is deterministic so the residual matches the raw
        # call bitwise.
        cob = tessera.cobordism
        cert = ColorFiber.constantAlgebraCertificate()
        self.assertEqual(cert.grade, cob.CertificateGrade.AlgebraicallyExact)
        self.assertEqual(cert.domain, cob.CertificateDomain.Static)
        self.assertTrue(cert.holds())
        self.assertEqual(cert.residual, ColorFiber.verifyConstantAlgebra())
        self.assertEqual(cert.tolerance, 1e-12)
        self.assertTrue(math.isnan(cert.conditioning))


if __name__ == "__main__":
    unittest.main()
