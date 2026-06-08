# MIT License
# Copyright (c) 2025 Andrew Kelleher
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Which standard quantum gates can the cobordism theory support (#198).

The realizable Dijkgraaf–Witten ``T²→T²`` map image is, on ``main``, the
invertible group ``S₃`` (six holonomy-class permutations from
``Cobordism.twistedCylinder``, #193/#194) together with the non-invertible
projector ``|st⟩⟨st|`` (from ``Cobordism.disjointUnion`` of two solid tori,
#197) — a 10-dimensional subalgebra of ``End(ℂ⁴)``. This test answers, through
that *genuine machinery* (no hardcoded DW matrices), exactly which standard
quantum gates equal a buildable DW map ``Z(W)``.

The two holonomy qubits are the ``ℤ₂`` flux around the two cycles of the torus,
so ``Z(T²)=ℂ⁴=ℂ²⊗ℂ²`` with the vacuum ``|00⟩`` = the trivial holonomy class
(index 0) and the three non-trivial classes ``{[a],[b],[a+b]}={1,2,3}``.

**Verdict.** A gate ``G`` is *supported* iff it equals a realizable ``Z(W)``.
For the unitary gates this is membership in the six ``S₃`` maps, which #194/#195
proved equals the clean separator

    realizable-unitary  ⟺  G is a 0/1 permutation matrix that fixes index 0.

Every gate's machinery verdict (membership in the realizable set) is
cross-checked against this separator; the two must agree on every gate, else
the realizable-set construction has a bug. Two subtleties the table makes
explicit:

  * the **non-invertible sector adds no gates** — gates are unitary, the
    realizable non-invertible maps are rank-1 projectors, and no unitary equals
    a projector; the supported set is identical with or without it;
  * the criterion is "permutation **fixing the vacuum**," not merely
    "permutation": ``X``, ``Y`` and ``X⊗X`` are permutations (``X⊗X`` a genuine
    0/1 one) yet **move** ``|00⟩``, so none of them is supported.

Supported, end to end: ``I⊗I`` (the ``S₃`` identity), ``SWAP`` (the
transposition ``(1 2)``), both ``CNOT`` orientations (``(2 3)`` and ``(1 3)``),
and ``DCX`` (the 3-cycle ``(1 3 2)``). Everything with a phase
(``Z, S, T, CZ, CS, CY, iSWAP``), every superposition-maker
(``H, √X, √SWAP, CH, ECR, B, DFT₄``, the Bell entangler) and every
vacuum-mover (``X, Y, X⊗X``) is not.
"""

import itertools
import unittest

import numpy as np

import tessera

cobordism = tessera.cobordism
DijkgraafWitten = cobordism.DijkgraafWitten
Cocycle = cobordism.Cocycle
Cobordism = cobordism.Cobordism

DIM = 4
TOL = 1e-7


# --------------------------------------------------------------------------- #
# Fixtures — the established cobordism-test constructions (#193/#194/#197).
# --------------------------------------------------------------------------- #
def _build(topology):
    signature = tessera.Signature(topology.dimension(), tessera.Lorentzian)
    metric = tessera.Metric(True, signature)
    spacetime = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                                  tessera.PREFERRED, topology)
    spacetime.build()
    return spacetime


def _circle():
    # S¹ = ∂Δ², the minimal triangle-boundary circle (3 vertices).
    return tessera.SimplexBoundarySphere(1)


def _product_torus():
    # T² = S¹ × S¹, the 9-vertex symmetric product torus; the coordinate swap
    # φ(u, v) = (v, u) is a simplicial automorphism (productId = 3u + v).
    return _build(tessera.SimplicialProduct(_circle(), _circle()))


def _solid_torus():
    # ST = S¹ × D²; ∂ST = T² (one boundary component, the longitude survives).
    return _build(tessera.SimplicialProduct(_circle(), tessera.SolidSimplex(2)))


# φ(u, v) = (v, u) on the 3×3 product torus: the order-2 swap (the modular S).
_SWAP = [v * 3 + u for u in range(3) for v in range(3)]

# The minimal 7-vertex (Möbius) torus and its order-3 multiplier i ↦ 2i (mod 7).
_SEVEN_VERTEX_TRIANGLES = sorted({
    tuple(sorted(((i) % 7, (i + step) % 7, (i + 3) % 7)))
    for i in range(7) for step in (1, 2)})


def _seven_vertex_torus():
    signature = tessera.Signature(2, tessera.Lorentzian)
    metric = tessera.Metric(True, signature)
    spacetime = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                                  tessera.PREFERRED, None)
    vertices = [spacetime.createVertex(i) for i in range(7)]
    for triangle in _SEVEN_VERTEX_TRIANGLES:
        spacetime.createSimplex([vertices[i] for i in triangle])
    return spacetime


def _multiplier(m, n):
    return [(m * i) % n for i in range(n)]


def _dw_map(spacetime, cocycle=Cocycle.Trivial):
    return np.asarray(DijkgraafWitten(spacetime, cocycle).map())


# --------------------------------------------------------------------------- #
# The realizable set, rebuilt through the genuine machinery (no hardcoding).
# --------------------------------------------------------------------------- #
def _s3_group():
    """The six S₃ DW maps, from twistedCylinder, closed under multiplication."""
    swap = _dw_map(Cobordism.twistedCylinder(_product_torus(), _SWAP)).real
    rotation = _dw_map(
        Cobordism.twistedCylinder(_seven_vertex_torus(),
                                  _multiplier(2, 7))).real
    identity = np.eye(DIM)
    group, frontier = [identity], [identity]

    def known(matrix):
        return any(np.allclose(matrix, seen, atol=1e-9) for seen in group)

    while frontier:
        element = frontier.pop()
        for generator in (swap, rotation):
            for product in (generator @ element, element @ generator):
                if not known(product):
                    group.append(product)
                    frontier.append(product)
    return group


def _projector():
    """The non-invertible cap-and-create map |st⟩⟨st| from disjointUnion."""
    return _dw_map(Cobordism.disjointUnion(_solid_torus(), _solid_torus())).real


# --------------------------------------------------------------------------- #
# The two verdict predicates: the machinery, and the proven separator.
# --------------------------------------------------------------------------- #
def _machinery_supported(gate, realizable):
    """Does the gate equal a buildable DW map Z(W)? (membership in the set)."""
    return any(np.allclose(gate, realized, atol=TOL) for realized in realizable)


def _separator_supported(gate):
    """The #194/#195 separator: a 0/1 permutation matrix that fixes index 0."""
    matrix = np.asarray(gate, dtype=complex)
    if not np.allclose(matrix.imag, 0.0, atol=1e-9):
        return False                                   # carries a phase
    real = matrix.real
    if not np.all(np.isclose(real, 0.0) | np.isclose(real, 1.0)):
        return False                                   # not 0/1
    if not (np.allclose(real.sum(axis=0), 1.0)
            and np.allclose(real.sum(axis=1), 1.0)):
        return False                                   # not a permutation
    return bool(np.isclose(real[0, 0], 1.0))           # fixes the vacuum [0]


# --------------------------------------------------------------------------- #
# Structural lenses (used to certify each "reason" the table reports).
# --------------------------------------------------------------------------- #
def _is_monomial(gate):
    """One non-zero entry per row and per column (a permutation up to phases)."""
    nonzero = ~np.isclose(np.asarray(gate, dtype=complex), 0.0)
    return (np.all(nonzero.sum(axis=0) == 1)
            and np.all(nonzero.sum(axis=1) == 1))


def _is_zero_one(gate):
    matrix = np.asarray(gate, dtype=complex)
    return (np.allclose(matrix.imag, 0.0, atol=1e-9)
            and np.all(np.isclose(matrix.real, 0.0)
                       | np.isclose(matrix.real, 1.0)))


def _fixes_vacuum(gate):
    """G|00⟩ = (phase)|00⟩  ⟺  |⟨00|G|00⟩| = 1."""
    return bool(np.isclose(abs(np.asarray(gate, dtype=complex)[0, 0]), 1.0))


def _permutation_tuple(gate):
    rounded = np.round(np.asarray(gate).real).astype(int)
    return tuple(int(np.argmax(rounded[r])) for r in range(DIM))


def _cycle_name(perm):
    """The cycle structure of a vacuum-fixing permutation, on {1, 2, 3}."""
    seen, cycles = set(), []
    for start in (1, 2, 3):
        if start in seen:
            continue
        cycle, current = [start], perm[start]
        seen.add(start)
        while current != start:
            cycle.append(current)
            seen.add(current)
            current = perm[current]
        if len(cycle) > 1:
            cycles.append("(" + " ".join(map(str, cycle)) + ")")
    return "".join(cycles) if cycles else "identity"


# --------------------------------------------------------------------------- #
# A comprehensive standard gate set as 4×4 matrices on ℂ⁴ = ℂ²⊗ℂ².
# --------------------------------------------------------------------------- #
_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_S = np.array([[1, 0], [0, 1j]], dtype=complex)
_SDG = _S.conj().T
_T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
_SX = np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex) / 2  # √X


def _kron(a, b):
    return np.kron(a, b)


def _expm_hermitian(theta, hamiltonian):
    """exp(i·θ·H) for Hermitian H, via its (genuine) spectral decomposition."""
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    return (eigenvectors @ np.diag(np.exp(1j * theta * eigenvalues))
            @ eigenvectors.conj().T)


# Two-qubit gates (index = 2·q0 + q1, so the vacuum |00⟩ is index 0).
_CNOT_Q0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0],          # control = q0
                     [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
_CNOT_Q1 = np.array([[1, 0, 0, 0], [0, 0, 0, 1],          # control = q1
                     [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex)
_CZ = np.diag([1, 1, 1, -1]).astype(complex)
_SWAP_GATE = np.array([[1, 0, 0, 0], [0, 0, 1, 0],
                       [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
_ISWAP = np.array([[1, 0, 0, 0], [0, 0, 1j, 0],
                   [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex)
_SQRT_SWAP = np.array([[1, 0, 0, 0],
                       [0, (1 + 1j) / 2, (1 - 1j) / 2, 0],
                       [0, (1 - 1j) / 2, (1 + 1j) / 2, 0],
                       [0, 0, 0, 1]], dtype=complex)
_DCX = np.array([[1, 0, 0, 0], [0, 0, 0, 1],              # two back-to-back CNOTs
                 [0, 1, 0, 0], [0, 0, 1, 0]], dtype=complex)
_CY = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                [0, 0, 0, -1j], [0, 0, 1j, 0]], dtype=complex)
_CH = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                [0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
                [0, 0, 1 / np.sqrt(2), -1 / np.sqrt(2)]], dtype=complex)
_CS = np.diag([1, 1, 1, 1j]).astype(complex)
_ECR = (_kron(_I2, _X) - _kron(_X, _Y)) / np.sqrt(2)     # echoed cross-resonance
_B = _expm_hermitian(np.pi / 8, 2 * _kron(_X, _X) + _kron(_Y, _Y))  # Berkeley B
_DFT4 = np.array([[(1j) ** (j * k) for k in range(DIM)]
                  for j in range(DIM)], dtype=complex) / 2
_BELL = _kron(_H, _I2) @ _CNOT_Q0                         # a Bell-prep entangler

# Reason buckets — asserted structurally below so the table never lies.
_MOVES_VACUUM = "moves the vacuum |00⟩"
_PHASE = "phase (monomial but not 0/1)"
_SUPERPOSITION = "superposition (not a permutation)"

# (name, matrix, expected_supported, reason-bucket). For supported gates the
# reason is filled in from the realized S₃ cycle at run time.
GATES = [
    ("I⊗I", np.eye(DIM, dtype=complex), True, "S₃ element"),
    # single-qubit embedded G⊗I and I⊗G
    ("X⊗I", _kron(_X, _I2), False, _MOVES_VACUUM),
    ("I⊗X", _kron(_I2, _X), False, _MOVES_VACUUM),
    ("Y⊗I", _kron(_Y, _I2), False, _MOVES_VACUUM),
    ("I⊗Y", _kron(_I2, _Y), False, _MOVES_VACUUM),
    ("Z⊗I", _kron(_Z, _I2), False, _PHASE),
    ("I⊗Z", _kron(_I2, _Z), False, _PHASE),
    ("H⊗I", _kron(_H, _I2), False, _SUPERPOSITION),
    ("I⊗H", _kron(_I2, _H), False, _SUPERPOSITION),
    ("S⊗I", _kron(_S, _I2), False, _PHASE),
    ("I⊗S", _kron(_I2, _S), False, _PHASE),
    ("S†⊗I", _kron(_SDG, _I2), False, _PHASE),
    ("I⊗S†", _kron(_I2, _SDG), False, _PHASE),
    ("T⊗I", _kron(_T, _I2), False, _PHASE),
    ("I⊗T", _kron(_I2, _T), False, _PHASE),
    ("√X⊗I", _kron(_SX, _I2), False, _SUPERPOSITION),
    ("I⊗√X", _kron(_I2, _SX), False, _SUPERPOSITION),
    # the permutation that nevertheless moves the vacuum
    ("X⊗X", _kron(_X, _X), False, _MOVES_VACUUM),
    # two-qubit gates
    ("CNOT(c=q0)", _CNOT_Q0, True, "S₃ element"),
    ("CNOT(c=q1)", _CNOT_Q1, True, "S₃ element"),
    ("CZ", _CZ, False, _PHASE),
    ("SWAP", _SWAP_GATE, True, "S₃ element"),
    ("iSWAP", _ISWAP, False, _PHASE),
    ("√SWAP", _SQRT_SWAP, False, _SUPERPOSITION),
    ("DCX", _DCX, True, "S₃ element"),
    ("CY", _CY, False, _PHASE),
    ("CH", _CH, False, _SUPERPOSITION),
    ("CS", _CS, False, _PHASE),
    ("ECR", _ECR, False, _SUPERPOSITION),
    ("B (Berkeley)", _B, False, _SUPERPOSITION),
    ("DFT₄", _DFT4, False, _SUPERPOSITION),
    ("Bell (H⊗I)·CNOT", _BELL, False, _SUPERPOSITION),
]

# The five gates we expect to be supported, with their S₃ cycle structure.
_EXPECTED_SUPPORTED = {
    "I⊗I": "identity",
    "SWAP": "(1 2)",
    "CNOT(c=q0)": "(2 3)",
    "CNOT(c=q1)": "(1 3)",
    "DCX": "(1 3 2)",
}


# Built once (the DW state sums are the same for every test).
_S3 = _s3_group()
_PROJECTOR = _projector()
_REALIZABLE = _S3 + [_PROJECTOR]


# --------------------------------------------------------------------------- #
# The realizable set itself is the merged machinery (a sanity gate).
# --------------------------------------------------------------------------- #
class TestRealizableSetThroughMachinery(unittest.TestCase):

    def test_s3_group_has_six_permutations_fixing_the_vacuum(self):
        self.assertEqual(len(_S3), 6)
        for element in _S3:
            self.assertTrue(_separator_supported(element))   # 0/1 perm fixing 0
            np.testing.assert_allclose(element.conj().T @ element,
                                       np.eye(DIM), atol=1e-9)

    def test_s3_group_is_all_six_vacuum_fixing_permutations(self):
        fixers = [np.asarray([[1.0 if perm[r] == c else 0.0 for c in range(DIM)]
                              for r in range(DIM)])
                  for perm in itertools.permutations(range(DIM)) if perm[0] == 0]
        self.assertEqual(len(fixers), 6)
        for fixer in fixers:
            self.assertTrue(any(np.allclose(fixer, g, atol=1e-9) for g in _S3))

    def test_projector_is_rank_one_non_invertible_and_not_unitary(self):
        self.assertEqual(np.linalg.matrix_rank(_PROJECTOR, tol=1e-9), 1)
        self.assertAlmostEqual(float(np.linalg.det(_PROJECTOR)), 0.0, places=9)
        self.assertFalse(np.allclose(_PROJECTOR.conj().T @ _PROJECTOR,
                                     np.eye(DIM), atol=1e-6))

    def test_projector_is_the_solid_torus_outer_product(self):
        # |st⟩ = [1, 1, 0, 0] (trivial + surviving longitude class).
        st = np.array([1.0, 1.0, 0.0, 0.0])
        np.testing.assert_allclose(_PROJECTOR, np.outer(st, st), atol=1e-9)


# --------------------------------------------------------------------------- #
# The headline: per-gate support, machinery vs. separator.
# --------------------------------------------------------------------------- #
class TestGateSupport(unittest.TestCase):

    def test_every_gate_is_unitary(self):
        # The whole gate set is unitary, so no gate can equal a projector.
        for name, gate, _, _ in GATES:
            np.testing.assert_allclose(gate.conj().T @ gate, np.eye(DIM),
                                       atol=1e-9, err_msg=name)

    def test_machinery_and_separator_agree_on_every_gate(self):
        # The cross-check that would catch a realizable-set bug: the genuine
        # membership verdict must equal the proven separator on EVERY gate.
        disagreements = []
        for name, gate, _, _ in GATES:
            machinery = _machinery_supported(gate, _REALIZABLE)
            separator = _separator_supported(gate)
            if machinery != separator:
                disagreements.append((name, machinery, separator))
        self.assertEqual(disagreements, [],
                         f"machinery vs separator disagreed: {disagreements}")

    def test_each_gate_matches_its_expected_verdict(self):
        for name, gate, expected, _ in GATES:
            self.assertEqual(_machinery_supported(gate, _REALIZABLE), expected,
                             f"{name}: machinery verdict")
            self.assertEqual(_separator_supported(gate), expected,
                             f"{name}: separator verdict")

    def test_supported_gates_are_exactly_the_expected_five(self):
        supported = {name for name, gate, _, _ in GATES
                     if _machinery_supported(gate, _REALIZABLE)}
        self.assertEqual(supported, set(_EXPECTED_SUPPORTED))

    def test_supported_gates_land_on_their_expected_s3_cycle(self):
        for name, gate, expected, _ in GATES:
            if not expected:
                continue
            self.assertTrue(_separator_supported(gate), name)
            cycle = _cycle_name(_permutation_tuple(gate))
            self.assertEqual(cycle, _EXPECTED_SUPPORTED[name], name)
            # and it really is one of the six S₃ DW maps.
            self.assertTrue(any(np.allclose(gate, g, atol=TOL) for g in _S3),
                            name)

    def test_supported_set_has_the_three_transpositions_and_a_three_cycle(self):
        # SWAP, CNOT(c=q0), CNOT(c=q1) are the three S₃ transpositions; DCX is the
        # 3-cycle (1 3 2). The sixth S₃ element is the other 3-cycle (1 2 3) — not
        # a single named gate here, but reachable as a product of supported gates
        # (CNOT(c=q0)·SWAP) and as DCX's inverse, both of which land back in S₃.
        cycles = {_cycle_name(_permutation_tuple(gate))
                  for name, gate, expected, _ in GATES if expected}
        self.assertEqual(cycles, {"identity", "(1 2)", "(2 3)", "(1 3)",
                                  "(1 3 2)"})
        other_three_cycle = _CNOT_Q0.real @ _SWAP_GATE.real          # = (1 2 3)
        self.assertEqual(_cycle_name(_permutation_tuple(other_three_cycle)),
                         "(1 2 3)")
        np.testing.assert_allclose(other_three_cycle, _DCX.real.T, atol=1e-9)
        self.assertTrue(any(np.allclose(other_three_cycle, g, atol=TOL)
                            for g in _S3))


# --------------------------------------------------------------------------- #
# Each "reason" the table reports is structurally certified.
# --------------------------------------------------------------------------- #
class TestReasonsAreCertified(unittest.TestCase):

    def test_moves_vacuum_reasons(self):
        # X, Y, X⊗X: each MOVES |00⟩ (|⟨00|G|00⟩| < 1), so even the genuine 0/1
        # permutation X⊗X is not supported — the criterion is "fixes the vacuum,"
        # not merely "permutation."
        for name, gate, _, reason in GATES:
            if reason is not _MOVES_VACUUM:
                continue
            self.assertFalse(_fixes_vacuum(gate), name)
            self.assertFalse(_separator_supported(gate), name)
        self.assertTrue(_is_zero_one(_kron(_X, _X)))       # X⊗X really is 0/1
        self.assertFalse(_separator_supported(_kron(_X, _X)))

    def test_phase_reasons(self):
        # Z, S, S†, T, CZ, CS, CY, iSWAP: monomial (permutation-patterned) and
        # vacuum-fixing, but NOT 0/1 — a phase blocks them.
        for name, gate, _, reason in GATES:
            if reason is not _PHASE:
                continue
            self.assertTrue(_is_monomial(gate), name)
            self.assertFalse(_is_zero_one(gate), name)
            self.assertTrue(_fixes_vacuum(gate), name)

    def test_superposition_reasons(self):
        # H, √X, √SWAP, CH, ECR, B, DFT₄, Bell: not monomial — genuine mixing.
        for name, gate, _, reason in GATES:
            if reason is not _SUPERPOSITION:
                continue
            self.assertFalse(_is_monomial(gate), name)


# --------------------------------------------------------------------------- #
# The non-invertible sector adds zero gates.
# --------------------------------------------------------------------------- #
class TestNonInvertibleSectorAddsNoGates(unittest.TestCase):

    def test_no_gate_equals_the_projector_or_any_non_unitary_map(self):
        non_unitary = [m for m in _REALIZABLE
                       if not np.allclose(m.conj().T @ m, np.eye(DIM), atol=1e-6)]
        self.assertEqual(len(non_unitary), 1)             # just the projector
        for name, gate, _, _ in GATES:
            for map_ in non_unitary:
                self.assertFalse(np.allclose(gate, map_, atol=TOL), name)

    def test_supported_set_is_identical_with_and_without_the_projector(self):
        # Dropping the projector from the realizable set changes no verdict —
        # the non-invertible sector contributes nothing to gate support.
        for name, gate, _, _ in GATES:
            self.assertEqual(_machinery_supported(gate, _REALIZABLE),
                             _machinery_supported(gate, _S3), name)


# --------------------------------------------------------------------------- #
# The clean, citable gate-support table (printed; pytest -s to see it).
# --------------------------------------------------------------------------- #
def _support_table():
    lines = [f"{'gate':18}{'supported?':12}{'machinery=sep':14}reason",
             "-" * 72]
    for name, gate, _, reason in GATES:
        machinery = _machinery_supported(gate, _REALIZABLE)
        separator = _separator_supported(gate)
        if machinery:
            detail = f"S₃ {_cycle_name(_permutation_tuple(gate))}"
        else:
            detail = reason
        lines.append(f"{name:18}{('yes' if machinery else 'no'):12}"
                     f"{str(machinery == separator):14}{detail}")
    return "\n".join(lines)


class TestSupportTablePrints(unittest.TestCase):

    def test_print_support_table(self):
        table = _support_table()
        print("\nQuantum gate support on Z(T²) = ℂ⁴ (DW realizable image):\n")
        print(table)
        # exactly five supported rows.
        self.assertEqual(table.count("yes"), len(_EXPECTED_SUPPORTED))
        self.assertNotIn("False", table)                  # every row agrees


if __name__ == "__main__":
    unittest.main()
