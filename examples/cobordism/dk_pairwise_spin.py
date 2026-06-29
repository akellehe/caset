# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Pairwise connected-correlator readout of the composite proton spin J² (#514, part of #410).

The emergent proton's composite total spin distinguishes a proton (J²=¾) from a Δ (J²=15/4).
A product of independently-extracted per-hole spinors floors the readout at the mixture
(~9/4): the proton's ¾ is an entangled, mixed-symmetry quantity. This module reads the same
J² through a lighter, sharper lens (cartan_weyl_gluon.tex §5a / §7 experiment 1).

## The key fact: J² is a two-body operator

For three spin-½'s, S_i² = ¾·I, so

    J² = Σ_a (Σ_i S_a^{(i)})²  =  Σ_i S_i²  +  2 Σ_{i<j} S_i·S_j  =  9/4 + 2 Σ_{i<j} S_i·S_j .

No three-body term — so ⟨J²⟩ is *exactly* determined by the three pairwise reduced two-qubit
states ρ_ij (⟨S_i·S_j⟩ = Tr(ρ_ij · Σ_a S_a⊗S_a)), never the full joint state. That is what
makes this cheaper than a total-space operator (#477/#495): three 4×4 marginals.

## What the per-hole read discards: the connected correlator

    ⟨S_i·S_j⟩ = ⟨S_i⟩·⟨S_j⟩ + C_ij ,     C_ij := ⟨S_i·S_j⟩ − ⟨S_i⟩·⟨S_j⟩ .

A per-hole Bloch/product read keeps only ⟨S_i⟩·⟨S_j⟩ (sets C_ij = 0):

    J²_disconnected = 9/4 + 2 Σ_{i<j} ⟨S_i⟩·⟨S_j⟩          (the floor),
    J²              = J²_disconnected + 2 Σ_{i<j} C_ij      (the truth).

On the exact proton eigenstate the per-hole Bloch read gives precisely 9/4 and the connected
part carries the missing −3/2. The whole proton/Δ distinction lives in the three numbers C_ij.

Pure NumPy; the spin lives in (C²)³. (`_j2_direct` is the self-contained full-operator
reference: proton eigenstate → ¾, product |uud⟩ → 7/4, Δ |uuu⟩ → 15/4.) The emergent entry
points at the bottom lazily import `tessera`.
"""
import collections

import numpy as np
import scipy.linalg

# Clean spin-½ operators in C².
_PAULI = [np.array([[0, 1], [1, 0]], complex),
          np.array([[0, -1j], [1j, 0]]),
          np.array([[1, 0], [0, -1]], complex)]
_S = [p / 2 for p in _PAULI]
_I2 = np.eye(2, dtype=complex)

# The two-qubit spin-spin operator S_i·S_j = Σ_a S_a⊗S_a (4×4); symmetric in its two factors.
_SS = sum(np.kron(_S[a], _S[a]) for a in range(3))


def _normalize(psi):
    psi = np.asarray(psi, complex)
    n = np.linalg.norm(psi)
    return psi / n if n > 1e-30 else psi


# ----- reduced states (partial traces of a 3-qubit pure state) -----
def reduced_states(psi8):
    """The three pairwise (`ρ_ij`, 4×4) and three single (`ρ_i`, 2×2) reduced density matrices
    of a normalized three-qubit pure state `psi8` (ordering hole0 ⊗ hole1 ⊗ hole2). Returns
    `(pairs, singles)` with `pairs[(i,j)]` and `singles[i]`; every reduced state is unit trace."""
    t = _normalize(psi8).reshape(2, 2, 2)
    tc = t.conj()
    pairs = {
        (0, 1): np.einsum("abc,dec->abde", t, tc).reshape(4, 4),   # trace out hole 2
        (0, 2): np.einsum("abc,dbe->acde", t, tc).reshape(4, 4),   # trace out hole 1
        (1, 2): np.einsum("abc,aef->bcef", t, tc).reshape(4, 4),   # trace out hole 0
    }
    singles = {
        0: np.einsum("abc,dbc->ad", t, tc),
        1: np.einsum("abc,adc->bd", t, tc),
        2: np.einsum("abc,abd->cd", t, tc),
    }
    return pairs, singles


# ----- the correlators -----
def spin_correlator(rho_ij):
    """`⟨S_i·S_j⟩ = Tr(ρ_ij · Σ_a S_a⊗S_a)` from the pairwise reduced state `ρ_ij` (4×4)."""
    return float(np.trace(np.asarray(rho_ij, complex) @ _SS).real)


def bloch(rho_i):
    """The spin Bloch vector `⟨S⟩ = (⟨S_1⟩,⟨S_2⟩,⟨S_3⟩)` from a single reduced state `ρ_i`."""
    r = np.asarray(rho_i, complex)
    return np.array([float(np.trace(r @ _S[a]).real) for a in range(3)])


def connected_correlator(rho_ij, rho_i, rho_j):
    """The connected two-hole spin correlator `C_ij = ⟨S_i·S_j⟩ − ⟨S_i⟩·⟨S_j⟩`. Vanishes iff
    the pair is uncorrelated (a product); it is exactly what a per-hole Bloch read discards."""
    return spin_correlator(rho_ij) - float(bloch(rho_i) @ bloch(rho_j))


# ----- the J² readouts -----
def pairwise_j2(rho01, rho02, rho12):
    """Composite `J² = 9/4 + 2 Σ_{i<j} ⟨S_i·S_j⟩`, read from the three pairwise reduced states.
    Exact (no three-body term): equals the full-operator `J²` on any state."""
    return 2.25 + 2.0 * (spin_correlator(rho01) + spin_correlator(rho02)
                         + spin_correlator(rho12))


def pairwise_j2_from_state(psi8):
    """`pairwise_j2` applied to the three pairwise marginals of a 3-qubit pure state."""
    pairs, _ = reduced_states(psi8)
    return pairwise_j2(pairs[(0, 1)], pairs[(0, 2)], pairs[(1, 2)])


def disconnected_j2(psi8):
    """The per-hole Bloch / product floor `9/4 + 2 Σ_{i<j} ⟨S_i⟩·⟨S_j⟩` (every `C_ij` zeroed) —
    the value any per-hole product read is pinned to."""
    return j2_decomposition(psi8)["j2_disconnected"]


def j2_decomposition(psi8):
    """The full chamber-coordinate breakdown of `J²` for a 3-qubit pure state. Returns a dict:
    `j2` (= `j2_disconnected + j2_connected`), `j2_disconnected` (the per-hole Bloch floor),
    `j2_connected` (`2 Σ C_ij`, the discarded entanglement), `C_ij`, `spin_correlators`, and
    `bloch` (per-hole Bloch vectors)."""
    pairs, singles = reduced_states(psi8)
    blochs = {i: bloch(r) for i, r in singles.items()}
    corr = {ij: spin_correlator(r) for ij, r in pairs.items()}
    cij = {ij: corr[ij] - float(blochs[ij[0]] @ blochs[ij[1]]) for ij in pairs}
    j2_disc = 2.25 + 2.0 * sum(float(blochs[i] @ blochs[j]) for (i, j) in pairs)
    j2_conn = 2.0 * sum(cij.values())
    return {"j2": j2_disc + j2_conn, "j2_disconnected": j2_disc, "j2_connected": j2_conn,
            "C_ij": cij, "spin_correlators": corr, "bloch": blochs}


def _j2_direct(psi8):
    """Reference full-operator `J² = ⟨ψ| Σ_a (Σ_i S_a^{(i)})² |ψ⟩` in `(C²)³`, pure NumPy — the
    validated instrument (proton → ¾, product |uud⟩ → 7/4, Δ |uuu⟩ → 15/4). Exists so the test
    can show the pairwise (two-body) reformulation is *exact*, not an approximation."""
    st = _normalize(psi8)

    def si(a, i):
        ops = [_I2, _I2, _I2]
        ops[i] = _S[a]
        return np.kron(np.kron(ops[0], ops[1]), ops[2])

    sa = [sum(si(a, i) for i in range(3)) for a in range(3)]
    j2 = sum(sa[a] @ sa[a] for a in range(3))
    return float((st.conj() @ j2 @ st).real)


# =====================================================================================
# Joint two-hole extraction — reading a genuinely correlated rho_ij off the carried field,
# so C_ij can come out NONZERO (the product read of the floor forces C_ij = 0).
#
# Two obstructions shape what is possible (both demonstrated in the tests):
#
#   1. <J^2> is rotationally INVARIANT, so the floor-escape must add rotationally-invariant
#      correlation. An isotropic Heisenberg entangler exp(i*theta*S_i.S_j) is therefore
#      useless: S_i.S_j is a total-spin scalar, it commutes with J^2 and leaves <J^2> exactly
#      unchanged (`isotropic_heisenberg`). The chamber angle cannot be isotropic.
#
#   2. A CLASSICAL scalar field cannot entangle by a local bilinear read. Any field-weighted
#      joint amplitude  sum_{f,g} psi(f) psi*(g) u_f (x) v_g  over a hole pair's cells
#      FACTORIZES = (sum_f psi(f) u_f) (x) (sum_g psi*(g) v_g): a rank-1 product, C_ij = 0.
#      The entanglement is not in the classical carried cochain; it lives in treating the
#      b3 color register as a quantum Hilbert space whose phases lock to spin (Pauli;
#      cartan_weyl_gluon.tex section 3).
#
# The constructive read below honors both: a rotationally-COVARIANT Werner state whose
# correlation strength lambda and singlet/triplet character are SOURCED from the field's own
# gauge-invariant inter-hole data (the transported spinor overlap and its color phase),
# never a chamber angle put in by hand.
# =====================================================================================

# 2-qubit total-spin projectors (rotationally invariant: they commute with every U(x)U).
_SINGLET = np.array([0, 1, -1, 0], complex) / np.sqrt(2.0)
_P_SINGLET = np.outer(_SINGLET, _SINGLET.conj())
_P_TRIPLET = np.eye(4, dtype=complex) - _P_SINGLET


def isotropic_heisenberg(theta):
    """The isotropic entangler `exp(i*theta*S_i.S_j)` (4x4). Because `S_i.S_j` is a total-spin
    scalar it commutes with `J^2`, so this gate leaves `<J^2>` UNCHANGED — the precise reason
    the floor cannot be escaped by an isotropic coupling (obstruction 1 above)."""
    return scipy.linalg.expm(1j * float(theta) * _SS)


def correlated_pair(singlet_weight):
    """The rotationally-invariant correlated 2-qubit state `w*P_singlet + (1-w)*P_triplet/3`.
    Its `<S.S> = w*(-3/4) + (1-w)*(1/4)`: w=1 -> singlet (-3/4), w=0 -> triplet (+1/4), and
    w=3/4 -> -1/2, the proton's u-d pair value and the natural image of a 120-degree phase."""
    w = float(singlet_weight)
    return w * _P_SINGLET + (1.0 - w) * _P_TRIPLET / 3.0


def werner_pair(q_i, q_j, lam, singlet_weight):
    """A rotationally-covariant joint two-hole state `(1-lam)*|q_i q_j><q_i q_j| + lam*rho_corr`.
    `lam in [0,1]` is the field-sourced correlation strength (lam=0 => the product read,
    C_ij=0); the correlated part is `correlated_pair(singlet_weight)`. Covariant: conjugating
    both qubits by the same `U in SU(2)` conjugates the whole state by `U(x)U`, leaving `C_ij`
    invariant."""
    p = np.kron(np.asarray(q_i, complex), np.asarray(q_j, complex))
    lam = float(np.clip(lam, 0.0, 1.0))
    return (1.0 - lam) * np.outer(p, p.conj()) + lam * correlated_pair(singlet_weight)


def color_phase_correlation(s_i, s_j):
    """Field-sourced `(lam, singlet_weight)` from the transported Dirac-spinor overlap
    `z = <s_i|s_j>/(|s_i|*|s_j|)`: `lam = 1 - |z|^2` (the misalignment budget — parallel
    spinors carry no extra correlation) and `singlet_weight = (1 - cos(arg z))/2` (the
    inter-hole color phase; the register's 120 degrees gives w=3/4 => <S.S>_corr = -1/2). A
    readout of the field's own phases, with no chamber angle inserted by hand."""
    a = np.asarray(s_i, complex)
    b = np.asarray(s_j, complex)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-30 or nb < 1e-30:
        return 0.0, 0.5
    z = complex(a.conj() @ b) / (na * nb)
    return float(np.clip(1.0 - abs(z) ** 2, 0.0, 1.0)), float((1.0 - np.cos(np.angle(z))) / 2.0)


def _ptrace_second(rho4):
    """`rho_i = Tr_j rho_ij` of a 4x4 two-qubit state."""
    return np.einsum("abcb->ac", np.asarray(rho4, complex).reshape(2, 2, 2, 2))


def _ptrace_first(rho4):
    """`rho_j = Tr_i rho_ij` of a 4x4 two-qubit state."""
    return np.einsum("abad->bd", np.asarray(rho4, complex).reshape(2, 2, 2, 2))


def decomposition_from_pairs(pairs):
    """The chamber-coordinate decomposition (same keys as `j2_decomposition`) read from three
    given pairwise states `pairs[(i,j)]` — the way the readout is meant to consume them
    (cartan section 5a): each hole's `<S>` comes from its own marginals (averaged across the
    two pairs it appears in). NOTE: when the three marginals are mutually inconsistent (the
    rho_ij do not arise from one global 3-qubit state) `j2` can fall outside `[0, 15/4]` —
    itself a useful signal that the pairwise reads are not yet globally consistent."""
    corr = {ij: spin_correlator(r) for ij, r in pairs.items()}
    acc = collections.defaultdict(list)
    for (i, j), r in pairs.items():
        acc[i].append(bloch(_ptrace_second(r)))
        acc[j].append(bloch(_ptrace_first(r)))
    blochs = {h: np.mean(np.array(v), axis=0) for h, v in acc.items()}
    cij = {ij: corr[ij] - float(blochs[ij[0]] @ blochs[ij[1]]) for ij in pairs}
    j2_disc = 2.25 + 2.0 * sum(float(blochs[i] @ blochs[j]) for (i, j) in pairs)
    j2_conn = 2.0 * sum(cij.values())
    return {"j2": j2_disc + j2_conn, "j2_disconnected": j2_disc, "j2_connected": j2_conn,
            "C_ij": cij, "spin_correlators": corr, "bloch": blochs}


# ----- demo (clean hand-fed states; the emergent entry points are appended once wired) -----
_UP = np.array([1, 0], complex)
_DN = np.array([0, 1], complex)


def _kr(*a):
    out = a[0]
    for x in a[1:]:
        out = np.kron(out, x)
    return out


def _clean_states():
    """The hand-fed clean states: proton (J²=¾), product |uud⟩ (7/4), Δ |uuu⟩ (15/4)."""
    return {
        "proton  2|uud>-|udu>-|duu>": 2 * _kr(_UP, _UP, _DN) - _kr(_UP, _DN, _UP)
                                      - _kr(_DN, _UP, _UP),
        "product |uud>": _kr(_UP, _UP, _DN),
        "Delta   |uuu>": _kr(_UP, _UP, _UP),
    }


def main():
    print("Pairwise C_ij composite-spin readout (#514) — J2 = 9/4 + 2 sum <S_i.S_j>\n")
    print(f"{'state':<26}{'J2':>8}{'floor':>9}{'conn':>9}   C_ij")
    for name, psi in _clean_states().items():
        d = j2_decomposition(psi)
        cij = "  ".join(f"{k[0]}{k[1]}={v:+.3f}" for k, v in d["C_ij"].items())
        print(f"{name:<26}{d['j2']:>8.4f}{d['j2_disconnected']:>9.4f}"
              f"{d['j2_connected']:>9.4f}   {cij}")
    print("\n  proton 3/4 (its per-hole Bloch read floors at 9/4; the connected part carries")
    print("  the -3/2). product |uud> 7/4. Delta 15/4. C_ij = 0 iff product.")


if __name__ == "__main__":
    main()
