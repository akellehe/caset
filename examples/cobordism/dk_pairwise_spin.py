# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Pairwise connected-correlator readout of the composite proton spin J² (#514, part of #410).

`dk_composite_spin.py` (#485) / `dk_joint_spin.py` (#489) established that a **product** of
independently-extracted per-hole spinors floors the composite spin at the mixture (`~9/4`),
because the proton's `J²=¾` is an **entangled**, mixed-symmetry quantity. This module reads the
same `J²` through a different, lighter lens — the one set out in
`docs/theory/cobordism/proton-spin/cartan_weyl_gluon.tex` §5(a) / §7 experiment 1.

## The key fact: J² is a two-body operator

For three spin-½'s, `S_i² = ¾·I`, so

    J² = Σ_a (Σ_i S_a^{(i)})²  =  Σ_i S_i²  +  2 Σ_{i<j} S_i·S_j  =  9/4  +  2 Σ_{i<j} S_i·S_j .

There is **no three-body term**. Hence `⟨J²⟩` is *exactly* determined by the three pairwise
reduced two-qubit states `ρ_ij` — `⟨S_i·S_j⟩ = Tr(ρ_ij · Σ_a S_a⊗S_a)` — and never needs the
full joint state. That is what makes this readout cheaper than a total-space operator on the
joint field (#477/#495): three 4×4 marginals instead of one operator on the whole carried field.

## What the per-hole read discards: the connected correlator

Split each pair correlator into a disconnected (single-hole Bloch) part and a connected part:

    ⟨S_i·S_j⟩ = ⟨S_i⟩·⟨S_j⟩ + C_ij ,     C_ij := ⟨S_i·S_j⟩ − ⟨S_i⟩·⟨S_j⟩ .

A per-hole Bloch / product read keeps only `⟨S_i⟩·⟨S_j⟩` (it sets `C_ij = 0`):

    J²_disconnected = 9/4 + 2 Σ_{i<j} ⟨S_i⟩·⟨S_j⟩          (the floor),
    J²              = J²_disconnected + 2 Σ_{i<j} C_ij      (the truth).

`C_ij = 0` **iff** the pair is uncorrelated (a product). On the exact proton eigenstate the
per-hole Bloch read gives precisely `9/4` and the connected part carries the missing `−3/2`
— the entanglement the product read throws away. So the chamber-coordinate diagnosis is sharp:
the whole proton/Δ distinction lives in the three numbers `C_ij`, computable from two-body
marginals alone.

Everything here is pure NumPy and matches `dk_joint_spin.j2_three_qubit` (the validated
instrument): the spin lives in `(C²)³`, not the 4-component Dirac space. The emergent entry
point at the bottom lazily imports `tessera`; it is exercised only in `__main__`.
"""
import collections
import importlib.util
import os
import sys

import numpy as np
import scipy.linalg

# Clean spin-½ operators in C² (matching dk_joint_spin._S / _PAULI exactly).
_PAULI = [np.array([[0, 1], [1, 0]], complex),
          np.array([[0, -1j], [1j, 0]]),
          np.array([[1, 0], [0, -1]], complex)]
_S = [p / 2 for p in _PAULI]
_I2 = np.eye(2, dtype=complex)

# The two-qubit spin-spin operator S_i·S_j = Σ_a S_a⊗S_a (4×4). Symmetric under the two
# factors, so the pair correlator does not depend on which qubit is named first.
_SS = sum(np.kron(_S[a], _S[a]) for a in range(3))


def _normalize(psi):
    psi = np.asarray(psi, complex)
    n = np.linalg.norm(psi)
    return psi / n if n > 1e-30 else psi


# ----- reduced states (partial traces of a 3-qubit pure state) -----
def reduced_states(psi8):
    """The three pairwise (`ρ_ij`, 4×4) and three single (`ρ_i`, 2×2) reduced density matrices
    of a normalized three-qubit pure state `psi8` (ordering hole0 ⊗ hole1 ⊗ hole2). Returns
    `(pairs, singles)` with `pairs[(i,j)]` and `singles[i]`; every reduced state has unit trace."""
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
    """The per-hole Bloch / product floor: `9/4 + 2 Σ_{i<j} ⟨S_i⟩·⟨S_j⟩` (the readout with
    every `C_ij` set to zero). This is the value any per-hole product read is pinned to."""
    return j2_decomposition(psi8)["j2_disconnected"]


def j2_decomposition(psi8):
    """The full chamber-coordinate breakdown of `J²` for a 3-qubit pure state. Returns a dict:

      `j2`               full `J²` (= `j2_disconnected + j2_connected`),
      `j2_disconnected`  the per-hole Bloch floor `9/4 + 2 Σ ⟨S_i⟩·⟨S_j⟩`,
      `j2_connected`     `2 Σ_{i<j} C_ij` — the entanglement the product read discards,
      `C_ij`             dict `{(i,j): C_ij}` (the connected correlators),
      `spin_correlators` dict `{(i,j): ⟨S_i·S_j⟩}`,
      `bloch`            dict `{i: ⟨S⟩_i}` (per-hole Bloch vectors).
    """
    pairs, singles = reduced_states(psi8)
    blochs = {i: bloch(r) for i, r in singles.items()}
    corr = {ij: spin_correlator(r) for ij, r in pairs.items()}
    cij = {ij: corr[ij] - float(blochs[ij[0]] @ blochs[ij[1]]) for ij in pairs}
    j2_disc = 2.25 + 2.0 * sum(float(blochs[i] @ blochs[j]) for (i, j) in pairs)
    j2_conn = 2.0 * sum(cij.values())
    return {"j2": j2_disc + j2_conn, "j2_disconnected": j2_disc, "j2_connected": j2_conn,
            "C_ij": cij, "spin_correlators": corr, "bloch": blochs}


def _j2_direct(psi8):
    """Reference full-operator `J² = ⟨ψ| Σ_a (Σ_i S_a^{(i)})² |ψ⟩` in `(C²)³`, pure NumPy. This
    mirrors `dk_joint_spin.j2_three_qubit` (the validated instrument) and exists so the test
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


# ----- emergent entry point (lazy tessera; exercised in __main__) -----
_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_example(name):
    """Load a sibling example module (lazy: pulls in `tessera` only when first called)."""
    sys.path.insert(0, _HERE)
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def emergent_pairwise_decomposition(sub, holes, joint=True):
    """Read the three per-hole spinors off the (color-correlated) carried field of the proton
    block, reduce each to a qubit, and run `j2_decomposition` on their product.

    With the **current** per-hole extraction the reconstructed state is a product, so every
    `C_ij ≈ 0` and `J²` floors — the honest localization of the open kernel: a *faithful joint
    two-hole read* of `ρ_ij` (the Whitney / Kähler–Atiyah fiber↔cells lift, #477/#495) is what
    would supply nonzero `C_ij`. Lazy `tessera` import via the sibling example modules."""
    cs = _load_example("dk_composite_spin")
    dj = _load_example("dk_joint_spin")
    tt = _load_example("emergent_optimizer")._top_tuple
    spinors = (cs.joint_spinors(sub, holes, tt) if joint
               else cs.emergent_spinors(sub, holes, tt))
    qubits = [dj.spinor_to_qubit(s) for s in spinors]
    psi = qubits[0]
    for q in qubits[1:]:
        psi = np.kron(psi, q)
    return j2_decomposition(psi)


def emergent_joint_pairwise(sub, holes, joint=True):
    """FIRST CUT at the genuine joint two-hole read: per hole take the Dirac spinor, transport
    all to a common frame via the spin `wilson_line`, and build each pairwise `rho_ij` as a
    field-sourced Werner state — its correlation `lambda` and singlet/triplet character read
    from the transported inter-hole spinor overlap (`color_phase_correlation`), NOT inserted by
    hand. Returns `decomposition_from_pairs`, so `C_ij` can be nonzero (unlike the product read
    `emergent_pairwise_decomposition`, which factorizes to C_ij≈0 — obstruction 2).

    Honest scope: this sources the rotationally-invariant correlation the floor lacks, but the
    Werner form is an ansatz (not a derived rho_ij), and it reads only the color/spin phase —
    the flavor-driven u-u spin triplet (the `+1/4` pair) is not captured, since that needs spin
    AND flavor jointly (cartan section 5). Lazy `tessera`. None if a Wilson line is missing."""
    cs = _load_example("dk_composite_spin")
    dj = _load_example("dk_joint_spin")
    tt = _load_example("emergent_optimizer")._top_tuple
    cells = list(sub.getTopSimplices())
    adj = cs._dual_adjacency(cells, tt)
    frames = cs._frames(cells, tt)

    def cell_of(h):
        hv = set(h)
        return max(cells, key=lambda c: len(hv & set(tt(c))))

    hc = [cell_of(h) for h in holes[:3]]
    lines = [cs.wilson_line(cells, adj, hc[0], hc[j], frames) for j in range(3)]
    if any(u is None for u in lines):
        return None
    spinors = (cs.joint_spinors(sub, holes, tt) if joint
               else cs.emergent_spinors(sub, holes, tt))
    s = [lines[j] @ spinors[j] for j in range(3)]              # transported to hole 0's frame
    pairs = {}
    for (i, j) in [(0, 1), (0, 2), (1, 2)]:
        lam, sw = color_phase_correlation(s[i], s[j])
        pairs[(i, j)] = werner_pair(dj.spinor_to_qubit(s[i]), dj.spinor_to_qubit(s[j]), lam, sw)
    return decomposition_from_pairs(pairs)


# ----- demo -----
_UP = np.array([1, 0], complex)
_DN = np.array([0, 1], complex)


def _kr(*a):
    out = a[0]
    for x in a[1:]:
        out = np.kron(out, x)
    return out


def _clean_states():
    """The same hand-fed clean states guarded in `test_dk_joint_spin.py`."""
    return {
        "proton  2|uud>-|udu>-|duu>": 2 * _kr(_UP, _UP, _DN) - _kr(_UP, _DN, _UP)
                                      - _kr(_DN, _UP, _UP),
        "product |uud>": _kr(_UP, _UP, _DN),
        "Delta   |uuu>": _kr(_UP, _UP, _UP),
    }


def main():
    print("Pairwise C_ij composite-spin readout (#514) — J² = 9/4 + 2 Σ ⟨S_i·S_j⟩\n")
    print(f"{'state':<26}{'J²':>8}{'floor(disc)':>14}{'connected':>12}   C_ij")
    for name, psi in _clean_states().items():
        d = j2_decomposition(psi)
        cij = "  ".join(f"{k[0]}{k[1]}={v:+.3f}" for k, v in d["C_ij"].items())
        print(f"{name:<26}{d['j2']:>8.4f}{d['j2_disconnected']:>14.4f}"
              f"{d['j2_connected']:>12.4f}   {cij}")
    print("\n  proton ¾=0.75 (its per-hole Bloch read floors at 9/4=2.25; the connected")
    print("  part carries the −3/2). product |uud> 7/4=1.75. Δ 15/4=3.75. C_ij=0 ⟺ product.")

    # Emergent demo (slow; honest C_ij ≈ 0 with the current per-hole extraction).
    try:
        dj = _load_example("dk_joint_spin")
    except Exception as exc:  # tessera not built / importable
        print(f"\n  [emergent demo skipped: {exc}]")
        return
    print("\nEmergent proton block (built by simultaneous pair creation, #489):")
    found = dj.build_converged_proton(seeds=range(3, 14), max_residual=0.5,
                                      n_refine=18, stage1_steps=60, stage2_iters=20)
    if not found:
        print("  no converged 3-hole proton block in the seed range (honest negative).")
        return
    opt, rd, seed = found
    prod = emergent_pairwise_decomposition(rd["_sub"], rd["_holes"], joint=True)
    jnt = emergent_joint_pairwise(rd["_sub"], rd["_holes"], joint=True)
    pc = "  ".join(f"{k[0]}{k[1]}={v:+.3e}" for k, v in prod["C_ij"].items())
    print(f"  seed {seed}")
    print(f"  product read (obstruction 2):  J²={prod['j2']:.4f}   C_ij = {pc}")
    print("                                 ⇒ factorizes to a product, C_ij≈0, J² floors")
    if jnt is not None:
        jc = "  ".join(f"{k[0]}{k[1]}={v:+.3f}" for k, v in jnt["C_ij"].items())
        print(f"  field-sourced joint (Werner):  J²={jnt['j2']:.4f}   floor={jnt['j2_disconnected']:.4f}"
              f"   connected={jnt['j2_connected']:+.3f}")
        print(f"                                 C_ij = {jc}")
    print("  ⇒ the field-sourced joint read makes C_ij nonzero (the color/spin locking); a")
    print("  derived (non-ansatz) ρ_ij and the flavor-driven u-u triplet remain the open kernel.")


if __name__ == "__main__":
    main()
