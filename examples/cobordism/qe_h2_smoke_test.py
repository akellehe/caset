"""H₂ smoke test: reproduce a Quantum-ESPRESSO observable via Regge-mediated cobordism.

Goal
----
Reproduce a QE-computable observable of the H₂ molecule -- the equilibrium bond
length (~0.74 Angstrom) and the binding curve E(d) -- by composing H₂ as a
charge-conserving quantum circuit, synthesizing it through the EXISTING cobordism
path, and Regge-mediated relaxing the result. This is the lowest-complexity
candidate in the QE-reproduction ranking; exact minimal-basis reference values
exist independently of QE, so there is ground truth either way.

The result is read out emergent-first: the geometric readout plus an overlay
against the exact/QE curve, AND a catalog of what does NOT line up. Interpretation
is discovered after-the-fact, never assumed.

Pipeline (each stage reuses existing machinery -- no new inputs, no new register)
--------------------------------------------------------------------------------
1. compose_h2_circuit (THIS COMMIT):
   Build the minimal-basis H₂ ground-state-preparation circuit U_H2 out of a
   number-conserving rotation. In STO-3G, H₂ has two spatial orbitals (sigma_g
   bonding, sigma_u antibonding); the closed-shell singlet ground state is the
   seniority-zero two-configuration state c_g |sigma_g^2> + c_u |sigma_u^2>. That
   is prepared from the Hartree-Fock determinant by a single Givens rotation
   theta(d) -- a particle-number-conserving 2-qubit gate. We embed the two
   configurations as register basis states HF -> [a] = |01>, double -> [b] = |10>,
   so U_H2 is the Givens block on the {[a],[b]} single-excitation subspace. The
   exact FCI energy E(d) and theta(d) come from pyscf (ground truth, independent
   of QE). Equilibrium ~0.7414 Angstrom, E ~ -1.1373 Ha.

2. synthesize_bulk (LATER):
   Feed U_H2 (as a unitary, via the existing column/Choi gate-realizability input)
   through Register / EigenstateSynthesis. Reuse the existing
   C^4 = C[H^1(T^2; Z2)] register -- H₂ is a 2-qubit state that embeds in it.
   Punch the holes, grow b_1, drive the spectral residual r_U -> 0.
   Do NOT switch to feeding raw state holonomies/periods (troubleshooting lever only).

3. relax (LATER):
   Mediate with F_beta = r_U + beta * dualReggeAction, keeping ReggeSolver's
   gravitational + matter action exactly as-is (no bolted-on elastic action).
   EXTREMIZE the action (delta S = 0) by minimizing ||grad S||^2 + Gamma * r_U over
   edge lengths -- the FULL complex action, keeping Im S -- regulated by the mass
   term to a convergent interior minimum of G = Re S + kappa * E + lambda * |Im S|
   when the conformal direction runs. Never minimize the bare action.

4. read_out (LATER):
   Read emergent primal AND dual geometry; occupied holes -> charge; deficit-angle
   distribution -> curvature/mass; Rayleigh quotient lambda = psi^dagger L_1 psi as
   a candidate energy. Do NOT interpret any single edge as a bond length. Do NOT
   fabricate an I(A:B) mutual-information "bond length" formula -- let the numbers
   emerge and find which quantity matches the correct bond length after-the-fact.

5. compare (LATER):
   Overlay against the H₂ E(d) curve. Does the emergent geometry track the
   equilibrium bond and stretch correctly? Catalog mismatches -- especially the
   U(1)-into-Z2^2 embedding near dissociation.

First emergent finding (stage 1)
--------------------------------
H₂'s prep is a number-conserving Givens rotation, but particle-number U(1) is NOT
the register's holonomy charge. The existing realizability criterion (conserves
holonomy charge == equal column sums of the {[a],[b],[a+b]} block, see
spectral_gate_realizability.conserves_charge) is satisfied by the Givens block
only in the theta -> 0 limit (Hartree-Fock-dominated). It is maximally violated as
theta -> 45 degrees near dissociation, exactly where static correlation is strong.
That is the predicted "U(1)-into-Z2^2" mismatch, surfaced -- not papered over. We
preview it here with the same column-sum diagnostic; the genuine spectral verdict
is computed by the tessera path in stage 3.

Reference: O'Malley et al., "Scalable Quantum Simulation of Molecular Energies",
PRX 6, 031007 (2016) -- the canonical 2-qubit reduced H₂.

See issue #350 for the full design and hard constraints.
"""

from __future__ import annotations

import os

import numpy as np

# Register holonomy basis order, shared with spectral_gate_realizability.py:
#   0 = [triv] = |00>, 1 = [a] = |01>, 2 = [b] = |10>, 3 = [a+b] = |11>.
HF_INDEX = 1       # Hartree-Fock determinant |sigma_g^2>  -> [a] = |01>
DOUBLE_INDEX = 2   # double excitation       |sigma_u^2>  -> [b] = |10>

# Equilibrium reference (STO-3G / FCI), used only for messaging.
EQUILIBRIUM_ANGSTROM = 0.7414


def h2_seniority_zero_state(distance_angstrom):
    """Exact minimal-basis (STO-3G) H₂ at a given bond distance.

    Returns (energy_hartree, c_g, c_u): the FCI ground-state energy and the two
    seniority-zero configuration amplitudes (sigma_g^2 and sigma_u^2), normalized
    so c_g**2 + c_u**2 == 1. The closed-shell singlet ground state has no
    open-shell (seniority-2) component, so these two amplitudes are the whole
    state. Requires pyscf (tessera[examples]).
    """
    from pyscf import gto, scf, fci

    mol = gto.M(atom=f"H 0 0 0; H 0 0 {distance_angstrom}",
                basis="sto-3g", unit="Angstrom", verbose=0)
    mean_field = scf.RHF(mol)
    mean_field.kernel()
    energy, civec = fci.FCI(mean_field).kernel()

    # civec[i, j] indexes (alpha determinant, beta determinant) over the two
    # spatial orbitals. The seniority-zero configs are the diagonal:
    #   civec[0, 0] = sigma_g^2,  civec[1, 1] = sigma_u^2.
    c_g, c_u = float(civec[0, 0]), float(civec[1, 1])
    norm = np.hypot(c_g, c_u)
    return float(energy), c_g / norm, c_u / norm


def givens_block(c_g, c_u):
    """The 2x2 number-conserving Givens rotation preparing the seniority-zero
    state from Hartree-Fock: it sends |HF> -> c_g |HF> + c_u |double>, with the
    orthogonal completion on |double>. cos(theta) = c_g, sin(theta) = c_u."""
    return np.array([[c_g, -c_u],
                     [c_u, c_g]], dtype=complex)


def compose_h2_circuit(distance_angstrom):
    """STAGE 1: the H₂ ground-state-preparation unitary U_H2 in the 4-dim register
    basis. A particle-number-conserving Givens rotation on the {[a],[b]}
    single-excitation subspace; [triv] and [a+b] are fixed. This is the unitary
    fed UNCHANGED through the existing gate-realizability path in stage 2.

    Returns (U, energy_hartree, theta_radians).
    """
    energy, c_g, c_u = h2_seniority_zero_state(distance_angstrom)
    block = givens_block(c_g, c_u)

    u = np.eye(4, dtype=complex)
    idx = (HF_INDEX, DOUBLE_INDEX)
    for i, a in enumerate(idx):
        for j, b in enumerate(idx):
            u[a, b] = block[i, j]
    theta = float(np.arctan2(c_u, c_g))
    return u, energy, theta


def holonomy_charge_defect(u, tol=1e-9):
    """Preview of the stage-3 realizability criterion (the genuine verdict is the
    tessera spectral residual). Holonomy charge is conserved iff the three column
    sums of the {[a],[b],[a+b]} block are equal. Returns (max_defect, conserved):
    max_defect is the spread of the column sums (0 iff conserved)."""
    block = np.asarray(u, dtype=complex)[1:4, 1:4]
    sums = block.sum(axis=0)
    defect = float(np.max(np.abs(sums - sums[0])))
    return defect, bool(defect < tol)


def reference_curve(distances_angstrom):
    """The exact H₂ binding curve plus, per distance, the prep angle and the
    holonomy-charge-defect preview. Returns a list of dict rows."""
    rows = []
    for d in distances_angstrom:
        u, energy, theta = compose_h2_circuit(d)
        defect, conserved = holonomy_charge_defect(u)
        rows.append({"d": float(d), "energy": energy,
                     "theta_deg": float(np.degrees(theta)),
                     "charge_defect": defect, "charge_conserved": conserved})
    return rows


def _register_path():
    """Import the working gate-realizability path (needs the tessera C++ build).
    We reuse its primitives so U_H2 is fed through the EXACT existing input -- no new
    register, no new modality. Honors the 16-CPU cap before the BLAS is pulled in."""
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "BLIS_NUM_THREADS"):
        os.environ.setdefault(_v, "16")
    import spectral_gate_realizability as sgr
    return sgr


def synthesize_bulk(distances_angstrom, sgr=None, reg=None):
    """STAGE 2: feed each U_H2(d) through the existing path UNCHANGED. post_interaction
    extracts U_H2's {[a],[b],[a+b]} block, applies it to the generic register input
    psi_B (Sigma = 0), and scores the genuine Hodge L_1 residual of the post-interaction
    state on the surgery-grown bulk -- r -> 0 iff U_H2|psi_B> is carried by ker L_1.

    This is the genuine spectral verdict (the stage-1 charge defect was only its
    algebraic shadow). Returns one row per distance.
    """
    if sgr is None:
        sgr = _register_path()
    if reg is None:
        reg = sgr.Register()
    rows = []
    for d in distances_angstrom:
        u, energy, theta = compose_h2_circuit(d)
        residual, b1, leak = sgr.post_interaction(reg, u)
        defect, _ = holonomy_charge_defect(u)
        rows.append({"d": float(d), "energy": energy,
                     "theta_deg": float(np.degrees(theta)),
                     "residual": float(residual), "b1": int(b1), "leak": float(leak),
                     "charge_defect": defect,
                     "realizable": bool(residual < sgr.REALIZE)})
    return rows


def _main():
    distances = [0.4, 0.5, 0.6, 0.7414, 0.9, 1.1, 1.4, 1.8, 2.5, 3.5]
    rows = reference_curve(distances)
    eq = min(rows, key=lambda r: r["energy"])

    print("H2 minimal-basis (STO-3G/FCI) reference + stage-1 circuit diagnostics")
    print(f"{'d (A)':>7} {'E (Ha)':>11} {'theta (deg)':>12} "
          f"{'charge defect':>14} {'conserved':>10}")
    for r in rows:
        print(f"{r['d']:7.4f} {r['energy']:11.6f} {r['theta_deg']:12.4f} "
              f"{r['charge_defect']:14.6f} {str(r['charge_conserved']):>10}")
    print(f"\nEmergent equilibrium: d = {eq['d']:.4f} A, E = {eq['energy']:.6f} Ha "
          f"(reference {EQUILIBRIUM_ANGSTROM} A, -1.1373 Ha)")
    print("Finding: holonomy-charge defect grows with theta (toward dissociation) "
          "-- the predicted U(1)-into-Z2^2 mismatch.")

    try:
        sgr = _register_path()
    except ImportError as exc:
        print(f"\n[stage 2 skipped: tessera C++ path unavailable -- {exc}]")
        return

    print("\nSTAGE 2 -- genuine spectral residual of U_H2|psi_B> on the surgery-grown "
          "register (the working path, fed unchanged)")
    reg = sgr.Register()
    s2 = synthesize_bulk(distances, sgr=sgr, reg=reg)
    print(f"{'d (A)':>7} {'theta (deg)':>12} {'r_U (residual)':>16} "
          f"{'b1':>4} {'leak |Sigma|':>13} {'realizable':>11}")
    for r in s2:
        print(f"{r['d']:7.4f} {r['theta_deg']:12.4f} {r['residual']:16.3e} "
              f"{r['b1']:4d} {r['leak']:13.6f} {str(r['realizable']):>11}")
    print("Compare leak/residual vs the stage-1 charge defect: the genuine spectral "
          "test should track the algebraic shadow. Stages 3-5 mediate + relax.")


if __name__ == "__main__":
    _main()
