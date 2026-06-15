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
1. compose_h2_circuit:
   Build the minimal-basis, tapered 2-qubit H₂ ground-state-preparation circuit
   U_H2 (fixed charge sector N=2 / Sz=0) out of supported charge-conserving gates.

2. synthesize_bulk:
   Feed U_H2 (as a unitary, via the existing column/Choi gate-realizability input)
   through Register / EigenstateSynthesis. Reuse the existing
   C^4 = C[H^1(T^2; Z2)] register -- H₂ is a 2-qubit state that embeds in it.
   Punch the holes, grow b_1, drive the spectral residual r_U -> 0.
   Do NOT switch to feeding raw state holonomies/periods (troubleshooting lever only).

3. relax:
   Mediate with F_beta = r_U + beta * dualReggeAction, keeping ReggeSolver's
   gravitational + matter action exactly as-is (no bolted-on elastic action).
   EXTREMIZE the action (delta S = 0) by minimizing ||grad S||^2 + Gamma * r_U over
   edge lengths -- the FULL complex action, keeping Im S -- regulated by the mass
   term to a convergent interior minimum of G = Re S + kappa * E + lambda * |Im S|
   when the conformal direction runs. Never minimize the bare action.

4. read_out:
   Read emergent primal AND dual geometry; occupied holes -> charge; deficit-angle
   distribution -> curvature/mass; Rayleigh quotient lambda = psi^dagger L_1 psi as
   a candidate energy. Do NOT interpret any single edge as a bond length. Do NOT
   fabricate an I(A:B) mutual-information "bond length" formula -- let the numbers
   emerge and find which quantity matches the correct bond length after-the-fact.

5. compare:
   Overlay against the H₂ E(d) curve. Does the emergent geometry track the
   equilibrium bond and stretch correctly? Catalog mismatches -- especially the
   U(1)-into-Z2^2 embedding near dissociation.

See issue #350 for the full design and hard constraints.
"""

# Implementation follows in subsequent commits on this branch (issue #350).
