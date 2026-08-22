# Referee response: recursive spectral fibers

This note records the disposition of the external review supplied after the first
whitepaper draft. The revised LaTeX/PDF and design specification are normative.
“Accepted” means the paper contained an internal inconsistency or an under-specified
construction. “Qualified” means the diagnosis was useful but the proposed repair
made a stronger claim than the mathematics supports.

## Disposition

| # | Disposition | Revision |
|---:|---|---|
| 1 | Accepted diagnosis; modified repair | A list of normalized edge qubit states would describe only a product preparation. The revised ontology says that every edge carries a two-level occupation **mode**, while the state is a generally entangled vector/density operator on `F_-(h)=Λ•h`. A spectral projector defines a quasi-free/Slater reference and derived occupations `<n_e>=P_ee`, but this is not imposed on the interacting proton sector. |
| 2 | Accepted | Simplicial gluing is now separated from Hilbert-space tensoring. Gluing builds a direct-sum/quotient one-particle chain space with coupling blocks; the Fock functor gives `F_-(h_A⊕h_B)≅F_-(h_A)⊗̂F_-(h_B)` and `dΓ` turns coupling blocks into hopping terms. The chain-complex tensor identity remains only for an actual product complex. |
| 3 | Accepted | Plain Schur/Kron reduction is exact only for supported static response. Nonzero bands use the energy-dependent Feshbach-Schur pencil. Reusable linear reductions use Craig-Bampton/AMLS and carry a frequency window, discarded-mode gap, and residual. Indefinite Hermitian elimination is stationarity; the non-normal compatibility test uses the left kernel. |
| 4 | Accepted ambiguity; physical interpretation qualified | The code may not silently divide by a cube root of `det V`. Generic transport is `U(r)`. At rank three it stores the full `U(3)` factor, determinant line, projective `SU(3)/Z3` class, and any continuously chosen center lift. Determinant winding is an exact integer only for a closed full-rank/gapped family. Identifying `B=ν/3` is explicitly a falsifiable hypothesis, not a group-theoretic theorem. |
| 5 | Accepted | Raw determinant holonomy includes Abelian Berry phase. Exchange and `2π` tests are now interferometric ratios against matched non-exchanging/co-moving reference loops. Structural permutation parity is reported independently. |
| 6 | Partly accepted | A tensor-product qubit/bitset implementation needs a deterministic mode order and permutation parity. The abstract exterior Fock space and CAR do **not** require a Jordan-Wigner order, Kasteleyn orientation, or spin structure. Kasteleyn orientations describe surface-dimer Pfaffians, not arbitrary 4D complexes. A spin lift and possible `w2` obstruction are required only when claiming that an emergent tangent-frame rotation is a physical continuum spinor rotation. |
| 7 | Accepted category objection; sheaf claim qualified | Beyond level zero, the output is called an operator-valued response network, not automatically a simplicial complex. Cellular sheaves are a natural realization when explicit stalk restriction maps reproduce the blocks. The revision does not assert that every higher-Hodge Schur complement is automatically a sheaf Laplacian; that factorization is itself certified. |
| 8 | Mostly accepted | Hermitian-indefinite bands report Krein inertia; non-normal bands use biorthogonal left/right frames and conditioning. Negative norm/signature is **not** identified automatically with an antiparticle. Current Newman-Girvan modularity is declared an unweighted/nonnegative heuristic proposal generator, with resolution-limit fixtures; all decisive downstream certificates are weight-aware. |
| 9 | Accepted bridge; modified repair | A rank-three band now needs an oriented-triangle anchoring certificate. Requiring concentration on one triangle would wrongly exclude extended clusters, so the production certificate uses an atlas score `Σ_τ w_τ|det(R_τΦ)|²` plus determinant-phase coherence. A literal triangle is the exact oracle fixture. |
| 10 | Rejected as applied; retained as conditional warning | `Λ•h_modes` is occupation-number Fock space. A Kähler-Dirac field is an inhomogeneous cochain field acted on by `d-d*`. The former is not “Kähler-Dirac in disguise” and has no automatic four-taste theorem. If Tessera later adopts the latter operator, known 4D taste multiplicity becomes a mandatory diagnostic. |

## Minor findings

- Accepted: generic transport uses `U(r)` polar; color interpretation begins only
  at an anchored rank-three level.
- Accepted: color normalization uses the stored complex squared lengths directly,
  `c=z/||z||_2`; perimeter gauge and Hilbert normalization remain distinct.
- Accepted: every nonzero coarse-response comparison names its frequency window.
- Not accepted as a theorem: Regge-Hodge stationarity alone does not imply that the
  first variation of transport Gram defect vanishes. Hellmann-Feynman/envelope
  arguments concern derivatives of the optimized spectral/action quantity, not an
  unrelated diagnostic. The simulation will measure the proposed correlation and
  may promote it only after additional hypotheses or a proof tie the defect to the
  stationary functional.

## Exact core after revision

The following remain exact in their stated domains:

1. weighted boundary/adjoint/Hodge identities;
2. positive static Schur energy response, indefinite Schur stationarity, and
   compatible block elimination;
3. energy-dependent finite-dimensional Feshbach isospectrality away from eliminated
   resonances;
4. exterior algebra, CAR, Pauli/Gram determinant, and `su(3)` bilinear algebra;
5. `F_-(h_A⊕h_B)≅F_-(h_A)⊗̂F_-(h_B)` and the `dΓ` direct-sum identity;
6. bifundamental covariance of accepted frame-overlap transport and conjugacy
   invariance of closed holonomies;
7. integer determinant winding for closed, continuous, full-rank loops; and
8. the existing Gram-defect amplitude bounds and inductive-limit dimension identity.

The quark, baryon-number, flavor, confinement, spin, and proton identifications
remain proposed physical readings with explicit failure states.
