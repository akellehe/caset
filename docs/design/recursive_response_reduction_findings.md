# Recursive response reduction — findings (#768)

Scope: `RecursiveQuotient` (static supported Schur response, Feshbach–Schur
pencil, Craig–Bampton surrogate, labeled retained-fiber sum, response
network/sheaf realization, nested quotients, #764 cache reuse). Layer:
READOUT only — no ontology or dynamics change; nothing here enters either
emergence objective.

## Positive results

- **Static identity exact on hand fixtures.** Path (weights 2, 3 →
  `w_eff = 6/5`), star (weights 1, 2, 3 → `w_i δ_ij − w_i w_j / 6`), and
  triangle (`3/2` parallel-series) reductions match the literal hand
  matrices and independent NumPy pinv-Schur references at 1e−12; the
  interior minimum was confirmed by a brute-force energy grid, independent
  of the solver path.
- **Determinant factorization holds on real geometry.** On the stored #562
  causal specimen (23 vertices, 102 edges, one timelike k=1 cell) the
  measured residual of `det(L−λ) = det(L_II−λ)·det F_B(λ)` is below 1e−9 at
  a complex probe shift, with the regime honestly detected as
  Hermitian-indefinite (signed `W`, `WL` Hermitian to 6e−17).
- **The regime trichotomy maps onto shipped conventions.** Detecting
  self-adjointness against the carried diagonal metric (`WL` Hermiticity)
  lands exactly where the physics does: k=0 graph Laplacian → positive
  (minimization), real signed `ℓ²` d'Alembertian → Hermitian-indefinite
  (stationarity), complex `ℓ²` → non-normal (certified block elimination
  with the left-kernel compatibility condition). One complex edge length on
  the causal specimen flips the regime to non-normal and the elimination
  certificate still holds.
- **Winding multiplicities separate cleanly.** The unwrapped det-phase
  winding of `F_B` plus the separately-computed interior winding recovers
  the algebraic multiplicity even when the contour encloses interior
  spectrum (the pole of `det F_B` is counted where it lives, not smeared
  into the pencil), and the defective Jordan fixture reports algebraic 2 vs
  geometric 1 as required.
- **Per-component caching is exact.** The component contribution depends
  only on operator entries with at least one interior index, and every such
  entry's cells lie inside the component (interior cells couple only within
  their component), so keying `AnalyticCache` entries by the component cell
  vertex-id set is an exact-support invalidation: cached equals cold
  bit-for-bit across accepted metric moves, siblings survive, and the
  benchmark shows deviation 0.0 over 60 move evaluations.

## Negative results and refusals (working as designed)

- **Static Schur does not preserve nonzero spectrum** — demonstrated, not
  just asserted: every eigenvalue of the 4×4 block-pencil fixture's static
  `L_eff` sits at distance > 0.05 from the fine spectrum, while the pencil
  zeros land on the true eigenvalues.
- **Naive internal direct sums miscount.** Two components sharing one
  interface cell give `G = [[1,1],[1,1]]` exactly: nominal labeled rank 2,
  true rank 1. `CertifiedNearIsometry` refuses (defect 1); `QuotientKernel`
  restates rank 1; `CarryGramExactly` stays exact.
- **Defective interiors are refused, not regularized.** A nilpotent interior
  block at the shift makes the kernel-complement block singular; the solve
  reports `holds() == false` with an infinite residual instead of adding a
  diagonal.
- **Sheaf realizations are rare.** The SVD-factored restriction maps
  reproduce the off-diagonal blocks by construction, so emission is decided
  entirely by the vertex-block identity `L_vv = Σ_e ρ†ρ` — which fails for
  any grounded/potential-carrying reduction (tested) and for every
  non-normal network (a sheaf Laplacian is self-adjoint). Generic coarse
  levels are response networks, exactly as the whitepaper anticipates.

## Observations worth carrying forward

- **k=1 interiors are scarce under vertex supports.** The down-Laplacian
  couples edges through shared vertices, so an edge is interior only when
  its full vertex star stays inside one support: on the 6-triangle strip
  only 2 of 13 edges are interior. Wave 2 partitions read at k ≥ 1 should
  expect interface-dominated splits unless supports are chosen with fat
  overlaps.
- **A reduced level made of pure round-off is rank-fragile.** Reducing the
  fully-decoupled two-triangle fixture leaves a 2×2 block of exact zeros
  plus one 1e−16 entry; a child quotient over that block sees σ_max itself
  at noise scale, and any σ_max-relative rank cut then "detects" structure.
  This is precisely the pseudoinverse-cutoff-as-topology trap the ticket
  forbids; the class keeps the standard σ_max-relative cut and the finding
  is recorded here as the reason nested fixtures must carry O(1) blocks
  (the shipped nested tests do).
- **One-shot cold reduction is construction-dominated.** At dim 785 the
  warm repeated reduction is 6.4× faster than the dense one-shot Schur and
  the window sweep is ~390× faster, but the cold path (classification +
  regime factorization + assembly) is 0.78× — the reuse pattern, not the
  single shot, is where the structured path pays, matching the spec §18
  "affected component/window factorization" target.
- **`smithNormalForm` cannot produce kernels.** It reports rank and
  invariant factors only; the exact integer kernel basis needed for the
  topological interior zero modes was added as `integerNullspace` beside
  `gf2Nullspace` (exact rational Gauss–Jordan, overflow-checked) rather
  than rewriting SNF with transform tracking.
