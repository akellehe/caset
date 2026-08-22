# Analytic-First Kernel and Cache Contract — Findings (#764)

Wave 0 of the recursive spectral-fiber epic (#763). This report records what
was implemented, the exact identities and their domains, and the measured
positive and negative results, per the merge discipline in the design
specification (section 23).

## What was implemented

| Deliverable | Class | Identity / contract | Domain |
|---|---|---|---|
| Certificate vocabulary | `cobordism::Certificate` | grade (algebraically exact / structure-exact / certified numerical / heuristic discovery) + measured residual, conditioning, dense-reference error, tolerance | static or band-window; positive-semidefinite / Hermitian-indefinite / non-normal |
| Cache keys | `cobordism::AnalyticCache`, `cobordism::TouchedStar` | entries keyed by order-independent component vertex-set fingerprint (`Fingerprint::fingerprintOf`, the `MultiCobordism` block convention) + kind + parameter, stamped with `Spacetime::metricRevisionKey()`; accepted moves publish a `TouchedStar`, invalidation is vertex-set intersection, disjoint siblings survive; unpublished drift serves nothing (fail-safe) | any per-component payload: Hodge blocks, factorizations, spectral projectors, transports, covariance blocks, Wick plans |
| Künneth/Kronecker rule | `cobordism::KuennethProduct` | `L_{A×B} = L_A ⊗ I + I ⊗ L_B`; `spec = {λ_i + μ_j}` (pairwise sums, no product eigensolve) | algebraically exact as a matrix identity for any square factors; as a statement about a complex only for actual product cell structures, verified at degree zero by `productCertificate` (Cartesian-product 1-skeleton with product weights); the staircase `SimplicialProduct` is refused |
| Spectrum-level second quantization | `cobordism::OccupationSpectra` | `spec dΓ(h)\|_{Λ^N} = {Σ_{i∈S} λ_i, \|S\|=N}` (occupation subset sums); `F_-(h_A⊕h_B) ≅ F_-(h_A)⊗̂F_-(h_B)` and `dΓ(h_A⊕h_B) = dΓ(h_A)⊗̂I + I⊗̂dΓ(h_B)` read at the spectrum level (`directSumSubsetSums` = merged pairwise sums over particle splits); one-particle `directSum`/`hoppingBlock` assembly with `C' = C†` default and an explicit reverse block for the non-normal regime | exact for any square one-particle operator (complex eigenvalues allowed); output-sensitive, refuses unmaterializable outputs |
| Low-rank updates | `cobordism::LowRankUpdate` | Woodbury `(A+UW)^{-1}b = A^{-1}b − A^{-1}U(I_r+WA^{-1}U)^{-1}WA^{-1}b` by factor solves only; secular `f(λ)=1+ρΣ\|z_i\|²/(d_i−λ)=0` by interlacing bisection with known endpoint signs | Woodbury: general complex square base (no Hermitian/positive assumption), structure-exact GIVEN the verified premise that `UW` spans the full affected change (`factorsFromTouched` support check, `spansAffectedChange`, `refactor` cold fallback); secular: Hermitian (indefinite allowed) only, non-ascending input refused |
| Dense reference | `cobordism::DenseReference` | LU factor solve, verified-Hermiticity eigensolve, spectrum-level dense-Fock oracle (dense eigensolve + explicit subset-sum enumeration) | fixtures only: refuses at/above the configurable crossover (default 512) |

Ontology/dynamics/readout: this ticket affects NONE of the three — it is
kernel and cache infrastructure. Nothing here enters either emergence
objective.

## Measured results

`OMP_NUM_THREADS=8 python scripts/benchmark_analytic_kernel.py` (defaults:
20×20 product grid → dimension 400, seed 764, medians of 5):

| Scenario | Structured | Dense/cold | Speedup | Agreement |
|---|---:|---:|---:|---:|
| local metric change (per-move: exact touched-star factors + verification + Woodbury solve vs cold LU + solve, dim 400, rank 2) | 8.3 ms | 14.7 ms | 1.8× | max solution deviation 3.9e-16; both residuals ~3.5e-16 |
| local topology change (edge creation, same comparison) | 6.8 ms | 12.3 ms | 1.8× | 3.9e-16 |
| marginal repeated solve (factorizations prepared) | 0.11 ms | 0.10 ms | ~1× | — |
| product-complex spectrum (two 20-dim factor eigensolves + pairwise sums vs dense eigensolve of the 400-dim Kronecker sum) | 0.070 ms | 90.7 ms | ~1300× | max spectrum deviation 9.5e-13; product certificate residual 2.0e-16 |
| second-quantized sector (C(20,2)=190 subset sums vs eager 2^20 Fock enumeration) | 6.8 µs | 113 ms | ~16700× | exact (same enumeration arithmetic); hopping block dim 192 solved in 15.5 ms, deviation vs numpy 5.1e-13 |
| cache invalidation (12 ring components × 48 vertices, 20 random accepted moves: publish + recompute one component vs recompute all) | 7.5 ms | 86.0 ms | 11.5× | payload deviation 0.0 (bit-identical); 1100 hits / 100 misses / 100 invalidations |

The structured path beats the dense reference well below the documented
crossover (`DenseReference::kDefaultCrossoverDimension = 512`): at dimension
400 every structured scenario wins, so the dense kernels' refusal at ≥512 is
conservative.

## Negative results and honest limits

- The bare `k = 0` graph Laplacian is exactly singular (gauge/constant
  kernel — phases on a tree gauge away), so benchmarking Woodbury on the
  unshifted operator produced garbage solves that the certificate correctly
  flagged (`residual ≈ 6.3`, `holds() == false`). The meaningful repeated
  solve is the shifted response pencil `L + σI` (design spec section 5.3);
  the benchmark uses `σ = 1`. This is a validation of the certificate
  contract, not a kernel defect.
- A first secular-update implementation evaluated the secular function at
  bracket endpoints; at double precision the pole-offset underflowed
  (`lo + width·ε == lo`) and two roots collapsed onto their poles, again
  flagged by the certificate (residual 0.02). The landed implementation
  bisects with the mathematically known endpoint signs (monotone `f`, poles
  only at interval endpoints) and agrees with dense `eigh` to ~1e-12 on all
  fixtures, including deflation, duplicates, and negative `ρ`.
- Per-move structured speedup at dimension 400 is 1.8×, not the marginal
  ~70×: the per-move path pays O(n²) for the exactness verification
  (`factorsFromTouched` support scan) and, from Python, the flat-list
  binding conversion. C++ callers and repeated solves amortize both.
- `AnalyticCache.publish` trusts the caller to publish the COMPLETE touched
  record for the revision jump; an incomplete star could serve a stale
  sibling. Unpublished drift, by contrast, is fail-safe (nothing served).
  Wiring automatic publication into the move classes is deliberately left
  to the consumers (#768/#769) so this ticket adds no hooks inside
  simulation dynamics.
- The dense-Fock oracle is spectrum-level only (dense eigensolve + subset
  enumeration). The occupation-basis `dΓ` OPERATOR matrix needs the
  fermionic sign algebra, which is the exterior-algebra track's (#766);
  once merged, that operator should be cross-wired as a second independent
  reference behind `DenseReference::fockSpectrum` crossover fixtures.
