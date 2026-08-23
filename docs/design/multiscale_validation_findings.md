# Multiscale convergence, the covariance-only dichotomy, and spectral 4D — findings (#777)

The Wave 4 validation experiment of epic #763. Everything else in the epic is
machinery; this ticket is the instrument that turns the machinery into a
measurement, and this report is what the instrument returned.

**The headline is a clean, rigorous negative.** On the unforced joint
Regge–Hodge emergence host, across five sizes and five seeds, no quark
candidate is certified at any size or any seed, no derived transport is
accepted anywhere, and no rank-three colour band exists at any point of a
six-decade threshold ladder. The covariance-only dichotomy is therefore
**not reached**: it is `inconclusive`, and the distribution of which
certificate fails first — plus a second, independent obstruction found here —
is the result. The recursive response construction **does not reach the
near-four-dimensional regime** at accessible sizes: the peak heat-kernel
spectral dimension rises monotonically with size but extrapolates to
2.84 ± 0.15 against the pinned 4.245 ± 0.024 baseline. The conjectured
stationarity–defect correlation is **not detected**: r = 0.024, 95 % CI
[−0.375, 0.415] over 25 points.

Nothing here was tuned toward a verdict. Every threshold is a shipped library
default or a value declared in the driver's `DECLARED_*` block before any
datum was examined, and it is identical at every size and every seed. No seed
was dropped.

## What it affects

**Dynamics** — nothing. The driver selects
`MultiCobordism::SimulationMode::Emergence` with
`EmergenceSubmode::Strict` and the merged `JointStationarity` objective, both
unmodified. The #776 firewall makes that structural: `objectiveOf` is a static
function of five declared scalars, and the checkpoints of every run in this
study record `carried_state_energy = 0.0` at weight `0.0`.
**Readout and orchestration** — everything else.

One engine defect was found and fixed (§8).

## 1. How to run it

```
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  BLIS_NUM_THREADS=8 .venv-build/bin/python \
  examples/cobordism/multiscale_validation.py [--quick] --out results.json
```

| mode | ensemble | measured wall time (16-core box, 8 threads) |
|---|---|---|
| `--quick` | 3 sizes × 2 seeds = 6 members | **30.1 s** |
| full (default) | 5 sizes × 5 seeds = 25 members | **246 s** (4.1 min), 9.5 ± 7.7 s per member |

Both modes run every negative control, every analytic invariant, the
threshold-sensitivity ladder and the cold-replay check. The full mode is
cheap enough that there was no reason to report quick-scale numbers: **every
number in this document comes from the full 25-member run**, config hash
`040e208c420f70f2328ab7b1d6235f76`, commit `e1f0fa3`.

## 2. The ensemble, exactly

Refined closed-S⁴ hosts (∂Δ⁵ plus `n_refine` PreGeometric stellar Pachner
adds, then the standard mild non-uniform metric), host seed 3 held fixed so
"size" varies alone:

| size (`n_refine`) | top cells | edges | vertices |
|---:|---:|---:|---:|
| 6 | 30 | 45 | 12 |
| 12 | 54 | 75 | 18 |
| 20 | 86 | 115 | 26 |
| 30 | 126 | 165 | 36 |
| 44 | 182 | 235 | 50 |

Node seeds {7, 11, 13, 17, 19}. Hodge degree {1}. Analysis modularity
resolution γ = 1 (Newman–Girvan, a literature default declared as such);
resolution scan {0.5, 1, 2}. Drive: **one** stage-1 combinatorial update plus
a 12-iteration stage-2 relaxation — the engine's deterministic unit, chosen
because #579/#776 measured that the move draw is not process-deterministic
past the first committed move.

**Reproducibility of a plotted point.** Every emitted point carries its
`config_hash`, `commit`, `size` and `seed`, and embeds its schema-3
checkpoint, so `MultiCobordism.replay_checkpoint` reproduces it exactly from
the record alone (§7). A fresh rebuild from (config, seed, commit) reproduces
the first committed move and the whole relaxation; past that the engine's
non-determinism is a property of the engine, is stated in the output under
`reproducibility`, and is not papered over.

## 3. What converges, what does not

Every dimensionless observable carries a `y = y_∞ + c/N` fit over the size
means with standard errors, and a verdict that describes the FIT and nothing
more. `converging` requires the 1/N coefficient to be separated from zero by
two standard errors AND R² > 0.9; `trending_but_not_inverse_size` means the
coefficient is resolved but a 1/N law does not explain the spread, so the
extrapolation is not to be trusted. Three to five sizes cannot prove a
continuum limit and this study never claims one.

| observable | 6 | 12 | 20 | 30 | 44 | verdict | y_∞ |
|---|---:|---:|---:|---:|---:|---|---|
| accepted band fraction | 1 | 1 | 1 | 1 | 1 | `exactly_constant` | 1 |
| **rank-3 accepted bands** | **0** | **0** | **0** | **0** | **0** | `exactly_constant` | **0** |
| **accepted transports** | **0** | **0** | **0** | **0** | **0** | `exactly_constant` | **0** |
| **transport leakage (min)** | **1** | **1** | **1** | **1** | **1** | `exactly_constant` | **1** |
| **certified quark fraction** | **0** | **0** | **0** | **0** | **0** | `exactly_constant` | **0** |
| response coverage residual | 0 | 0 | 0 | 0 | 0 | `exactly_constant` | 0 |
| components | 2.0 | 2.8 | 3.6 | 4.0 | 4.0 | `converging` (R²=0.956) | 4.32 ± 0.16 |
| modularity Q (γ=1) | 0.092 | 0.162 | 0.234 | 0.293 | 0.368 | `trending_but_not_inverse_size` (R²=0.851) | 0.351 ± 0.037 |
| mean component volume | 5.8 | 6.6 | 7.4 | 9.0 | 12.5 | `trending_but_not_inverse_size` (R²=0.581) | 10.7 ± 1.5 |
| mean component conductance | 0.529 | 0.489 | 0.495 | 0.467 | 0.370 | `flat_within_uncertainty` | 0.416 ± 0.036 |
| hierarchy max depth | 1.6 | 1.4 | 2.0 | 1.8 | 2.0 | `flat_within_uncertainty` | 1.96 ± 0.18 |
| joint stationarity residual | 16.0 | 43.7 | 107 | 213 | 332 | `trending_but_not_inverse_size` (R²=0.649) | — |
| **peak spectral dimension** | **1.906** | **2.080** | **2.393** | **2.684** | **2.885** | `trending_but_not_inverse_size` (R²=0.812) | **2.839 ± 0.154** |
| static solve residual | 0 | 0 | 0 | 0 | 1.2e−16 | `flat_within_uncertainty` | ≤1e−16 |
| vacuum-embedding defect | 3e−19 | 1.2e−16 | 5.0e−17 | 5.5e−17 | 6.7e−17 | `flat_within_uncertainty` | ≤1e−16 |
| covariance purity defect | 1.3e−15 | 1.2e−15 | 7.9e−16 | 1.1e−15 | 1.5e−15 | `flat_within_uncertainty` | ≤2e−15 |
| failed certificates per quark | 11 | 10.87 | 11 | 11 | 11 | `flat_within_uncertainty` | 10.98 |

The shape is stark: **the exact reduction layer is exact at every size, and
every certificate that would identify a particle is identically zero at every
size.** Nothing drifts toward a proton and nothing drifts away from one.

### The one structure that does converge

The component layer is the only part of the construction that behaves like a
converging multiscale object: the component count fits 1/N cleanly
(R² = 0.956, y_∞ = 4.32 ± 0.16) and modularity Q rises steadily with size
(0.092 → 0.368). Hierarchy depth stays at ~2 levels. The persistence tracks
across the declared resolution scan exist and are load-bearing (§4).

## 4. Where the certificates die, in order

`ParticleClusters` evaluates its gates in a fixed order, so
`failedCertificates[0]` is the first failing certificate. **82 quark reads**
over the ensemble:

| pass | first-failing certificate | count | fraction |
|---|---|---:|---:|
| γ = 1 (analysis resolution) | `persistence` | 82 | 1.00 |
| declared resolution scan {0.5,1,2} | `parity-odd` | 19 | 0.543 |
| | `persistence` | 16 | 0.457 |

The γ = 1 column is an **artifact of the measurement, not physics**, and the
study says so in its own output: with a single modularity resolution the
overlay's persistence lifetime is identically 1, so `persistence` is
structurally unpassable. That is why the driver runs the classifier a second
time on the same relaxed geometry with the whole declared resolution scan,
where persistence *is* reachable — and 19 of 35 reads then pass persistence
and die on `parity-odd` instead.

The full failure set is the more informative object. Across the ensemble every
one of these fails on every read at both passes: `anchor`, `color-rank-three`,
`flavor-doublet`, `gauss-consistency`, `occupation-one`, `parity-odd`,
`refinement-stability`, `transport-leakage`, `winding`, `winding-unit`. Zero
quark reads were anchored, zero carried a determinant winding, zero carried a
baryon flux, an isospin or an electric flux, and **zero baryon candidates were
even proposed** at any size or seed, at either pass.

### The three structural reasons

1. **Every certified band is rank 1.** At every size and every seed the
   degree-1 metric Hodge spectrum on a component is non-degenerate, so band
   grouping produces 25–153 rank-1 bands per run and *never* a rank-three
   one. The
   `color-rank-three` gate can therefore never pass, and with it neither can
   the anchor, the colour wedge, the flavour read or the charge read. This is
   not a threshold artifact: §6 shows no point on a six-decade threshold
   ladder produces a rank-3 band.
2. **Every derived transport is rejected.** 204 cross-component transports
   were derived over the ensemble; **all 204** were rejected with the single
   reason `rank-deficient overlap` and leakage exactly 1.0. The derived
   transfer is the off-diagonal Hodge block between two components' cells;
   between rank-1 bands of a vertex-partition's components, that block
   contracts to a numerically zero overlap. With no accepted link there is no
   Wilson loop, no determinant/projective/center holonomy and no determinant
   winding — all four channels report `null` with the reason "no accepted
   derived transport on this complex", never a fabricated value.
3. **The geometry supplies no spin carrier.** `ExchangeHolonomy` requires the
   rotation-loop frame's row count to equal the spinor dimension; an accepted
   band's frame lives on its component's cells, so no emergent band supplies
   one at any size. The physical 2π certificate is `null` with that reason.
   The spin lift is likewise `null`: `spinLift` needs Čech SO(d) edge
   rotations over a cover, and the relaxed complex supplies no tangent-frame
   atlas.

On the DECLARED analytic carriers the same machinery is exact at every size:
χ̂(2π) = −1 on the transverse spinor frame and +1 on the vector control, in
both d = 3 and d = 4, with `exp(2π Σ_ab) = −I` to 1.2e−16.

## 5. What IS exact, at every size

The parts of the construction the whitepaper calls exact are exact, and they
stay exact under refinement. This is the positive half of the result.

* **Static Schur reduction.** Worst solve residual over the ensemble
  2.9e−16, compatibility residual exactly 0, `StructureExact`, certificate
  holds at every size and seed.
* **Shifted (Feshbach) response.** 125 declared frequency windows —
  λ/scale ∈ {0, 0.25, 0.5, 0.75, 1} with half-width 0.1·scale, a
  *dimensionless* grid so it is literally the same grid at every size.
  **125/125 certified**, 0 resonant, worst solve residual 8.6e−16, worst
  compatibility residual 0, worst determinant-factorization residual 1.5e−14.
* **Response network.** Coverage residual **exactly 0** at every size,
  `AlgebraicallyExact`, certificate holds. Stalk dimensions grow with size
  ([6,24,15] → [45,45,39,17,88]) at a roughly constant 4–8 network edges.
* **Inductive-embedding compatibility.** Padding the covariance with empty
  modes changes no Wick amplitude by more than **2.2e−16** anywhere in the
  ensemble. Falsifier 7 does not fire.
* **Quasi-free closure.** Worst covariance purity defect **4.3e−15** over
  every accepted band state at every size.
* **Pauli / grading.** The graded amplitude with a repeated mode is ≤1.0e−17
  everywhere; its ungraded (permanent) counterpart is ≥6.7e−3.
* **Analytic invariants.** All eleven exact at machine precision, worst
  residual **1.9e−15**: `F_3` unitarity and unit-modulus determinant, SU(3)
  invariance of the Λ³C³ singlet Gram, the spin double cover in d = 3 and
  d = 4, the sharp spin-½ fixture (⟨J²⟩ = 3/4 exactly, Var = 1.1e−16), the
  mandated NEGATIVE fixture (⟨J²⟩ = 3/4 with Var = 15/16 — the right
  expectation is not a sharp spin), the composable near-isometry budget
  identity, and the Berry-cancelled single exchange at exactly −1.

### AMLS is unavailable, structurally

`RecursiveQuotient::craigBampton` refuses at **every size and seed**:

> Craig-Bampton refuses the non-normal regime (a self-adjoint solver is never
> applied to a non-self-adjoint operator); use the exact Feshbach pencil
> instead.

The degree-1 metric Hodge operator on a relaxed Lorentzian complex is
`CertificateRegime::NonNormal` with complex eigenvalues, so the certified
AMLS/Craig–Bampton linear surrogate has no domain here and the exact Feshbach
pencil is the only reduction available. That refusal is correct behaviour and
is recorded with its reason rather than forced. The same non-normality is why
the sheaf realization is never emitted: a cellular-sheaf Laplacian is
self-adjoint, so the reduction retains the general response network at every
size (`type: general_response_network`, `emitted: false`).

### The amplitude Gram defect is bimodal, and we know why

The retained-fiber Gram defect is never in between: over 25 runs it is either
≈2.2e−16 (**10 runs**) or ≈2.0 (**15 runs**). The Gram is `G = J†WJ`; an
all-positive weight diagonal makes the embedding an exact isometry, while a
single negative (Krein) weight flips one diagonal entry to −1 and puts
‖G − I‖ at exactly 2. The driver classifies the regime instead of averaging
across it, and the classification agrees with band self-adjointness in
**25/25 runs** (isometric ⟺ the bands came back self-adjoint). So the
observable is not drifting — it is a two-valued function of whether the
relaxed metric left the degree-1 operator in the positive or the Krein
sector, and that sector varies with the seed (self-adjoint band fraction 0.8,
0.2, 0.2, 0.4, 0.4 across sizes, sd ≈ 0.5 at every size). This is ensemble
variance of a discrete geometric property, not finite-size drift of a
continuous one, and it must not be quoted as a mean.

## 6. Separating the four error sources

* **Finite-size drift** — the size-to-size movement of each mean and the 1/N
  slope in §3. Real for components, modularity Q, mean component volume, the
  stationarity residual and the spectral dimension; identically zero for every
  particle certificate.
* **Ensemble variance** — the per-size sample sd over the five seeds, in every
  row of §3. Small for the spectral dimension (≤0.036 at every size, i.e.
  ~1 %) and for modularity Q (≤0.010); large and bimodal for the amplitude
  Gram defect (§5) and for the self-adjoint band fraction.
* **Solver residual** — the certificate residuals of §5, all ≤1.5e−14 and
  most at double round-off. No conclusion in this report is within an order of
  magnitude of a solver residual.
* **Threshold sensitivity** — a declared six-decade ladder on the shipped band
  isolation thresholds (`minRelativeGap`, `gapDominance`), reported as a
  curve. Raising the isolation floor can only ever REDUCE acceptance, so the
  ladder cannot be a search for a verdict:

  | factor | 1e−2 | 1e−1 | 1 | 1e1 | 1e2 | 1e3 | 1e4 | 1e5 | 1e6 |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | accepted / 30 bands | 30 | 30 | 30 | 30 | 30 | 30 | 28 | 2 | 0 |
  | rank-3 accepted | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

  The shipped default sits about four decades below the acceptance boundary,
  so the "every band is accepted" read is insensitive over five decades — and
  **no point on the ladder produces a rank-three band**. The rank-three
  failure is not a threshold choice.

## 7. Replay

All 25 checkpoints replay cold. **Every discrete verdict is identical** —
classification, named failed certificates, band ranks and acceptance,
component ids and supports, labeled-sum ranks, transport verdicts, active
modes, baryon verdicts, and the raw complex. 23 of 25 are byte-identical;
the other 2 differ only in the `hierarchy` block's continuous aggregates
(`strength`, `conductance`, `modularity_contribution`) at a worst relative
difference of **6.8e−15** over the whole ensemble, inside the declared 1e−12
tolerance. That is #776's documented behaviour: the same weights accumulate in
a different order when the checkpoint's edge list is in construction rather
than `fromCells` order. The driver reports the measured difference and where
it occurred, so a real divergence could never hide inside the tolerance.

## 8. Engine defect found and fixed

`RecursiveQuotient::responseNetwork` **segfaulted in a Release build** on an
empty reduction. A partition whose single component covers every cell keeps no
interface coordinate, and a component whose interior block has no kernel
retains no mode, so the reduced operator is 0 × 0 — and `Eigen::maxCoeff` is
undefined at size zero. This is the same failure mode as #776's finding 4
(the empty labeled fiber sum's zero-size `JacobiSVD`), in the sibling method
that the earlier fix did not reach. It now reports the exactly-empty network:
one empty stalk per component, no edges, coverage residual 0,
`AlgebraicallyExact`. Reproduced by a regression test in
`tests/cobordism/test_multiscale_validation.py`.

A second, cosmetic wart is recorded but **not** fixed, because it would widen
a core struct for one consumer: when `sheafRealization` short-circuits on the
non-normal regime it leaves `reconstructionResidual` at its `0.0` default,
which reads as "exact" for a realization that reconstructed nothing. The
driver nulls it when `emitted` is false so its own output cannot mislead; a
future ticket should make the struct default NaN, per the epic's own
"unknown is null, never zero" rule.

## 9. The covariance-only dichotomy

**Outcome: `inconclusive`.** Not `covariance_only_proton`, and not
`quasi_free_sharp_spin_obstruction`. Two independent obstructions block the
branch point, and both are results in their own right.

**Obstruction 1 — the accepted class is empty.** 82 quark reads, 0 certified,
0 baryon candidates, at every size and every seed. There is no accepted
covariance-only class on which to ask whether Var(J²) converges. The
first-failing-certificate distribution of §4 is what the experiment returns
instead, and it is reported as data.

**Obstruction 2 — the geometry supplies no spin structure, so Var(J²) is
dominated by the readout convention.** `CovarianceState::
wickSpinSquaredExpectation` takes CALLER-SUPPLIED one-particle spin matrices,
so a convention has to be named. The driver declares one — J_α = ⊕ σ_α/2 over
consecutive mode pairs in the covariance's own deterministic order — and then
measures how much the answer depends on it by shifting the pairing by one
mode. It depends on it almost entirely:

* 81 spin reads; **57 of them (70 %) have a pairing spread larger than the
  value itself**;
* mean Var(J²) pairing spread 0.050 ± 0.165, maximum **1.47**;
* 38 reads sit on fully paired rank-1 states, where ⟨J²⟩ = 3/4 and
  Var(J²) = 0 are *identities of the readout* — a single fermion in a
  doublet-block operator is trivially a j = ½ eigenstate, so these numbers
  contain no information about the geometry at all;
* the other 43 sit on rank-1 states with one unpaired (spin-0) mode, where the
  very same state becomes a j = ½ ⊕ j = 0 superposition with a genuinely
  nonzero Var(J²) — the design spec §5.12 negative shape, produced here by the
  pairing convention rather than by physics.

**This is the sharpest thing this experiment has to say about the dichotomy.**
Even if a candidate were certified tomorrow, its Var(J²) would not mean
anything until the one-particle spin structure is *derived from the geometry*
rather than declared by the reader. The whitepaper's branch point presumes a
spin operator the ontology has not yet supplied. Until it does, "an exact
covariance-only proton exists" and "a genuinely non-Gaussian interaction is
required" are not yet distinguishable by measurement, and this study will not
pretend otherwise by quoting a convention-dependent number as a physical one.

## 10. The stationarity–defect correlation

Reported as a **conjectural scaling relation, never a theorem** — the
whitepaper explicitly rejects the Hellmann–Feynman/envelope argument, because
the transport Gram defect is not the optimized functional.

Over 25 (size, seed) points:

| pair | statistic |
|---|---|
| joint stationarity residual vs amplitude Gram defect | **r = 0.024**, n = 25, 95 % CI **[−0.375, 0.415]** |
| joint stationarity residual vs transport leakage | **undefined** — leakage is exactly 1.0 at all 25 points, so its variance is exactly zero |
| log₁₀ Gram defect vs log₁₀ residual | slope **4.02 ± 3.26**, intercept −13.9 ± 6.5, **R² = 0.062** |

**No correlation is detected.** The interval on r comfortably contains zero,
and a power law explains 6 % of the spread with a slope whose uncertainty is
80 % of its value. Two caveats keep this from being a strong refutation
rather than a null: the Gram defect is bimodal for the structural reason in
§5, so a Pearson r across it is measuring a discrete sector flip more than a
continuous relation; and the leakage arm has no variance to correlate with
because every transport is rejected identically. The honest statement is that
at these sizes the conjectured relation is **not measurable**, and any future
test of it needs an ensemble in which transports are actually accepted.

## 11. Spectral dimension against the pinned near-4D baseline

The EXISTING estimator is reused verbatim —
`Spacetime::getSpectralDimensionOnSkeleton`, the heat-kernel return
probability on the weighted 1-skeleton of top simplices — over a fixed
20-point geometric σ grid (0.05 × 1.5^i). The definition is not replaced.

| size | peak D_S | sd over 5 seeds |
|---:|---:|---:|
| 6 | 1.906 | 0.036 |
| 12 | 2.080 | 0.008 |
| 20 | 2.393 | 0.006 |
| 30 | 2.684 | 0.010 |
| 44 | 2.885 | 0.013 |

Monotone in size at every step, ensemble variance ~1 %. The 1/N extrapolation
gives **2.839 ± 0.154** with R² = 0.812 — a `trending_but_not_inverse_size`
fit, so the extrapolation is indicative rather than trustworthy, and the
honest reading is that the curve is still rising and has not saturated.

**Verdict: the recursive response construction does not reach the
near-four-dimensional regime at accessible sizes.** The pinned baseline is
peak D_S = 4.245 ± 0.024 at T = 20k with a geometric extrapolation to ≈4.07
(`docs/source/quantum-experiments/overview/h_ds4_status.md`). The gap here is
**1.41 ± 0.16**, some nine baseline sigmas — not a marginal miss.

It does not *destroy* the regime either, and the comparison is deliberately
not like-for-like: the baseline was measured on interaction-history complexes
of 2 500–20 000 events under a β scan, while this study measures the same
estimator on emergence hosts of 30–182 top cells. The correct statement is
that at 182 cells the emergence host is far from four-dimensional and rising,
and that nothing in this measurement contradicts the baseline. What it does
establish is that **the sizes at which the particle certificates are being
evaluated are nowhere near the sizes at which the 4D regime was ever
observed** — which is very likely part of why every certificate fails.

## 12. Unexplained degeneracy

Reported raw over a declared tolerance ladder, on 1868 band eigenvalues:

| relative tolerance | cluster-size histogram | fourfold clusters | largest cluster |
|---|---|---:|---:|
| 1e−9 | {1: 1868} | 0 | 1 |
| 1e−6 | {1: 1868} | 0 | 1 |
| 1e−3 | {1: 1677, 2: 74, 3: 13, 4: 1} | **1** | 4 |

**There is no robust near-fourfold degeneracy.** At any tolerance tight enough
to be meaningful the spectrum is completely non-degenerate; at 1e−3 exactly
one fourfold cluster appears out of 1765, 0.06 % of clusters, which is
consistent with accidental proximity in a dense spectrum. Per the ticket, it
is reported and **not** called Kähler–Dirac taste: naming a mechanism would
need a prediction from the stated one-particle operator, and this study does
not have one. Note that the complete absence of degeneracy is exactly the same
fact as "every certified band is rank 1" (§4) seen from the spectral side.

## 13. Negative controls

All eight fired at the control size. A control that silently passes is a bug
in the instrument, so each is reported with what it measured.

| control | fired | what it showed |
|---|---|---|
| shuffled phases | ✔ | randomizing every complex edge phase moved mean band localization (2.8 % rel) and the mean band gap (51 % rel). **It moved nothing else**: component count, band count, accepted fraction and modularity Q are all exactly unchanged, because `PersistentModularity`'s similarity graph is `w = exp(−|ℓ|)` and therefore phase-blind by construction. The unmoved reads are named in the output. |
| destroyed modularity | ✔ | randomly permuting the discovered partition's labels drove Q from 0.097 to −0.046 |
| modularity resolution limit | ✔ | a ring of 16 4-cliques is merged into 8 components at γ = 1 and resolved into 16 at γ = 4 — the Fortunato–Barthélemy limit, reproduced |
| unanchored rank-three band | ✔ | an empty atlas is refused (`ColorAnchor: an anchor needs at least one declared oriented triangle`), and a rank-3 band whose support misses the declared triangles scores exactly 0 |
| closed spectral / rank gaps | ✔ | forcing the isolation floor past any achievable gap accepts 0 of 30 previously accepted bands |
| cube-root branch change | ✔ | the adjoint PU(3) image is branch-independent to 1.1e−16 while the fundamental representatives differ by 1.38 — distinct center lifts, identical projective observables |
| uncancelled Berry loop | ✔ | the raw exchange-loop determinant is −0.203 + 0.979i, visibly not ±1, while the reference-cancelled character is exactly −1 |
| disabled grading | ✔ | the graded amplitude with a repeated mode is 0.0 (Pauli exclusion) and its ungraded permanent counterpart is 0.210 |

## 14. Whitepaper falsifiers, as measured

Mapping onto the eleven-item falsification programme. "Not decided" means the
construction never got far enough to test the falsifier, which is itself the
finding.

| # | falsifier | measured |
|---:|---|---|
| 1 | no persistent rank-three clusters | **fires.** Zero rank-three bands at every size, seed and threshold. Persistence tracks do exist across the resolution scan and 19/35 reads pass the persistence gate — but never at rank three. |
| 2 | no oriented colour anchor | **fires, downstream of 1.** Zero anchored quark reads; the anchor is unreachable without a rank-three band. |
| 3 | no faithful coarse response | **does NOT fire.** Static reduction exact to 2.9e−16, 125/125 shifted windows certified, coverage residual exactly 0. AMLS has no domain (non-normal) and says so. |
| 4 | no derived gauge covariance | **not decided.** All 204 transports rejected before polar normalization, so no Wilson value, center sector or winding was ever emitted to test. |
| 5 | no fermion holonomy | **not decided** on the emergent carrier; exact (−1) on the declared analytic carrier. |
| 6 | no spinor rotation | **not decided.** χ̂(2π) = −1 exactly on the declared spinor frame in d = 3 and 4; the emergent geometry supplies no spinor carrier and no tangent-frame atlas, so no lift decision is reachable. |
| 7 | no inductive compatibility | **does NOT fire.** Vacuum embedding preserves every amplitude to 2.2e−16. |
| 8 | no quasi-free proton | **not decided.** §9: the accepted class is empty, and Var(J²) is dominated by a declared readout convention rather than by the geometry. |
| 9 | no unforced baryon | **fires so far.** The stationary geometric ensemble produced no baryon candidate at any size or seed without a proton-specific term. This is a statement about 30–182-cell hosts, not about the theory. |
| 10 | no continuum stability | **not decided.** The certificates are identically zero rather than drifting, so there is nothing to converge or diverge. The component layer does converge (R² = 0.956). |
| 11 | unexpected multiplicity | **does NOT fire.** No robust degeneracy at any tolerance (§12). |

## 15. What this changes for the epic

1. **The software returns a rigorous negative, which the design spec §21.4
   names as the completion condition.** One command runs a neutral host under
   a labeled emergence sub-mode, builds the hierarchy, maintains the exact
   covariance state, reports every certificate, distinguishes the verdicts,
   replays cold to the same verdicts, and emits scaling data at five sizes.
   That is what it did.
2. **The rank-three failure is the load-bearing one and is not a threshold.**
   Every colour, flavour, charge and anchor certificate hangs off a rank-three
   band, and none exists at any size or anywhere on a six-decade ladder. Any
   next step that hopes for a quark has to explain where a three-fold spectral
   degeneracy is supposed to come from on a non-degenerate non-normal
   operator. The epic's own answer — hole/window topology carrying b₂ — is not
   what the current emergence host grows.
3. **The transport layer needs overlapping supports.** Deriving transport as
   the off-diagonal Hodge block between the cells of a *vertex partition's*
   components guarantees a rank-deficient overlap. Either the components must
   be allowed to overlap, or the transport must be taken along a path of
   shared cells. As it stands the gauge layer cannot certify by construction,
   independent of the geometry.
4. **The dichotomy needs a derived spin operator before it can be asked.**
   §9's second obstruction is not a scale problem and will not go away with a
   bigger host.
5. **The particle certificates are being evaluated three to four
   dimensions away from the regime the epic assumes.** Peak D_S ≈ 2.9 at 182
   cells against a 4.245 baseline. Before concluding anything about protons
   from an emergence host, the host should be shown to be in the
   near-four-dimensional regime at all.

## 16. Raw results

The machine-readable document (schema version 1) carries the full config, its
hash, the commit, every per-member measurement, every embedded schema-3
checkpoint, every aggregate with its fit and uncertainty, the two named
studies, the negative controls, the analytic invariants, the threshold ladder
and the replay comparison. Reproduce the numbers in this report with:

```
OMP_NUM_THREADS=8 .venv-build/bin/python \
  examples/cobordism/multiscale_validation.py --out full.json
```

at commit `e1f0fa3`; the emitted `config_hash` must be
`040e208c420f70f2328ab7b1d6235f76`. Per the repository convention that issue
artifacts do not live in the repository tree, the raw documents are published
on the `issue-attachments` release rather than committed here:

* [`777-multiscale-validation-full.json.gz`](https://github.com/akellehe/tessera/releases/download/issue-attachments/777-multiscale-validation-full.json.gz)
  — the 25-member run every number above comes from;
* [`777-multiscale-validation-quick.json`](https://github.com/akellehe/tessera/releases/download/issue-attachments/777-multiscale-validation-quick.json)
  — the 6-member `--quick` run, for comparison at reduced scale.
