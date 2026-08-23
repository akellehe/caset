# The complete recursive baryon simulation — findings (#778)

The capstone ticket of epic #763. Everything else in the epic is machinery or
a single measurement; this ticket is the instrument assembled into ONE
documented command that starts from a neutral complex, runs the unforced
joint Regge–Hodge emergence dynamics, runs the recursive post-hoc analysis,
and returns a particle verdict — then replays it, benchmarks it and draws it.

**The headline is a clean, rigorous negative, and that is valid completion.**
Design spec §21.4: *"An unforced proton is a scientific success condition, not
a software completion condition. The software is complete if it can return a
rigorous negative result."* It does. On the neutral closed-S⁴ emergence host,
at every size and seed run here, the verdict is **`no baryon`**, the first
failing certificate is the structural **`constituent-quarks`** gate, and the
reason is named at every level: zero certified quark candidates, because zero
rank-three colour bands exist, because every band of the degree-one metric
Hodge operator on a component is non-degenerate.

Nothing here was tuned toward a verdict. Every threshold is a shipped library
default or a value declared in the driver's `DECLARED_*` block before any
datum was examined, and it is identical at every size and every seed. No
member was dropped. The verdict string is
`ParticleClusters::classifyBaryon`'s own `classification`, relayed verbatim.

## What it affects

**Dynamics** — nothing. The driver selects
`MultiCobordism::SimulationMode::Emergence` with `EmergenceSubmode::Strict`
and the merged `JointStationarity` objective, both unmodified, and #776's
firewall makes that structural: `objectiveOf` is a *static* function of five
declared scalars, `refinementDecisionOf` is a *static* function of five
declared geometric indicators, and every checkpoint of every run in this
report records `carried_state_energy = 0.0` at weight `0.0`. The optional
refinement is the #776 rule unchanged, driven by the two DECLARED
dimensionless mesh-quality thresholds below.
**Readout and orchestration** — everything else.

No engine defect was found. Two reporting decisions were changed (§9).

## 1. How to run it

```
OMP_NUM_THREADS=8 .venv-build/bin/python \
  examples/cobordism/recursive_baryon_simulation.py run --out run.json

OMP_NUM_THREADS=8 .venv-build/bin/python \
  examples/cobordism/recursive_baryon_simulation.py replay --from run.json

OMP_NUM_THREADS=8 .venv-build/bin/python \
  examples/cobordism/recursive_baryon_simulation.py campaign --out camp.json

OMP_NUM_THREADS=8 .venv-build/bin/python \
  examples/cobordism/recursive_baryon_simulation.py animate \
  --from run.json --out overlay.png
```

| command | what it does | MEASURED wall time (32-core box, 8 threads) |
|---|---|---|
| `run` (fast default) | size 12, 2 drive steps, 54 top cells | **7.40 s** (median of three: 7.38 / 7.40 / 7.51) |
| `run --animate overlay.gif` | the same plus the 2-frame overlay | **11.0 s** |
| `run --large` | size 30, 4 drive steps, 122 top cells | **52 s** |
| `replay --from run.json` | cold-cache replay of both frames | **0.10 s** (median of three) |
| `campaign` (declared default) | 3 sizes × 2 seeds = 6 members | **47 s** |
| `campaign --sizes 6,12,20,30 --seeds 7,11,13` | 4 sizes × 3 seeds = 12 members | **155 s** |
| `fixtures` | the eleven exactness fixtures alone | **0.15 s** (median of three) |

`run` exits 0 whether or not a proton emerges — the exit code reports whether
the SOFTWARE ran, never whether the physics obliged. `replay` exits non-zero
only when a stored verdict or content hash fails to reproduce.

Every number in §§2–8 comes from the fast-default run at commit `e7456f9`,
config hash `5e7ada57efad6aa2aea6c9a1855cf088`; §7 comes from the 12-member
campaign at the same commit.

## 2. The neutral initial complex, exactly

The bare boundary of a 5-simplex — a combinatorial closed S⁴, the smallest
closed 4-manifold triangulation — refined by `--size` PreGeometric stellar
Pachner adds at host seed 3, then given the mild deterministic non-uniform
metric ℓ² = 1 + 0.01·(index mod 6). It is byte-identical in construction to
`tests/cobordism/_closed_s4.py` and to #777's host, so this run's numbers are
directly comparable with that study's ensemble; a test asserts the two
drivers build the same cells.

NEUTRAL means every structure the epic looks for is absent by construction:
**no holes, no colour windows, no pinned carrier, no boundary blocks, no
target register, no proton-specific term.** `inputs()` and `outputs()` are
empty. That is the point: a proton must EMERGE or not, and the software must
be able to say which.

## 3. The verdict, and why

**`no baryon`.** `classification: "no-baryon"`, confidence 0.0714 (one of
fourteen gates passed — a passed-gate fraction, not a probability),
certificate grade `HeuristicDiscovery`, `holds = false`.

The four verdicts are the design spec's, and the classifier produces them
from ONE code path: the same evidence bundle is assembled whatever the run
produced — possibly nothing — and handed to `classifyBaryon`, which names
every missing or failed certificate. There is no target-dependent branch and
no target-dependent success string; a test asserts the vocabulary is exactly

> `no baryon` | `baryon candidate` | `certified proton` |
> `quasi-free sharp-spin obstruction`

**Thirteen of fourteen baryon gates failed**, in the classifier's fixed
order, with `constituent-quarks` first:

| gate | outcome |
|---|---|
| `constituent-quarks` | **first failure** — 0 of 3 constituents are certified quark reads |
| `bound-supercomponent` | no binding was emitted |
| `color-singlet`, `color-flux-zero` | colour columns UNSUPPLIED (no rank-3 anchored band) |
| `baryon-flux-unit`, `composite-parity-odd`, `flavor-uud`, `electric-flux-unit` | every leg uncertified |
| `spin-expectation`, `sharp-spin` | no covariance state of a certified bound composite |
| `rotation-character` | the emergent geometry supplies no spinor carrier |
| `finite-radius`, `profile-stability` | the neutral host grows no register hole, so the existing mass-radius battery has no shell seed |
| `spin-lift` | NOT demanded — no continuum spin claim was declared, so the lift is not applicable |

Each of those is emitted with a NAMED reason in the run document's
`verdict.missing_evidence`, and the animation prints them verbatim.

### The chain of reasons, from the bottom

1. **Every accepted band is rank 1.** 44 bands enumerated on 3 components, 44
   accepted, rank histogram `{1: 44}`, **rank-three accepted: 0**. The
   degree-one metric Hodge spectrum on a component is non-degenerate, so band
   grouping never produces a rank-three one. This reproduces #777 §4 finding
   1 exactly, at a different commit and through a different assembly path.
2. **No rank-three band ⟹ no anchor, no colour wedge, no flavour, no
   charge.** 0 of 3 quark reads are anchored; `triangle_anchor_score` is
   `null`, never 0.
3. **Every derived transport is rejected.** 6 derived, **0 accepted**,
   leakage exactly 1.0, single reason `rank-deficient overlap` ×6. With no
   accepted link there is no Wilson value, no determinant, no projective, no
   center read and no winding: all five channels report `null` with the same
   named reason, never a fabricated value.
4. **No emergent spinor carrier.** `ExchangeHolonomy` needs the rotation-loop
   frame's row count to equal the spinor dimension; an accepted band's frame
   lives on its component's cells, so none qualifies. The spin lift is
   likewise `null` — `spinLift` needs Čech SO(d) edge rotations over a cover
   and the relaxed complex supplies no tangent-frame atlas.
5. **No certified quark ⟹ no binding.** `boundSupercomponentSearch` counts
   only CERTIFIED quark candidates toward a supercomponent's membership. It
   examined 3 next-level components against 3 quark reads and emitted no
   binding.

The quark layer's own first failures at the analysis resolution are
`persistence ×3` — and the driver says in its own output that this is an
**artifact of the measurement, not physics**: with a single modularity
resolution the persistence lifetime is identically 1, so the gate is
structurally unpassable. That is why the run ALSO checkpoints a pass over the
whole declared resolution scan {0.5, 1, 2}, where persistence *is* reachable;
there the first failure moves to `parity-odd`, and the full failure set is
unchanged. Both passes travel in the document (`particles` and
`particles_resolution_scan`) so neither can be mistaken for the other.

## 4. What IS exact — the positive half

Eleven fixtures, each against an analytic or dense reference computed
independently in the driver (and independently again in the test suite).
Worst residual over all eleven: **2.8 × 10⁻¹⁶**.

| fixture | residual | reference |
|---|---:|---|
| static Schur (Kron) reduction on a path | **0** | `L_BB − L_BI L_II⁻¹ L_IB`, computed here |
| shifted Feshbach–Schur pencil | 1.1e−16 | `(L_BB − λI) − L_BI (L_II − λI)⁻¹ L_IB` |
| static Schur does NOT preserve the pencil | resolved | eigenvalue separation, required nonzero |
| second-quantized subset sum | **0** | `itertools.combinations` sums |
| second-quantized hopping block | **0** | dense block assembly + `eigvalsh` |
| triangle anchor | **0** | `\|det A\|² = 1` at full concentration; disjoint atlas scores exactly 0; post-hoc reweighting and an empty atlas both refused |
| center branch | 2.8e−16 | `Tr H̃(s) = Tr H̃(0)·ω^{−s}`; `Ad(ωU) = Ad(U)`; one shared sector; fundamental representatives differ by 1.4 |
| closed determinant winding | **0** | the declared single turn; ν = 1 exactly, closure `closed-family` |
| Berry cancellation | **0** | matched single/double exchange = −1 / +1 while the raw loop determinant is −0.203 + 0.979i, visibly not a sign |
| sharp spin-½ | 1.1e−16 | ⟨J²⟩ = 3/4 with Var = 0 |
| generic Slater is NOT a sharp spin | 2.2e−16 | ⟨J²⟩ = 3/4 with Var = 15/16 (design spec §5.12) |

And on the emergent geometry, at every size measured:

* **Static reduction** — solve residual **exactly 0**, compatibility residual
  exactly 0, `StructureExact`, certificate holds, 75 kept coordinates.
* **Shifted (Feshbach) response** — **5/5 declared windows certified**, 0
  resonant, worst solve residual **0**, worst determinant-factorization
  residual **0**. The grid is dimensionless (λ/scale ∈ {0, ¼, ½, ¾, 1} with
  half-width 0.1·scale), so it is literally the same grid at every size.
* **Response network** — coverage residual **exactly 0**,
  `AlgebraicallyExact`, certificate holds; stalk dimensions [15, 28, 1, 31]
  over 6 edges.
* **Quasi-free closure** — worst covariance purity defect **1.7 × 10⁻¹⁶**
  over every accepted band state.
* **Inductive compatibility** — padding the covariance with empty modes
  changes no Wick amplitude by more than **2.6 × 10⁻¹⁶**. Falsifier 7 does
  not fire.
* **Spin double cover** — χ̂(2π) = −1 exactly on the declared transverse
  spinor frame and +1 on the vector control, in both d = 3 and d = 4, with
  `exp(2π Σ_ab) = −I` to **1.2 × 10⁻¹⁶**.

### Two structural refusals, correctly reported

`RecursiveQuotient::craigBampton` refuses at every size:

> Craig-Bampton refuses the non-normal regime (a self-adjoint solver is never
> applied to a non-self-adjoint operator); use the exact Feshbach pencil
> instead.

The degree-one metric Hodge operator on a relaxed Lorentzian complex is
`CertificateRegime::NonNormal`, so the certified AMLS surrogate has no domain
and the exact Feshbach pencil is the only reduction available. The same
non-normality is why `sheafRealization` is never emitted — a cellular-sheaf
Laplacian is self-adjoint — so the general response network is retained
(`type: general_response_network`, `emitted: false`, reconstruction residual
`null` rather than the struct's misleading 0.0 default).

Both are **refusals, not failures**, and the certificate ledger says so:
of 42 certificate entries, **21 held, 20 were refused with a named reason,
and exactly 1 failed** (`derived-transport`, which WAS evaluated and did not
certify). Conflating those three would have made a correct out-of-domain
refusal look like a broken certificate.

### The labeled-fiber-sum Gram defect is bimodal, and we say which mode

`‖G − I‖ = 1.967` on this run — the **`signature_flipped`** regime. The Gram
is `G = J†WJ`; an all-positive weight diagonal makes the embedding an exact
isometry (defect ~1e−16), while a single negative (Krein) weight flips one
diagonal entry to −1 and puts the defect at exactly 2. #777 §5 measured this
as a two-valued function of whether the relaxed metric left the degree-one
operator in the positive or the Krein sector. The driver classifies the
regime rather than quoting a number that is a mean of two unrelated values.

## 5. Persistence, exactly

Design spec §20's schema-3 checkpoint is embedded verbatim, once per drive
step, and the run document (schema version 1) adds the layers the checkpoint
does not carry. Every block is content-hashed.

| layer | where | note |
|---|---|---|
| raw geometry | `raw_geometry` | top cells in intrinsic vertex order, edges in canonical endpoint order with complex lengths; rebuilds through `Spacetime.fromCells` |
| edge-mode data | `edge_mode_data` | the cells each carried band lives on, its eigenvalues and weight diagonal |
| covariance state Γ | `covariance` | one Γ per accepted band, through the sidecar; purity, occupation, parity, vacuum-embedding defect |
| Fock state / DAG | `fock` | ORACLE / non-Gaussian path only; absent by default with the reason named |
| response hierarchy | `response_hierarchy` | static, 5 shifted windows, AMLS, response network with stalks/edges, realization, labeled fiber sum |
| fibers and signatures | `fibers` | every band's rank, gaps, residuals, Krein signature; the candidate band's projector through the sidecar |
| transports | `transports` | every derived link with leakage, conditioning, Krein sectors and rejection reason, plus the determinant / projective / center / winding channels |
| particles | `particles`, `particles_resolution_scan` | quark reads, bound supercomponents, and the agreement with the C++ overlay's own reads |
| certificates | `certificates` | every certificate with grade, status and residual |
| provenance | `provenance` | seed, host seed, config hash, commit, Python, platform, thread count |

**The binary sidecar** (design spec §20) is one uncompressed `.npz` — NPY
format 1.0, a versioned container, not an ad-hoc blob. A matrix of more than
64 entries goes to the container; smaller ones inline as split real/imaginary
lists so a small run needs no sidecar file at all. Every array carries the
SHA-256 of its own C-contiguous bytes and the container carries its own
SHA-256, so a replay verifies both. This run: 4 arrays (2 covariances, 2
candidate-band projectors), container hash `d8b04b91…`, document 207 KB.

**The driver's reads agree with the C++ overlay's.** The run cross-checks its
Python-side quark classifications and failed-certificate sets against the
checkpoint's and records the agreement; on this run both match exactly. A
divergence could not hide.

## 6. Replay

```
REPLAY VERIFIED
  frames          : 2 of 2 discrete-identical, 2 byte-identical
  worst continuous: 0 against tolerance 1e-12
  content hashes  : 17 of 17 matched
  sidecar         : file hash True, 6 of 6 arrays verified
  fixtures        : 11 of 11 exact on both paths
  verdict         : stored 'no baryon', replayed 'no baryon', match True
  runtime         : 0.11 s
```

Every frame is rebuilt through `MultiCobordism::replayCheckpoint`, which
disables every cache and recomputes every derived hierarchy and certificate.
Thirteen checkpoint blocks are compared — `schema_version`,
`emergence_submode`, `raw_complex`, `edge_quantum_data`, `objective`,
`hierarchy`, `fibers`, `labeled_fiber_sums`, `transports`, `covariance`,
`fock_oracle`, `particles`, `certificates` — and both frames came back
**byte-identical**, worst relative difference exactly 0. Every DISCRETE
verdict must match with no tolerance at all; only continuous aggregates get
the declared 1e−12 budget, and the measured difference is always reported.

Six blocks are deliberately NOT compared, each with the reason recorded in
the report so a reader sees the exclusions rather than inferring them:

| block | why a replay cannot reproduce it |
|---|---|
| `mode` | the replayed document is stamped `"replay"` by design |
| `geometry_revision` | the metric revision key counts THIS process's metric writes |
| `refinement` | `solver_error` is the magnitude of the last accepted stage-2 improvement, and a replay never relaxed |
| `analysis` | the pass and cache counters are this process's, and a replay runs cold by design |
| `provenance` | the replay stamps its own |
| `invalidated_ancestry` | invalidation is relative to the accepted move a replay did not make |

Beyond the checkpoints, the replay independently verifies: all 17 block
content hashes; the sidecar file hash and every array's own hash; the eleven
exactness fixtures recomputed cold; and — the strongest statement available —
the VERDICT itself, recomputed from the persisted raw geometry rebuilt
through `Spacetime.fromCells`, together with its failed-certificate list.

A corrupted matrix hash, a tampered verdict, an unknown or missing
`schema_version`, an unknown checkpoint version and malformed JSON are all
caught; tests assert each.

## 7. The campaign and its scaling

Four sizes × three seeds = **12 members, 12 of 12 ran, none dropped**;
160 s total, at commit `4d75a4f`. Silently omitting a failed seed is explicitly out of scope, so
a failing member is recorded with its error and counted.

| size | top cells | wall s | drive s | analysis s | readout s | RSS MiB | peak RSS MiB | cache hits/misses | invalidations | entries | components | bands | rank-3 accepted | accepted transports | peak D_S |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 28.3 | 3.34 | 3.32 | 0.0067 | 0.0041 | 69.5 | 68.0 | 9.0/21.0 | 15.0 | 12.0 | 2.00 | 28.3 | **0** | **0** | 1.910 |
| 12 | 52.3 | 6.95 | 6.87 | 0.0145 | 0.0160 | 74.2 | 74.6 | 16.0/35.0 | 27.3 | 19.0 | 3.00 | 42.7 | **0** | **0** | 2.057 |
| 20 | 84.7 | 14.11 | 13.90 | 0.0343 | 0.0365 | 83.0 | 86.8 | 16.0/44.0 | 38.7 | 19.0 | 3.00 | 65.3 | **0** | **0** | 2.369 |
| 30 | 124.7 | 28.99 | 28.70 | 0.0620 | 0.0817 | 98.6 | 108.7 | 25.0/59.0 | 50.0 | 34.0 | 4.00 | 96.0 | **0** | **0** | 2.667 |

Scaling against top cells, over the four sizes (log–log OLS, uncertainty
from the fit):

| quantity | exponent | R² |
|---|---|---|
| wall seconds | **1.44 ± 0.10** | 0.990 |
| drive seconds | **1.44 ± 0.10** | 0.990 |
| analysis seconds | **1.52 ± 0.08** | 0.995 |
| Python readout seconds | **2.00 ± 0.07** | 0.997 |

The cost is the DRIVE — the stage-2 relaxation — at every size: the analysis
pass is **0.2 %** of the member. Memory grows from 70 to 99 MiB across a
4.4× growth in cells; the report distinguishes the process RSS at a member's
end, the process peak so far (monotone, hence never a per-member cost), and
the per-member RSS delta, and says so in its own output.

Cache activity is aggregated over EVERY analysis pass a member ran, not just
the last: the last pass is warm by construction (two passes already ran on
the same complex), so quoting it alone would report a hit rate the run did
not pay for. The hit fraction is flat at 0.27–0.31 across the whole range
while misses, invalidations and live entries grow roughly linearly in cells —
the published-star invalidation is local, and disjoint siblings are served.

**Every one of the 12 members returned `no baryon` with
`constituent-quarks` first**, 0 rank-three bands and 0 accepted transports at
every size and seed, transport leakage exactly 1.0 at all 12 points, and all
eleven exactness fixtures exact in every member. Nothing drifts toward a
proton and nothing drifts away from one; the particle certificates are
identically zero rather than converging or diverging.

Peak spectral dimension rises monotonically with size (1.910 → 2.667 at 125
cells) against the pinned 4.245 ± 0.024 baseline
(`docs/source/quantum-experiments/overview/h_ds4_status.md`), reproducing
#777 §11 on the same estimator (`Spacetime::getSpectralDimensionOnSkeleton`,
reused verbatim). **The particle certificates are being evaluated one to two
spectral dimensions away from the regime the epic assumes.**

## 8. What the animation shows

`animate` renders eight panels from the PERSISTED run document — the same
object the headless path emits and the replay verifies — so the two can never
disagree. A test asserts the overlay is a pure function of that document.

1. **component world tubes** over the drawing layout, coloured by component,
   with Q, level count and next-level component count;
2. **response vertices and stalks** at component centroids, sized by stalk
   dimension, joined by the response-network edges, with the coverage
   residual and the realization type; a stalk beyond the discovered
   components (the reduction covers every cell, so an uncovered remainder
   gets its own) is drawn on an outer ring and named as such;
3. **fibers** — the rank spectrum, enumerated vs accepted, with ranks 1–3
   always on the axis so a MISSING rank three is a visible empty slot rather
   than an axis that never mentions it;
4. **derived transports** — accepted solid, rejected dashed, with the
   rejection-reason histogram;
5. **holonomy channels** — determinant on the unit circle, the three center
   branches with their sector, and the winding;
6. **Berry-cancelled exchange and rotation characters** — the exact values on
   the declared analytic carriers beside the emergent carrier's status;
7. **certificate failures** — the baryon gates in classifier order with the
   FIRST failure highlighted, and the quark first-failure histogram;
8. **the verdict** and every named missing-evidence reason.

**On this host, five of the eight panels have nothing positive to draw.**
That was the design constraint the ticket named, and it is met: each such
panel says **ABSENT** in red and prints the named reason underneath, so a
reader sees *what* is missing and *why* instead of an empty frame. The figure
footer states that the layout is a classical-MDS DRAWING of the 1-skeleton,
not a spacetime coordinate system.

`--all-frames` (or a `.gif`/`.mp4` output) renders every persisted
checkpoint, so a longer drive animates with no extra data source.

## 9. Reporting decisions, stated

Two things this ticket chose to report differently from the obvious way, both
because the obvious way would mislead:

1. **A domain refusal is not a certificate failure.** AMLS on a non-normal
   operator, an unemitted sheaf realization, and a holonomy channel with no
   accepted link are `status: "refused"` with the reason named, not counted
   among the certificates that were evaluated and failed. On the fast run
   that is 20 refusals against exactly 1 real failure; reporting 21 failures
   would have hidden the one that matters.
2. **The bimodal Gram defect is classified, not averaged.** See §4.

One warning inherited from #777 §8 remains unfixed and is repeated here: when
`sheafRealization` short-circuits on the non-normal regime it leaves
`reconstructionResidual` at its `0.0` default, which reads as "exact" for a
realization that reconstructed nothing. Both drivers null it when `emitted`
is false so their own output cannot mislead; the struct should default to NaN
per the epic's "unknown is null, never zero" rule. That is a core-struct
change for one consumer and belongs in its own ticket.

## 10. Against the design spec's ten end-to-end capabilities

Design spec §21.4 — the epic is complete when one command can:

| # | capability | met |
|---:|---|---|
| 1 | start from a documented neutral initial complex and seed | ✔ §2 |
| 2 | run either labeled emergence sub-mode with particle-blind refinement | ✔ `--submode` selects either; refinement is #776's static rule over five geometric indicators. NOTE: this driver adopts no carried state, so even under `certificates-blind-mean-field` the one permitted coupling is exactly 0 and the run says so in its own `firewall.mean_field_note`. #776 owns the mean-field schedule |
| 3 | build and persist the recursive component hierarchy | ✔ §5 |
| 4 | maintain the exact covariance state on the quasi-free path, using a certified Fock DAG only for oracle / non-Gaussian boundary data | ✔ purity defect 1.7e−16; `--fock-oracle` builds the DAG exactly at degree 0 (1 node, 0 discarded norm) and reports the engine's OWN refusal at k ≥ 1 |
| 5 | report all quark, gauge, exchange and baryon certificates | ✔ §3, §4 |
| 6 | distinguish the four verdicts without a target-dependent code path | ✔ §3 |
| 7 | replay the checkpoint with cold caches and reproduce the verdict | ✔ §6 |
| 8 | render the hierarchy, colour transport, Wilson loops and particle world tubes | ✔ §8, with absences named |
| 9 | emit scaling data for at least three problem sizes | ✔ §7, four sizes |
| 10 | keep the analytic/structured path faster than the dense reference on the crossover fixture while agreeing within certificate | inherited from #764's benchmark; **not re-measured here** |

Item 10 is #764's contract and this ticket did not re-measure it. Everything
else is measured above.

## 11. What this leaves for a next step

Nothing in this run contradicts #777's five conclusions; it reproduces them
through an independent assembly path and adds one:

1. **The rank-three failure is still the load-bearing one.** Every colour,
   flavour, charge and anchor certificate hangs off a rank-three band, and
   none exists at any size here. Any next step that hopes for a quark has to
   explain where a three-fold spectral degeneracy is supposed to come from on
   a non-degenerate non-normal operator.
2. **The transport layer still cannot certify by construction.** Deriving
   transport as the off-diagonal Hodge block between the cells of a *vertex
   partition's* components guarantees a rank-deficient overlap. Either the
   components must be allowed to overlap, or the transport must be taken
   along a path of shared cells.
3. **The bound-supercomponent search is downstream of certification.** It
   counts only CERTIFIED quark candidates, so with zero certified quarks it
   emits no binding at all — which is why `constituent-quarks` rather than
   `persistence` is the baryon layer's first failure. The two layers fail for
   the same root reason at different depths.
4. **The dichotomy still needs a derived spin operator.** #777 §9's second
   obstruction is not a scale problem: `wickSpinSquaredExpectation` takes
   CALLER-SUPPLIED one-particle spin matrices, so until the ontology derives
   them, ⟨J²⟩ and Var(J²) are readout conventions. This driver therefore
   supplies no spin read for an uncertified composite and NAMES the gap
   rather than quoting a convention-dependent number as a physical one.
5. **The host is nowhere near four-dimensional.** Peak D_S ≈ 2.67 at 125
   cells against a 4.245 baseline, still rising.
6. **New here: the analysis is not the cost.** The recursive analysis pass is
   0.2 % of a member's wall time and scales as cells^1.52 ± 0.08, while the
   stage-2 relaxation is 99 % and scales as cells^1.44 ± 0.10 — the same
   exponent within uncertainty. Whatever limits the accessible size, it is the
   emergence dynamics, not the recursive readout: the readout is already
   ~500× cheaper and does not scale worse. A bigger host is a relaxation
   problem.

## 12. Raw results

The machine-readable documents (run schema version 1) carry the full config,
its hash, the commit, every embedded schema-3 checkpoint, every persisted
layer, the certificate ledger, the exactness fixtures and the verdict with
its named reasons. Reproduce every number above with the commands in §1 at
commit `e7456f9`; the fast run's emitted `config_hash` must be
`5e7ada57efad6aa2aea6c9a1855cf088`.

Per the repository convention that issue artifacts do not live in the
repository tree, raw documents belong on the `issue-attachments` release
rather than committed here.
