# Unforced recursive-analysis integration — findings (#776)

The Wave 4 integration ticket of epic #763: run the recursive
component/fiber/transport/covariance/particle analysis alongside
`MultiCobordism` while preserving the unforced joint Regge–Hodge dynamics and
the no-feedback emergence firewall. This report records what the integration
established, positive and negative, in the merge-discipline form of design spec
§23.

## What it affects

**Dynamics** — only through the one declared, labeled `certificates_blind_mean_field`
coupling, which is inactive unless a run selects it. **Readout and orchestration** —
everything else.

## The identity implemented, and its domain

The emergence objective is the design spec's base functional, unchanged and
reused rather than redefined: `MultiCobordism::ObjectiveMode::JointStationarity`
is exactly

$$F_{\mathrm{base}}=\beta_R\|\nabla_zS_{\mathrm{Regge}}\|^2+\eta_H\sum_k\|\nabla_zS_{\mathrm{Hodge},k}\|^2 .$$

The `certificates_blind_mean_field` sub-mode adds exactly one term,
$\beta_E E_{\mathrm{carried}}(\Gamma,g)$, with

$$E_{\mathrm{carried}}(\Gamma,g)=\operatorname{Re}\operatorname{tr}\bigl(\Gamma_S\,h_S(g)\bigr),
\qquad h_S(g)=\tfrac12\bigl(L_k+L_k^{\dagger}\bigr)\Big|_S ,$$

the exact quasi-free expectation $\langle d\Gamma(h)\rangle$ of the one-particle
generator, where $S$ is the set of carried mode cells present in the current
complex and $L_k$ is the metric Hodge operator at the carried degree. Its
gradient is the closed form

$$\frac{\partial E}{\partial z_e}
=\operatorname{Re}\operatorname{tr}\Bigl(\Gamma_S\,\bigl[\partial L_k/\partial z_e\bigr]_S\Bigr)$$

from `HodgeLaplacian::laplacianGradient` — no finite differences. **Domain**:
$k\ge1$ (there is no $\partial L/\partial z$ below degree one); the metric
(symmetric) Hodge operator, whose $\ell^2$ dependence enters only through the
real inner-product weights, so $\partial E/\partial(\operatorname{Im}z)=0$ and
the real-plane ascent displacement is the real derivative alone. Validated
against a central difference at $10^{-6}$ to a worst relative deviation below
$10^{-5}$ — an honest approximate check of an exact closed form, not a
fallback for it.

The same $h_S(g)$ drives `CovarianceState::meanFieldEvolve`, so the objective
coupling and the state's propagation are one functional. $h$ is Hermitian, the
covariance evolves by unitary conjugation, and the measured purity defect stays
below $10^{-9}$ across the schedule: the sub-mode is Gaussian-closed by
construction and the certificate MEASURES that closure.

## The firewall, and how it is enforced

Structurally, not by convention:

* `MultiCobordism::objectiveOf` is a **static** function of `ObjectiveTerms`, a
  record with exactly five named scalars
  (`regge_stationarity`, `hodge_stationarity`, `register_residual`,
  `action_magnitude`, `carried_state_energy`). Having no `this`, it cannot
  consult an analysis member. `objectiveTermNames()` enumerates the list as
  data so a test asserts it.
* `MultiCobordism::refinementDecisionOf` is likewise **static** over two
  `RefinementIndicators` records — five base geometric/numerical quantities
  (`regge_stationarity_residual`, `hodge_stationarity_residual`,
  `curvature_concentration`, `mesh_quality`, `solver_error`). No coarse-response
  residual, band gap, modularity, transport leakage, Wilson/center read,
  exchange read, anchor score, amplitude Gram defect, or particle score is in
  scope at the call site.
* `include/cobordism/MultiCobordism.h` includes no analysis header, and
  `MultiCobordism.cpp` — where the objective and its gradient live — names none
  of the epic's derived types. The overlay's includes are confined to
  `src/cobordism/RecursiveFiberSimulation.cpp`.
* The single channel from the carried quantum state to the geometry is one
  `double`, identically zero outside the labeled backreaction sub-mode.

Behaviourally, adversarially: forcing the band gaps closed, flipping the
overlay to a different degree/resolution configuration so that entirely
different bands, transports and verdicts are produced, selecting the Fock
oracle, and disabling the caches all leave the objective's five terms, the
score assigned to an explicit candidate move, the accepted move, the relaxation
trajectory, and the refinement decision unchanged.

## Negative and unexpected results

**1. The engine's move draw is not reproducible past the first committed move.**
Measured again here (consistent with #579): three identically seeded nodes agree
exactly on the first stage-1 update and diverge on the second. A committed move
rebuilds the complex and the redraw against the rebuilt complex depends on
allocation order. The firewall identity comparisons are therefore made over the
engine's deterministic units — one stage-1 update, the whole stage-2 relaxation
(no draw at all), and the score of an explicit candidate — and the suite pins
the non-reproducibility itself so it cannot be misread as a breach.

**2. The stage-2 trajectory is order-sensitive to read-only Hodge observables.**
Evaluating `HodgeLaplacian::spectrum`, `spectralEntropy`, `MultiCobordism::betti`,
or `hodgeEntropy` BEFORE `ReggeSolver::actionGradientExact` rather than after it
shifts the subsequent relaxation trace by ~$10^{-11}$ relative. A bare
`HodgeLaplacian(st).spectralEntropy(1)` call with no analysis whatsoever
reproduces the shift value for value, so this is a pre-existing engine property
and not a #776 leak. The overlay would nevertheless inherit it, so an analysis
pass now records the objective's own terms FIRST — the order `objective()`
itself uses — which restores bit-identity. Worth a ticket of its own: an
objective that depends on the order of read-only observables is a latent
reproducibility hazard for any campaign that interleaves diagnostics.

**3. A cached transport named its component but not its band (#770).** Every
band of one component restricts to the same cells, so `FiberConnection::fiberKey`
— the fingerprint of that cell-vertex set — is identical across them.
`transportOnSpacetimeCached` keyed on the two component keys alone, so all of a
component pair's band-to-band transports collided on one entry and every one
after the first was served the first's read. The incremental-versus-cold
comparison measured 169 of 170 transports coming back stale. Fixed by folding a
`bandFingerprint` (degree, rank, exact eigenvalue bit patterns) into the
transport and Wilson-loop cache parameters; `toKey`/`fromKey` and the holonomy
chaining rule keep using `fiberKey`, which is the component-level identity they
mean.

**4. An empty labeled fiber sum faulted (#768).** A modularity resolution
coarse enough to put the whole complex in one component leaves no interface
cell to keep, and a component whose interior block has no kernel retains no
mode either, so `labeledFiberSum` reached a $0\times0$ Gram and ran
`JacobiSVD` at size zero — a segmentation fault in a Release build. An empty
sum is a legitimate reduction and is now reported as the exact isometry it is.

**5. Cross-component links are not a lifetime family.** The first integration
fed every cross-component transport into `QuarkCandidateEvidence::lifetimeTransports`.
That field is one candidate's world tube across frames, and one analysis pass
sees one frame. The links belong in `BoundCandidateEvidence::mutualTransports`;
the lifetime family stays unsupplied and the certificate is NAMED as missing.
Besides being the correct reading, this collapsed the pass from $O(\text{bands}^2)$
to $O(\text{candidates}^2)$ derived transports — 663 to 6 at 62 cells, 1.23 s to
0.043 s.

**6. The canonical component hash is not fully label-free (#765).** Under a
global vertex relabeling the discovered PARTITION is exactly covariant — the
supports map through the permutation with no tolerance at all, and every
discrete read (classification, named failed certificates, band ranks, band
acceptance, active modes, component count, labeled-sum ranks, transport count,
baryon verdicts) is identical. The continuous reads agree to double round-off
(declared and measured at $10^{-12}$ relative; the canonical cell enumeration
is reordered, so the same operator entries are summed in a different order).
But the canonical structural hash of a component is NOT invariant for every
component of the closed-S⁴ fixture: the individualization-refinement that
produces it breaks its remaining ties by index. #777 and #778 must not key a
campaign on the hash across a relabeling. The observation is pinned by a test
that skips itself if the hash is ever made fully canonical.

**7. The lazy Slater oracle exists on the positive sector and honestly refuses
the signed one.** The #771 `LazyFockEngine::slaterFromProjector` requires an
orthogonal projector. At degree zero — the positive graph Laplacian — the band
projector is one, and the oracle is built exactly, with one DAG node and zero
discarded norm. At $k\ge1$ the signed-weight operator's band projectors are
oblique (measured $\lVert P^2-P\rVert\approx2.6\times10^{-2}$), so no exact
Slater reference exists and the engine refuses. The checkpoint records
`present: false` with the refusal's own message in `absent_reason` rather than
claiming an oracle it does not have. The quasi-free path is unaffected:
`CovarianceState::fromBandProjector` adopts the oblique projector verbatim and
reports its defect, which is the documented behaviour.

**8. On the current fixtures no candidate is certified.** Every emitted quark
read comes back `classification: "none"` with its gaps NAMED — most often the
missing lifetime-transport family, the missing determinant winding, and an
unanchored band. That is the designed behaviour of a rigorous negative result,
and it is what the ticket asks the software to be able to return.

## Replay identity, exactly

Cold replay rebuilds the raw complex, disables every cache, and recomputes
every derived hierarchy and certificate.

* From a checkpoint taken after at least one accepted move — that is, on any
  optimizer-produced complex, which is always a `Spacetime::fromCells` rebuild —
  the `hierarchy`, `fibers`, `labeled_fiber_sums`, `transports`, `covariance`,
  `particles`, `certificates`, and `raw_complex` blocks are **byte-identical**.
* From a checkpoint taken on a hand-built host, whose edge list is in
  construction order rather than `fromCells` order, the particle and gauge
  **verdicts and the fiber block are still identical** while the modularity
  aggregates differ by double round-off (measured $\le10^{-12}$ relative),
  because the same weights accumulate in a different order.

An unknown `schema_version`, a missing version, malformed JSON, and a
checkpoint without a raw complex are all rejected.

## Benchmark — the analysis cadence

Median of three, `OMP_NUM_THREADS=8`, closed-S⁴ hosts, `JointStationarity`,
one Hodge degree and one modularity resolution:

| refine | cells | edges | stage-1 update, overlay off | stage-1 update, overlay on | pass, cold cache | pass, warm cache | pass, caches disabled | cold replay |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 22 | 35 | 0.0136 s | 0.0236 s | 0.0075 s | 0.0046 s | 0.0074 s | 0.0083 s |
| 8 | 38 | 55 | 0.0270 s | 0.0447 s | 0.0177 s | 0.0107 s | 0.0169 s | 0.0162 s |
| 14 | 62 | 85 | 0.0469 s | 0.0915 s | 0.0414 s | 0.0229 s | 0.0464 s | 0.0802 s |
| 22 | 94 | 125 | 0.0895 s | 0.1577 s | 0.0697 s | 0.0387 s | 0.0740 s | 0.0789 s |

One analysis pass costs about 0.75 of one stage-1 update at every size
measured. A warm cache roughly halves it: on a pure metric change the published
star drops only the entries whose component it meets, and the disjoint siblings
are served. The disabled path is one integer increment and one boolean test per
accepted move — there is no overlay object to construct and no analysis header
in the objective's translation unit.

## What the next tickets consume

#777 and #778 consume the modes (`SimulationMode`, `EmergenceSubmode`), the
checkpoint document, and the static `replayCheckpoint` entry point unchanged.
The checkpoint's `analysis` block carries the cadence, the degrees and
resolutions the pass used, and that pass's own cache activity, so a campaign
can be replayed and costed from the record alone.

The document this integration wrote was schema 3, in which `particles.baryons`
held the §16.2 bound-supercomponent SEARCH records — the overlay never ran the
§16.4 baryon classifier, so no baryon verdict existed to record. #802 closed
that: the overlay now classifies every binding of exactly three certified
constituents, the search records moved to `particles.bound_supercomponents`,
`particles.baryons` carries the `BaryonRead` verdict, and the version is 4. A
schema-3 document is rejected on read rather than reinterpreted, because its
`baryons` entries mean a different thing.
