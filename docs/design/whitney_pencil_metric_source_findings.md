# MultiCobordism on the chain-level Whitney pencil: findings

Ticket #910 of epic #905. Every number below was measured on the build of
this branch; nothing was tuned toward an outcome.

## What changed

`HodgeLaplacian` gained a `MetricSource`: `DiagonalWeights` (the historical
per-simplex diagonal weights of the `WeightConvention`, unchanged) or
`WhitneyPencil` (the chain-level Whitney Hodge pencil of `tessera.chainhodge`,
dressed at every degree by the C* links read from the edge phases). Under
`WhitneyPencil`, `laplacian(k)` is the dense covariant operator `h_k(s,U)` on
chains, `laplacianGradient` its analytic `∂h_k/∂s_e`, and the new
`laplacianPhaseGradient` its analytic `∂h_k/∂φ_e`. `EigenstateSynthesis`
and `MultiCobordism` carry the source and forward it to every operator they
build; the static readouts take it as a trailing argument; `Proton` passes
its node's source to its color readouts.

**One knob.** The source defaults to the process-wide
`HodgeLaplacian.defaultMetricSource()`, read at construction (never captured
by a Python default), exactly as the weight convention is. It ships as
`DiagonalWeights`, so every historical consumer is bit-identical; an
experiment on the pencil flips it once at startup with
`HodgeLaplacian.setDefaultMetricSource(WhitneyPencil)`, and then a node, its
static readouts, the observables, the capstone, and checkpoint replay all
agree. The alternative, a Whitney default on `MultiCobordism` alone while the
statics and observables followed the process default, produced five
composition mismatches in the existing suite (a node's `r_u` compared with a
statically computed `nearKernelResidual` of the same geometry) and would have
left the capstone's run and its replay on different operators.

## Exact checks that pass

- `HodgeLaplacian(st, ·, WhitneyPencil).laplacian(k)` equals
  `CovariantChainHodge.covariantOperator(k)` at every degree on 2- and
  3-complexes, with and without phases; with trivial phases it equals
  `ChainHodge.hodgeOperator(k)`.
- The pencil operator is homogeneous of degree −1 in the squared lengths at
  every degree, `Σ_e s_e ∂L_k/∂s_e = −L_k`, the same degree as the diagonal
  `V²` path, so every Euler identity downstream is unchanged.
- The analytic phase gradient satisfies the exact gauge identity
  `Σ_e (χ_y − χ_x) ∂h/∂φ_e = −i [diag(χ_{b(σ)}), h]` for random χ.
- The analytic operator derivative agrees with central differences to 3e-9
  (both sources; a diagnostic, not a test).
- The register residual `r_U` is homogeneous of degree −2 under the pencil:
  `Σ_e l²_e ∂r_U/∂l²_e = −2 r_U` at relative 1e-10 at k = 1 (the general
  core with the pencil's dense derivative), for both sources.
- The combinatorial operator (`metric=False`) is identical under both sources.

## Stored orientations

Directed cone surgery stores some simplices in non-ascending vertex order,
so `ChainComplex::fromSpacetime`'s boundary maps differ from the reference
(ascending id) maps by a sign per cell. The first build of this branch
refused such complexes and the refusal escaped as `std::terminate` inside
`buildStep`. `ChainComplex::orientationSigns()` now derives and verifies
those signs from the stored maps (`∂^stored = D_{k−1} ∂^ref D_k` exactly),
and the pencil's operators are reported in the stored basis as `D h D`.
Freshly built and Pachner-refined complexes have every sign +1.

## Admissibility

Under the pencil the configuration space is the closure of the
Kontsevich–Segal allowable domain: margin `min_T (π − Σ_i |arg λ_i(g_T)|) ≥ 0`.
Real Lorentzian data sit exactly on the boundary (margin 0) and are admitted
and certified as the boundary; a strictly non-allowable proposal is not a
member (stage 1's manifold gate and stage 2's line search both refuse it, as
they refuse a non-manifold). A uniform complex conformal factor on a
Lorentzian base lands exactly on the boundary, not outside it; the
specification's non-allowable instance is the curved Lorentzian torus with a
complex conformal factor (margin < 0 measured).

## Deltas from the diagonal-weight baselines

| quantity | diagonal weights | Whitney pencil |
|---|---|---|
| deterministic objective core, closed S⁴, k = 3 (`test_multi_cobordism`) | 502.9804372639758 | 502.9755207221824 |
| its near-kernel term | +0.0093451 | +0.0044286 |
| fixed-boundary eigenstate, k = 1, 2-layer triangle tube, no growth, 8 restarts × 100 iterations | converged, residual 4.1e-9, 2.7 s | not converged, residual 3.0e-3, 29 s |
| same with 2 stellar growths, 16 restarts × 200 iterations | — | converged, residual 9.9e-9, 377 s |
| stage-2 accepted steps, unclamped fixture (`test_stage2_unclamped`), 3 iterations | 2 | 1 |
| single-pair period fit residual (`test_geometric_operators` identifiability) | < 1e-20 | 142.7 |
| recursive readout AMLS refusal reason on the capstone fixture | "non-normal" | "modeCutoff must cover the window upper edge" surfaces first |

The last four rows were measured with the Whitney default forced on
`MultiCobordism` alone; with the single knob the existing tests run on the
diagonal path unchanged and the pencil rows stand as the recorded deltas.
The fixed-boundary and period-fit rows are the ones the operator experiments
(#912) must revisit: the pencil couples cells that share a top simplex, so the
same pinned boundary data need a larger interior to be realized.

## Pre-existing failures found on main

`test_joint_stationarity_objective_python.py::test_zero_weights_disable_both_joint_terms`
and `::test_joint_objective_is_two_stationarity_residuals` fail on `main`
before this branch: the objective carries a `connection_stationarity` term of
7.1187e-06 on the closed-S⁴ fixture with both joint weights zero. Not touched
here.

## Not reproduced from the specification

The T6 harmonic Gram determinant 0.211555 (recorded on #907): the value
depends on the kernel-basis normalization, which the specification does not
state; the basis-independent statements (rank two, signature (1,1)) hold.

## Bipartite operator representation (#911)

Protocol: the two input states are the boundary, ∂W = A ⊔ B; the whole
complex carries the output; the operator is READ on the frozen bulk as the
Poincaré–Steklov extension ψ_I = −(L_II − λ)^{-1} L_I∂ ψ_∂ at the common
eigenvalue, so linearity in the inputs is exact by construction and the
content of the protocol is that the relaxed witnesses are that extension.

Exact (always-on) on the two-layer triangle tube under the pencil, with
jittered interior lengths: the whole-complex harmonic chain is the extension
of its own boundary values; scaled inputs read as scaled outputs; rotating
the boundary circles' cells is matched by an attachment permutation.

The coupled boundary-state relaxation (`relaxBoundaryStatePairs`, the #900
fit) at degree 1 on this fixture family, measured:

| pinned pair | metric | layers | growth | restarts × iterations | converged | residual | eigenvalue | time |
|---|---|---|---|---|---|---|---|---|
| (cycle, cycle), exact λ = 0 solution exists | Whitney | 2 | ≤1 | 4 × 100 | no | 4.2e-3 | 2.39 | 26 s |
| (v_a, v_a) in the λ = 6 sector | diagonal | 2 | ≤1 | 4 × 150 | no | 5.3e-3 | 2.25 | 5 s |
| (v_a, v_a), (v_b, v_b) | diagonal | 2 | ≤1 | 4 × 150 | no | 1.2e-3 | 3.43 | 25 s |
| (v_a, v_a) | Whitney | 2 | ≤1 | 4 × 150 | no | 3.6e-3 | 5.33 | 45 s |
| (v_a, v_a), (v_b, v_b) | Whitney | 2 | ≤1 | 4 × 150 | no | 9.8e-3 | 2.90 | 115 s |
| (v_a, v_a), (v_b, v_b) | Whitney | 3 | ≤1 | 8 × 200 | no | 1.4e-3 | 2.47 | 500 s |
| (v_a, v_a), (v_b, v_b) | Whitney | 2 | ≤2 | 8 × 200 | no | 1.9e-2 | 3.79 | 376 s |

Neither metric reaches the exact harmonic solution or the nonzero sector at
these budgets: the relaxation settles at an intermediate eigenvalue. This
is the realizability question #901 and #903 characterize (the qutrit sector
fails there for two structural reasons); the convergence gates of the
protocol run under `TESSERA_SLOW_TESTS=1`, and #912 re-runs the operator
experiments under the pencil once those land.

## Register readouts read geometric images (#931)

The epic's validation run of the merged operator-transfer experiment
(`examples/cobordism/geometric_operators.py`) under the pencil failed its
period-fit checks: identity `r_U = 168`, cycle `r_U = 210`, held-out errors
0.34, where the diagonal metric reads `1e-27` and `1e-14`. The cause was not
the landscape: `EigenstateSynthesis` pairs cycles with the kernel vectors of
`laplacian(k)` and adds leak amplitudes on cells, which is the edge integral
only when those vectors are geometric images (specification §4.3: "z is the
readout"; §6: the left frame's entries are edge integrals). The pencil path
had returned the chain-space operator, whose kernel vectors are chains
`h = M z`. `laplacian(k)` under the pencil is now the operator on images,
`L_z = (M^U)^{-1} h M^U`, with its analytic derivatives, and the chain-space
operator stays available as `CovariantChainHodge::covariantOperator`.

With that, the experiment passes every check under both metrics:

| read | diagonal weights | Whitney pencil |
|---|---|---|
| single-pair period residual `r_U` | 1.2e-27 | 1.1e-20 |
| identity / cycle held-out error | 8e-15 / 2e-14 | < 1e-8 (pinned in tests) |
| generic charge-preserving target `r_U` after relaxation | 0.0193518 | 7.56818e-06 |
| hard gap / transport error (metric-independent reads) | 0.135345 / 0.520279 | 0.135345 / 0.520279 |
| wall time | 6 s | 71 s |

The single-pair cutoff `_TINY_PERIOD_RESIDUAL` moved from 1e-20 to 1e-18: both
values are round-off of an exactly realizable fit, the pencil's operator
entries being an order of magnitude larger on that fixture.
