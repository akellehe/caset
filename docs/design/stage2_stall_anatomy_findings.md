# Stage-2 stall anatomy on campaign geometry dumps (#600)

**Status: exploratory — findings, not canon.** This document must not be used as a
base for further implementations. Every engine or worker change it motivates gets its
own ticket and lands only at a campaign worker-generation boundary.

## 1. Question and definitions

`MultiCobordism::runStage2` relaxes a fixed-topology host by descending the objective

- `F = β·‖g‖² + γ·r_U`, where
- `g = ∂S/∂ℓ²` is the exact complex Regge-action gradient over edges
  (`ReggeSolver::actionGradientExact`; `S` is the complex Lorentzian Regge action and
  `ℓ²` the real signed squared edge lengths),
- `H = ∂²S/∂ℓ²∂ℓ²` is the exact complex Hessian (`actionHessianExact`),
- `r_U` is the register residual, and
- the descent direction is `d₀ = 2β·Re(H̄·g)` — the exact gradient of the `β‖g‖²`
  term on the real-ℓ² manifold — with a backtracking line search (initial scale
  `alpha0 = 0.05`, growth ×1.3 capped at 1.0, at most 24 halvings) and the
  stationarity declaration "no tested step improves `F` by more than
  `relTol·max(|F|,1)`" with `relTol = 1e-9`.

Campaign attempts (issue #562, joint arm) end `converged` when this test fires with the
input `r_U` at its floor (≈1e-30) but `F ≈ O(0.1–10)`: the `‖∇S‖²` term stalls at
order one, far from a stationary geometry (`δS = 0` would be `‖g‖² ≈ 0`). The owner's
proposed policy — on a `b₃ = 3` register discovery, relax to machine precision — needs
to know **why** stage-2 stalls, and which finisher (explicitly **not** vanilla Newton;
the landscape is strongly nonlinear) could go further.

## 2. Data and method

All measurements ran on rebuilt campaign geometry dumps
(`Spacetime.fromCells` + recorded vertex times + per-edge complex `ℓ²`, the
`analyze_attempt.py::rebuild_from_dump` contract), never on the live campaign worktree,
and touched no engine code. Probes, per specimen:

- **A0** — an exact mirror of the `runStage2` loop (`alpha0 = 0.05`, `relTol = 1e-9`)
  from the rebuilt state. Zero accepted steps validates that the rebuilt state is the
  recorded stall.
- **A1** — the same loop with `relTol = 0`: does an unlimited improvement threshold
  alone recover descent?
- **B** — a Levenberg-damped Gauss–Newton loop on `‖g(ℓ²)‖²` over the real-ℓ²
  manifold: solve `(JᵀJ + λI)·d = Jᵀr` with `J = [Re H; Im H]`, `r = [Re g; Im g]`
  (so `∇F = 2Jᵀr = d₀` exactly), a trust-region-style λ update, and the
  actual-versus-predicted reduction ratio ρ per step as the nonlinearity gauge. As
  λ grows this step degrades to a short gradient step; it is never an undamped
  (vanilla) Newton step and needs no third derivatives of `S`.
- **1-D scans** — `F(x − t·d₀)` for `t = ±10⁻¹⁶ … ±10⁻²`, compared against the
  analytic slope `dF/dt|₀ = −‖d₀‖²`.

Specimens: converged+stationary attempts spanning `max_b3 ∈ {0,1,2}` (no `b₃ = 3`
exists yet). The four `b₃ ∈ {1,2}` specimens rebuild **bit-faithfully** (fresh `‖g‖²`
matches the worker-recorded F to ≤3×10⁻¹⁶ relative; A0 accepts zero steps). The two
`b₃ = 0` specimens do not — see Finding 5 — and are excluded from the stall anatomy.

## 3. Finding 1 — recorded stationarity is not criticality

At the four faithful recorded stalls the true objective gradient is enormous:

| seed | b₃ | edges | F at stall | ‖∇F‖ at stall | σ_max(J) |
|---|---|---|---|---|---|
| 2001004 | 2 | 51 | 5.620 | 1.4×10³ | 9.7×10³ |
| 15001020 | 2 | 55 | 11.924 | 2.8×10⁶ | 2.2×10⁷ |
| 5001034 | 1 | 35 | 6.441 | 6.9×10⁴ | 3.8×10⁵ |
| 2001002 | 1 | 36 | 2.488 | 4.4×10² | 1.1×10⁵ |

"Stationary" in the worker sense therefore means "the specific line-search protocol
found no acceptable step", not `∇F ≈ 0`. Every specimen also shows exactly one zero
singular value of `J` — the global scale mode (`S` is degree-1 homogeneous in `ℓ²`,
so `H·ℓ² = 0` by the Euler identity), the expected conformal null direction.

## 4. Finding 2 — the backtracking floor is quantifiably too high

The smallest step the line search can test is `alpha0/2²⁴ ≈ 3×10⁻⁹` (at most
`1.0/2²³ ≈ 1.2×10⁻⁷` after growth). The curvature scale along the gradient is of
order `σ_max²`; a descent step must sit below ~`1/σ_max²`:

- 15001020: required ≈ 2×10⁻¹⁵ — 6 decades below the tested floor;
- 5001034: required ≈ 7×10⁻¹² — 3 decades below;
- 2001002: required ≈ 9×10⁻¹¹ — 2 decades below;
- 2001004: the 1-D scan finds an actual descent window `t ∈ [3×10⁻¹³, 3×10⁻¹⁰]`,
  one decade below the tested floor.

With `relTol = 0` (probe A1) all four still accept **zero** steps — the tested scales,
not the improvement threshold, are the binding constraint.

## 5. Finding 3 — the stalls are pinned on branch walls where `F` jumps discontinuously

The 1-D scans falsify the smooth picture outright, at scales where no smooth model
survives:

- 2001002: `F` **increases in both directions** `±d₀` at every tested scale from
  `t = 10⁻¹⁶` to `10⁻⁸`. At `t = 10⁻¹⁶` (displacement `‖δx‖ ≈ 4×10⁻¹⁴`) the smooth
  prediction is `dF = −2×10⁻¹¹` and quadratic corrections are negligible by seven
  orders; the measured value is `+1.1×10⁻⁴` — a **jump discontinuity**, twelve
  orders above evaluation rounding. 15001020 behaves identically (a `+6.6×10⁻³`
  step for infinitesimal perturbations).
- The evaluation is bit-deterministic (repeated evaluations at the same lengths agree
  to the last digit), so the jump is structure, not noise.
- The jump is **anisotropic**: at fixed infinitesimal displacement in eight random
  directions from 2001002's stall, several probes stay smooth (`dF ≈ +10⁻¹¹`) while
  the rest jump by `+10⁻⁵…+10⁻⁴` — the signature of a discontinuity wall (or corner
  of several walls) passing through the point, with all sampled sides uphill. This
  stall is a genuine **nonsmooth local minimum**: the damped probe (Finding 4) also
  found no descent at any damping.
- 2001004 differs: a real descent window exists (`t ∈ [3×10⁻¹³, 3×10⁻¹⁰]`) whose
  measured slope (`−1.5×10⁹`) is ~800× steeper than the analytic one (`−1.9×10⁶`) —
  descent along a crease of the low branch, walled at both larger and smaller steps.

Mechanism: `g = ∂S/∂ℓ²` is branch-classified (dihedral-angle branches, causal wedge
classification, near-degenerate wedge saturation). Where an infinitesimal length
perturbation flips a classification, `g` — and hence `F = ‖g‖²` — jumps. The
analytic `g`, `H` are one-branch values: perfectly good in smooth regions (they drove
thousands of productive descent steps per trajectory), meaningless as stationarity
measures on a wall. Which wall type is active at each stall is **not yet measured**
(all four faithful specimens are spacelike-only hosts, so wedge-degeneracy saturation
is the leading candidate there); this is Recommendation 4.

## 6. Finding 4 — what a damped Gauss–Newton finisher actually buys

- **2001004 (b₃ = 2):** from the dead stall (zero line-search steps possible), the
  damped Gauss–Newton loop immediately recovers descent — `F: 5.620 → 3.149` over
  3,570 iterations — with near-perfect model agreement on accepted steps early on
  (median ρ ≈ 0.99), then a hard plateau: the final ~250 iterations buy ~0.0015 while
  λ climbs and the one-branch `‖∇F‖` jumps erratically (10³–10⁵), the signature of
  crawling along walls. The damped-method plateau in this basin is **positive,
  ≈ 3.15 — not 0**; whether some other continuous path reaches lower was not
  observed.
- **2001002 (b₃ = 1):** immovable for the damped loop as well (three accepted steps
  of cumulative `ΔF ≈ 4×10⁻⁶`, then rejection at every damping up to λ = 10¹⁸) —
  consistent with Finding 3's diagnosis of a wall-pinned nonsmooth local minimum.
- On the two (unfaithfully rebuilt, hence illustrative-only) `b₃ = 0` geometries the
  same loop ran to near-criticality at `F ≈ 0.02–0.08` with `‖∇F‖` down to `4×10⁻⁴`:
  bona-fide positive-floor critical points of `‖∇S‖²` exist in these landscapes.

Conclusion for the machine-precision policy: **no probe observed a route to
`F → 0`** in any examined basin. `δS = 0` is not attainable there along continuous
relaxation; the attainable convergence object is a crease-stationary point with a
positive, honestly-reported `F` floor. "Machine precision" must therefore be defined
as a stationarity test compatible with a piecewise-smooth objective (no descent along
a probe set of directions at scales down to numerical resolution), **not** as
`F < 10⁻²⁶` and **not** as `‖∇F‖ < 10⁻¹⁵` (the one-branch gradient does not vanish
at a crease minimum).

## 7. Finding 5 — geometry dumps are unfaithful exactly on timelike-edge hosts

Rebuilding **all 199** recorded attempts and comparing the fresh `‖g‖²` against the
worker-recorded F (computed live via `node.objective()`):

| host class | count | F match (<10⁻⁹ rel) |
|---|---|---|
| all edges spacelike (`re_min > 0`) | 188 | **188 — every one** |
| at least one timelike edge (`re_min < 0`) | 11 | **0 — every one mismatches** (2.3% … 1190% rel) |

The dump records cells, vertex times, and complex `ℓ²`, and the rebuild reproduces all
of them bit-exactly — yet the action differs. The rebuild is itself **bit-deterministic**
(three rebuilds of the same dump agree to the last digit), so the reconstruction
deterministically lands on a *different* state than the live one: some Regge-relevant
state on timelike-carrying hosts is **not determined by (cells, times, lengths) as
consumed by the rebuild path** — the leading candidate is the same-sign wedge boost
orientation, which the mixed-hinge work found is not length-determined. Implications:

- The "faithful record" contract (`dump → fromCells`) silently breaks on exactly the
  causally-nontrivial specimens the causal-content program needs.
- Any verdict recomputed from a reloaded dump on such a host is currently wrong; and
  the identical-recorded-F puzzle among no-hole attempts is fully explained (the
  live records were right; the dumps do not round-trip).

## 8. Recommendations (each its own ticket; all land at a generation boundary)

1. **Dump fidelity on timelike hosts** — highest priority; it gates every reload-based
   readout (including the interim mitigation for the #587 in-process drift). Either
   extend the dump schema with the missing orientation state or make the rebuild path
   rederive it identically; add a round-trip invariant (fresh `‖g‖²` equals live F on
   mixed hosts) to the suite.
2. **Backtracking floor knob** — make the halving count / minimum tested scale
   configurable (24 halvings → ≈48 reaches the 10⁻¹⁵ scale). Exact, physics-free,
   small change; it would have recovered 2001004's descent window unaided.
3. **Register-bearing finisher** — damped Gauss–Newton / trust region as the
   workhorse (it demonstrably restarts descent from dead stalls), terminated by a
   crease-aware stationarity test (multi-direction, multi-scale descent probes) and an
   honest stall verdict recording the achieved F, the probe scale, and the one-branch
   `‖∇F‖`. Explicitly not vanilla Newton. Whether the `γ·∇r_U` term joins the descent
   direction (via `residualForPeriodsGradient`) is a separate owner decision — this
   probe omitted it (input `r_U ≈ 1e-30` makes it endpoint-negligible, but it is
   path-relevant on register-bearing hosts).
4. **Crease mechanism instrumentation** — classify the active wall at each stall
   (wedge-degeneracy saturation versus causal-class flip) from per-hinge angle data;
   this decides whether crease-stationary endpoints are physics (geometric folds) or
   avoidable parameterization artifacts.
5. **Policy wording** — replace "relax until machine precision" with "finish with the
   damped method until crease-stationarity at numerical resolution; report the F floor
   as an observable". A stalled `b₃ = 3` specimen is itself a finding.

## 9. Threats to validity

- Four faithful specimens (plus two excluded unfaithful ones); `b₃ ≤ 2` only — no
  `b₃ = 3` specimen exists yet, so transfer to the target class is an extrapolation.
- Probe B descends `β‖g‖²` alone (γ-term omitted; see R3).
- The `b₃ = 0` floor values (≈0.02–0.08) were measured on unfaithful rebuilds and are
  illustrative of the landscape class, not of those live states.
- The crease mechanism (Finding 3) is established behaviorally (two-sided increase,
  slope inconsistency at `10⁻¹⁶`-scale steps, far above rounding); its geometric
  origin is unmeasured (Recommendation 4).
- The rebuild engine is the current main; the campaign generation matches it in all
  action-relevant physics.

## 10. Artifacts

The probe scripts (`stall_anatomy.py`, `scan_1d.py`, `floor_probe.py`) and raw results
(`stall_anatomy_results.json`, `fidelity_sweep.json`, run logs) are attached to issue
#600 (issue-attachments), not committed to the tree.
