# Why the relaxed proton geometry stays all-spacelike

*Investigation for #541; observed in #535/#538 (the dual temporal-curvature panel).*

## Observation

The dual heat map's **temporal** curvature channel — `Im(lorentzianDeficitAngle)·|★|`, the
boost / light-cone content carried by spacelike hinges (whose normal plane is timelike) — is
**identically zero** on the converged proton. There is no temporal curvature because there
are no timelike hinges: the relaxed geometry is effectively Euclidean.

## What pins it spacelike — and what doesn't

Two candidates were checked. Only the first is the real reason.

1. **The dynamics keep it Euclidean (the actual cause).** The optimiser is a *local descent*
   from an **all-spacelike seed** (`buildMinimalSeed` / `formation_node` set every edge
   `ℓ² = +1`), and the real-Euclidean configuration is a stable basin of
   `F = ‖∇S_Regge‖² + Γ·r_U`. The relaxed edge lengths stay clustered around `ℓ² ≈ 1` and never
   approach the light cone `ℓ² = 0`, so no edge ever crosses into the timelike half-line
   (`Re ℓ² < 0`). Because the seed is real (`Im ℓ² = 0`) the action, its gradient, and its
   Hessian stay real, and the descent preserves real, positive `ℓ²`: there is no symmetry-
   breaking toward Lorentzian signature.

2. **The `runStage2` clamp is NOT the binding constraint.** `MultiCobordism::runStage2`
   clamps each trial `Re ℓ²` to `[0.05, 20.0]` (`boundedRealPart = min(max(trial.real(),
   0.05), 20.0)`), which *would* forbid `Re ℓ² ≤ 0`. It is natural to suspect this is why the
   geometry stays spacelike — but the data shows the clamp **never fires**: the converged
   `min Re ℓ² = 0.262`, far above the `0.05` floor. The clamp is a degeneracy safeguard
   (it keeps simplices from collapsing as `ℓ² → 0`), not the reason there are no timelike
   edges. Even with the clamp removed the descent would not produce timelike edges, because
   it never points that way.

So: **the geometry is Euclidean by dynamics, not by clamping.** The Lorentzian-Regge
machinery is exercised only in its real (angle-defect) branch; the imaginary (boost) branch
is dormant because the geometry it would act on never forms.

## Evidence (seed 3, `formation_node`, `origin/main` @ 30047e0)

```
SEED (before relaxation):  10 edges   Re ℓ² = 1 (all)            Im ℓ² = 0
CONVERGED (after stage-2): 213 edges  Re ℓ² ∈ [0.262, 1.464], mean 1.001
                                      # at 0.05 clamp floor = 0   # timelike (Re<0) = 0
                                      Im ℓ² = 0 (all)
```

The converged `min Re ℓ² = 0.262` is the decisive number: it sits well *interior* to the
`[0.05, 20]` clamp (the clamp never fires) and far from the light cone `ℓ² = 0`, and the
descent there is stationary (`∇F ≈ 0`), so nothing is pushing any edge toward the timelike
half-line. The Euclidean configuration is simply where the relaxation settles.

## Recommendation (out of scope for this note)

If a genuine Lorentzian regime is wanted (so temporal curvature can be non-trivial):

- **Seed some causal structure.** From an all-spacelike real seed the descent stays in the
  Euclidean basin; introduce timelike edges (`ℓ² < 0`) in the seed, or matter / boundary
  conditions that impose a time direction, so the relaxation has Lorentzian structure to
  act on.
- **Replace the hard `Re ℓ² ≥ 0.05` floor with a causal-aware guard** — forbid only the
  degenerate light-cone band (`|ℓ²| < ε`, both signs) rather than the entire timelike
  half-line, so timelike edges are admissible while degeneracy is still excluded.
- **Verify the complex deficit / Lorentzian volume path across signature changes**
  (`Simplex.lorentzianDeficitAngle` already returns the complex Sorkin / Asante–Dittrich
  deficit), and watch the conformal runaway — the action is unbounded below, so the existing
  regulators must still hold in the Lorentzian regime.

## Related

- #535 / #538 — the temporal-curvature panel that surfaced `Im ≡ 0`.
- `src/cobordism/MultiCobordism.cpp` — `runStage2` (the clamp; the only geometry mover) and
  `buildMinimalSeed` / `Proton.formation_node` (the all-spacelike seed).
