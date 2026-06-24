# Incremental local ΔF = Δr_U + β·ΔS_Regge (T4)

Part of the Emergent Color Topology epic (#457). Working design note for #461; this
file is finalized as the ticket's findings report (`incremental_delta_f_<hash>.md`).

## Problem

The emergent optimizer evaluates `ΔF` for every candidate move, every step, where

```
F = r_U + β · S_Regge          (the #10/#11 mediation F_β, full-complex Lorentzian/Sorkin action — keep Im)
```

A full recompute of `F` per candidate is the #418 cost spike (10³–10⁴×). The same `F`
and the **same single** `r_U` govern both optimization stages — only the move class
differs:

- **Stage 1** — combinatorial/topological moves at fixed edge length: Pachner
  (`AddMove`/`RemoveMove`/`FlipMove`) + gated surgical cone-out/in (`SurgicalCone`,
  `OrientedCone`, #468/#469).
- **Stage 2** — continuous edge-length perturbations.

So `ΔF` must cover **both** move classes and stay exact.

## Gauge state is the single existing `r_U` (no flavor register)

Per the epic owner's decision (recorded on #457/#461/#462/#463/#464): **flavor is
emergent** from the fully general optimization driven by the action we already have.
There is **no** second isospin/`ℂ³⊗ℂ²` register, no parallel holonomy, no per-sector
read-out. `r_U` is `EigenstateSynthesis.residualForPeriods` read at the spatial
register degree `k = 2` (the `b₂` color holes on `S⁴`), and that single residual is
the whole gauge state. `Δr_U` is built against it directly; nothing here waits on a
flavor-representation design.

## Existing infrastructure (what we build on)

- `ReggeSolver::dualReggeAction()` — full-complex action, already **hinge-local**:
  `S = Σ_h |*h| · ε_h` with `|*h| = Simplex::dualVolume()` and
  `ε_h = Simplex::lorentzianDeficitAngle()` (complex). Per-hinge analytic
  gradients/Hessians exist on `Simplex`.
- `ReggeSolver::actionGradientExact()` (#365/#371) — per-hinge product-rule
  accounting with the `relabel=false` / stable-id machinery; the template for
  hinge-local recomputation.
- Move classes track what they change: `AddMove` (`touchedVertexIds`,
  `createdSimplexVerts`, `createdEdges`), `RemoveMove` (`deletedEdges`,
  `createdSimplexVerts`), `FlipMove` (`oldSimplexVerts`/`newSimplexVerts`);
  `SurgicalCone`/`OrientedCone` wrap them.
- `EigenstateSynthesis::residualForPeriods(holes, target)` at degree `k`, plus
  `cyclePeriods`, `residualForPeriodsGradient`, `dualComplexValid` — currently
  **full recompute** (eigendecomp of `L_k`, `|λ|<1e-9` threshold).

## Plan (no shortcuts)

1. **ΔS_Regge (hinge-local).** Given a move's touched-cell set, enumerate the hinges
   whose `(|*h|, ε_h)` change (the closed star of the touched simplices), decrement
   their old contributions and increment the new ones — Re **and** Im — reusing the
   stable-id machinery so orphaned hinges are accounted for (cf. #365/#371). Covers
   both the combinatorial moves (cells added/removed) and the edge-length
   perturbation (the hinges incident to the perturbed edge).
2. **Δr_U.** The change in the state residual at `k = 2` induced by the move,
   including across a topology change where `b_k` (and thus the register dimension)
   shifts. Validate the incremental update against `residualForPeriods` recompute.
3. **Correctness tests.** `ΔF` matches a full `F` recompute to ~machine precision
   across **every** move class — Pachner, surgical cone-out/in, and edge-length
   perturbation — including across a topology change. Keep
   `tests/cobordism/test_epic410_invariants.py` green.

## Status

Scaffolding. Implementation tracked on `feat/incremental-delta-f` (PR for #461).
