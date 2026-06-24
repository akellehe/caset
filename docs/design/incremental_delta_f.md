# Incremental local ΔF = Δr_U + β·ΔS_Regge (T4)

Part of the Emergent Color Topology epic (#457). Working design note for #461; this
file is finalized as the ticket's findings report (`incremental_delta_f_<hash>.md`).

## Problem

The emergent optimizer evaluates `ΔF` for every candidate move, every step. The
objective **extremizes** the action (drives `δS = 0`) — it does **not** minimize the
action magnitude (a bare action minimum collapses the curvature, `∫R → 0` /
conformal runaway). So the geometry term is the **squared gradient norm**:

```
F = ‖∇S_Regge‖²  +  Γ · r_U          (full-complex Lorentzian/Sorkin gradient — keep Im)
    ‖∇S_Regge‖² = Σ_e |∂S/∂ℓ²_e|²    (complex modulus, NOT ‖∇Re S‖²)
```

This corrects the earlier `F = r_U + β·S_Regge` / "not a ‖∇S‖² proxy" wording in the
epic and #461 (owner decision; note posted on #457/#461/#462/#463/#464). The
stationary point (`∂S/∂ℓ² = 0`, the Regge equations) is where `‖∇S_Regge‖² = 0`, with
`Γ·r_U` the only matter/state term. Matches `CobordismRelaxer` (`β·‖∇S‖² + r_state`),
the existing stationary-action relaxation.

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

## Dual vs primal (pinned)

`F` uses the **dual** Regge action `dualReggeAction()` (`ReggeSolver.cpp:141`):
`S = Σ_h |*h| · ε_h`, where the weight `|*h| = Simplex::dualVolume()`
(`Simplex.cpp:1390` → `dualVolRec`, `:1218`) is the **circumcentric dual cell
volume** — built from circumradii/circumcenters, *not* the primal hinge area. The
legacy primal `reggeAction()` (`:133`, `Σ hingeArea·deficit`) is **not** in `F`.

The deficit `ε_h = lorentzianDeficitAngle()` (`Simplex.cpp:861`) is
`2π − Σ_{σ∋h} σ.lorentzianDihedralAngle(h)` over the incident top cells, with the
dihedral the **Lorentzian boost** angle (un-clamped Cayley–Menger; `acos` → ordinary
angle for `|r|≤1`, complex boost for `|r|>1` — the source of `Im`). `ε_h` is the
hinge **curvature** and is the *same* in the primal `A_h·ε_h` and dual `|*h|·ε_h`
actions; the *dual* character of the action is the `|*h|` weight, not the deficit.
There is no separate "dual deficit angle".

The objective term is `‖∇S_Regge‖²`, whose per-edge component
`∂S/∂ℓ²_e = Σ_{h∋e}[∂|*h|·ε_h + |*h|·∂ε_h]` is built from the **same** per-hinge
dual-measure terms (`Simplex::dualVolumeGradient` / `lorentzianDeficitAngleGradient`),
so the gradient — and hence `‖∇S_Regge‖²` and its increment — inherits the dual
measure identically. The localized action `dualReggeActionOverHinges` is the
hinge-accounting building block (and a value-level diagnostic); the objective uses
the localized **gradient norm** over the affected edges.

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

1. **Affected-hinge accounting (building block, done).** `hingeFacesOfCells` →
   the `(d-2)` hinges that are faces of a move's touched cells; `dualReggeActionOverHinges`
   → the localized dual action over a FIXED hinge set (genuine-only), so
   `ΔS_Regge = after − before` is exact. The action value is a diagnostic; the
   objective (below) uses the gradient norm built from the same per-hinge terms.
2. **Δ‖∇S_Regge‖² (hinge-local, the geometry objective — DONE).**
   `ReggeSolver::affectedEdgesOfCells` (the edges a move on the touched cells can
   change) + `gradientNorm2OverEdges` (`Σ_e|∂S/∂ℓ²_e|²` with each `∂S/∂ℓ²_e` the full
   per-edge gradient over `e`'s star, complex modulus). Over a fixed affected-edge
   set across a move, `Δ‖∇S‖² = after − before` is exact. Verified to machine
   precision for an edge perturbation and a Pachner move; `gradientNorm2OverEdges`
   over all edges equals `Σ_e|actionGradientExact()_e|²`. Genuinely local: one edge
   perturbation touches ~23% of a 260-edge toroid (shrinking as the mesh grows).
3. **Δr_U.** The change in the state residual at `k = 2` induced by the move,
   including across a topology change where `b_k` (and thus the register dimension)
   shifts. Validate the incremental update against `residualForPeriods` recompute.
4. **ΔF = Δ‖∇S_Regge‖² + Γ·Δr_U**, and correctness tests: `ΔF` matches a full `F`
   recompute to ~machine precision across **every** move class — Pachner, surgical
   cone-out/in, and edge-length perturbation — including across a topology change.
   Keep `tests/cobordism/test_epic410_invariants.py` green.

## Status

Scaffolding. Implementation tracked on `feat/incremental-delta-f` (PR for #461).
