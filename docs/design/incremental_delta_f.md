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
   shifts. `r_U` is a **global** spectral quantity — `residualForPeriods` =
   `‖L_k ψ − λ ψ‖²` with `ψ` built from the harmonic subspace `harmonicMatrix(k)`
   (the eigendecomposition is the cost), so it has no exact hinge-local delta like
   the action. The **exact** `Δr_U` is therefore a before/after `residualForPeriods`
   recompute — confirmed end-to-end: `ΔF = Δ‖∇S_Regge‖²(local) + Γ·Δr_U(recompute)`
   matches a full `F` recompute to 2.4e-15 on the holed-S³ **k=2** register (ω carried
   ~3e-3, singlet ~47).

   *Residual choice (investigated):* `residualForPeriods` (eigenvector residual) is
   the `r_U` — it is the only residual that supports general `k` (`periodGapForPeriods`
   is k=1-only, throwing on tetrahedral holes), and its analytic gradient
   `residualForPeriodsGradient` is exact to the 1e-9 spectral floor (matches a
   Richardson-extrapolated FD to 4e-9). The period-gap family is the separate #377
   hard period-pin `r_ψ`, kept as-is.

   *Arbitrary-k analytic gradient (Stage-2 affordability) — DONE.* The existing
   `residualForPeriodsGradient` was k=1-only (hardcoded `laplacian(1)` / edge-loop
   covector). It now generalizes through three exact, hand-verified pieces:
   - (a) **`Simplex::volumeGradient`** — the per-degree Hodge-weight gradient `∂W_j/∂ℓ²`
     (Jacobi on the Gram determinant, reusing the #354 machinery). Verified by
     closed forms (equilateral triangle `1/(4√3)`, regular tet `1/(24√2)`) and the
     Euler identity `Σ_e ℓ²_e ∂V/∂ℓ²_e = (j/2)V`.
   - (b) **`HodgeLaplacian::laplacianGradient`** — general-k `∂L_k/∂ℓ²` (`L_k =
     BₖᵀBₖ + Bₖ₊₁Bₖ₊₁ᵀ`, `dBₖ = diag(a_{k-1})Bₖ + Bₖdiag(b_k)`). Verified by the Euler
     identity `Σ_e ℓ²_e ∂L_k/∂ℓ²_e = −½L_k` (3.4e-15 at k=1, k=2).
   - (c) **`EigenstateSynthesis::periodGradientGeneral`** — the general-k `∂r_U/∂ℓ²`:
     `M = L_k`, the per-edge `∂L_k/∂ℓ²` through the same eigenvector-perturbation
     derivation, period covector from each removed-(k+1)-cell hole's facets. Matches
     the k=1 `residualForPeriodsGradient` to 1.7e-15, and satisfies the exact Euler
     identity `Σ_e ℓ²_e ∂r_U/∂ℓ²_e = −r_U` (4e-16 at k=1, 3.4e-14 at k=2).

   Finite difference is roundoff-limited and does **not** converge; the optimizer
   uses these analytic gradients, and the tests certify them against hand-derived
   identities, not FD. (Consolidation — routing `residualForPeriodsGradient` through
   the general path and retiring the k=1-only `periodGradientOverLoops` — is a clean
   follow-up once the general path has soaked.)
4. **ΔF = Δ‖∇S_Regge‖² + Γ·Δr_U**, and correctness tests: `ΔF` matches a full `F`
   recompute to ~machine precision across **every** move class — Pachner, surgical
   cone-out/in, and edge-length perturbation — including across a topology change.
   Keep `tests/cobordism/test_epic410_invariants.py` green.

## Status

Scaffolding. Implementation tracked on `feat/incremental-delta-f` (PR for #461).
