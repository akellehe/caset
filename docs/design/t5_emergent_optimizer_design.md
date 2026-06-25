# T5 — Emergent optimizer loop on closed S⁴ (opening design note)

Part of the Emergent Color Topology epic (#457). The opening design task for #462:
settle the architecture and the remaining open questions **before** writing the loop.
This note is sign-off-only; the optimizer code lands in follow-up PRs.

## 1. What is locked (from #457 + the owner decisions on #462)

- **Objective — one functional, both stages, EXTREMIZE the action (δS = 0):**
  `F = ‖∇S_Regge‖² + Γ·r_U`. The geometry term is the **squared gradient norm** of
  the full-complex dual (Sorkin) Regge action — `‖∇S_Regge‖² = Σ_e |∂S/∂ℓ²_e|²`, keep
  `Im` (complex modulus). It drives the geometry to a **stationary point** (the Regge
  equations `∂S/∂ℓ² = 0`); it is **not** the action magnitude `|S_Regge|` (which would
  collapse the curvature / run the conformal mode away). `Γ·r_U` is the only matter term.
- **Incremental form (T4, #461, merged):** `ΔF = Δ‖∇S_Regge‖² + Γ·Δr_U`, evaluated
  hinge-locally. `Δ‖∇S_Regge‖²` is exact and local (`ReggeSolver::affectedEdgesOfCells`
  + `gradientNorm2OverEdges`); `Δr_U` is an exact before/after `residualForPeriods`
  recompute (r_U is a global spectral quantity — no hinge-local delta).
- **Flavor is EMERGENT** — no second register, no `ℂ³⊗ℂ²` isospin index, no
  flavor-representation gate. The single existing `r_U` (`EigenstateSynthesis.residualForPeriods`
  at the spatial register degree `k = 2`) is the whole gauge-state residual.
- **Host:** closed **S⁴** (no Dirichlet *outer* boundary); start/end states are fixed
  **interior** sub-complexes, not boundary data.
- **Moves:** existing **Pachner** suite (reused, never reimplemented) — topology-
  preserving — plus **gated surgical cone-out/in** (`SurgicalCone`, #469) — the only
  topology-changer. **Stage 1 uses ONLY** Pachner moves + surgical cone-in/out (no
  star-add/remove or other surgery). Every move is gated by the manifold check.
- **Optimizer:** greedy best-`ΔF` move + **random restarts** (optimizer, not sampler).

## 2. The host: a closed S⁴ with pinned interior states

Build the closed 4-manifold from the existing dimension-generic stacking. The
reference oracle (`S3WindowSurface`, #453) shows the path: a triangulated **S³** slice
stacked over apex slices by `Spacetime::symmetricStackCells` (#429,
`Spacetime.h:369`) gives a genuine closed **4-complex** (pentatope top cells), gated by
the `n ≥ 4` recursive manifold check. For the emergent loop the slice need **not** be
the hand-placed window surface — the minimal `SimplexBoundarySphere(3)` (a tetrahedral
S³), refined by a few `PreGeometric` Pachner moves so disjoint top cells exist, stacked
to a small closed S⁴, is the starting host. `Spacetime::fromCells(4, …)` materializes it.

**Pinned interior states — the crux.** A closed S⁴ starts at `b₂ = 0`: there is no
color register yet, so `r_U` has nothing to score. The states must therefore be
**seeded as fixed interior register holes**, not as abstract "regions":

- An input (resp. output) state is a small **fixed sub-complex** carrying a removed
  top-cell hole at the register degree (`k = 2` → a removed tetrahedron, the `b₂`
  hole) with a pinned color **target period** (the ω color charge `[1, ω, ω²]`, frame
  up to §4). Its edge lengths are **Dirichlet-fixed** (the "fixed interior region").
- The optimizer relaxes the **bulk** between/around the pinned states and, via gated
  surgery, may grow **additional** `b₂` holes when a pinned color charge cannot be
  carried without them. `r_U = residualForPeriods(allPinnedHoles, allTargets)` on the
  current emergent register is then well-defined from step 0 (high while the register
  can't carry the pinned charge, → 0 once it can).

This reuses the established pinning machinery: `CobordismRelaxer::relaxInterior`
(`CobordismRelaxer.h:54`) already partitions **Dirichlet-fixed boundary/pinned edges**
from the **free interior edges** and relaxes `β‖∇ᵢS‖² + r_state` to a stationary point;
the pinned holes flow through its `stateHoles`/`holeTargets`. The novelty of T5 is that
the *topology* between the pinned states is no longer hand-built — it **emerges** under
the move set. (The minimal-event question — one seeded input + one seeded output, grow
the connecting register — is exactly what T6/#463 then validates.)

## 3. The move set + the gate

| Move | Class | Topology | Notes |
|---|---|---|---|
| 1↔(d+1) stellar refine | `AddMove`/`RemoveMove` (`PachnerMode::PreGeometric`) | preserving | combinatorial freedom |
| bistellar flip | `FlipMove`/`IFlipMove` | preserving | combinatorial freedom |
| cone-out / cone-in | `SurgicalCone::coneOut`/`coneIn` (#469) | **changing** (`Δb₂ = ±1`) | the hole creator/capper |

Every accepted move is gated by **`EigenstateSynthesis::dualComplexValid()`**
(`EigenstateSynthesis.h:281`, wrapping `ChainComplex::dualComplexIsValid`,
`ChainComplex.h:71`) — the `n ≥ 4` recursive manifold-with-boundary check (facet coface
counts ∈ {1,2}, ridge links single paths/cycles, vertex links validated as 3-spheres/
disks). Surgery is allowed **because** it is gated; a rejected move rolls back
bit-identically (`SurgicalCone::rollback`, the Pachner moves' `rollback`). Cone-out is
the **exact inverse** of cone-in (T4 verified: it removes only the k apex edges of the
created cell), so the move set is closed under inversion — required for an honest
greedy `ΔF` landscape.

**A b₂ hole takes a disjoint PAIR of cone-outs** (T4 finding): the first cone-out
opens the manifold (`b₃ → 0` on a slice / the local sphere), the second *disjoint* one
raises `b₂` by 1. The optimizer's surgical proposals must therefore be **disjoint-pair
aware**, not single-cell.

## 4. Open item (c): the Z₃-frame — score the nearest ω-rep over embeddings

On the **hand-placed** symmetric windows the color `Z₃` is the window-cycling symmetry
`σ = τ^(K/3)` (`proton_observables.py`: `P_out` eigenvalues `{1, ω, ω²}`, the ω-rep
input as the eigenvector of `P_in`). An **emergent** complex carries **no a priori
`Z₃`**: the holes the optimizer grows are not a symmetry orbit, so there is no canonical
frame in which the target reads `[1, ω, ω²]`.

**Decision: do not impose a frame — score the nearest ω-rep over embeddings.** `r_U`
for a color-charged state is

```
r_U = min over frame assignments φ of  residualForPeriods(holes, ω-target ∘ φ)
```

i.e. the residual of carrying the standard color rep, **minimized over the ways to
attach the ω-rep to the emergent holes** (the `Z₃` cosets / hole orderings). This is
frame-free and emergent-faithful: it asks "can this register carry a color charge **at
all**," never "does it carry it in a frame we picked." For a 3-hole window the
minimization is over the finite set of cyclic/labelling assignments (cheap); the
relabeling-invariant `σ_R` (#412) is read in the winning frame after convergence. The
alternative — *emerging* an exact `Z₃` symmetry of the metric — is rejected here as
over-constraining (symmetry is not naturally a relaxation fixed point), consistent with
"emergent-first, never insert the answer by fiat."

## 5. Open item (d): r_U hardening — exact b_k cross-check

`r_U = residualForPeriods` reads the carried register through
`HodgeLaplacian::harmonicMatrix(k, 1e-9, metric)` — a spectral kernel thresholded at
`1e-9`, so the carried **dimension** is only as exact as the spectral gap. Near a
topology change (a hole opening/closing) the gap can be small and the count wrong.

**Decision: gate the register dimension with the exact topological Betti number.**
`ChainComplex::bettiNumbers()` (`ChainComplex.h:77`, free rank over ℚ) and the
integer Smith-normal-form path (`IntegerLinalg::smithNormalForm`/`integerRank`,
`IntegerLinalg.h:28,31`) compute `b₂` exactly and metric-free. The loop:

- after every accepted **surgical** move, recompute `b₂` exactly and assert the
  harmonic-subspace dimension (`harmonicMatrix` rank at `1e-9`) **equals** it; a
  mismatch flags an ill-conditioned readout (tighten the threshold / reject the move).
- the residual *value* stays the spectral `residualForPeriods` (it must be
  differentiable for Stage 2); only the **dimension** is exact-checked. This keeps the
  single `r_U` while removing the silent miscount risk #457 flagged.

## 6. Scale regulation (no boundary pin)

There is **no explicit `λ|Im S|` term** in the codebase, and none is needed: the
objective's geometry term is `‖∇S_Regge‖²` over the **complex** dual action
(`ReggeSolver::dualReggeAction`/`actionGradientExact` return `std::complex`; the
imaginary part is the boost/light-cone content of timelike-normal hinges). Minimizing
`Σ_e|∂S/∂ℓ²_e|²` drives **Re S and Im S to stationarity together** — this *is* the
regulation, and it matches the existing `CobordismRelaxer` cost `β‖∇ᵢS‖² + r_state`
(`CobordismRelaxer.cpp:77`). The conformal/scale flat direction (rescaling a Regge
solution stays a solution) is pinned by `Γ·r_U` (the matter backreaction), never a
boundary. **Stage 1** runs at fixed edge length (scale DOF frozen); **Stage 2** re-opens
it and must be watched for runaway — diagnosed by the restoring force, not bolted to a
boundary ("don't sidestep the conformal runaway").

## 7. Architecture of the loop

A new cobordism class — `EmergentOptimizer` (working name) — composing the merged T4
primitives, with **no new objective and no reimplemented moves**:

```
EmergentOptimizer(host: closed S⁴, pinnedStates: [(holes, targets)], Γ, β):
  Stage 1 (combinatorial, fixed edge length):
    repeat until no move lowers F (greedy), with random restarts:
      for each candidate move m in {Pachner refine/flip, surgical cone-out/in pair}:
        if not gate(apply m): rollback; continue          # dualComplexValid
        ΔF = Δ‖∇S_Regge‖²(touched edges)  + Γ·Δr_U(recompute)   # T4
        keep the best ΔF < 0; else rollback
  Stage 2 (continuous):
    relaxInterior on the free edges → stationary point of β‖∇S‖² + Γ·r_U
  persist best-F complex (§8); read σ_R / b₂ / r_U for the verdict (T6 reads it)
```

- **Greedy + restart**: best-`ΔF` per sweep; restart from independently perturbed
  seeds to escape local minima (optimizer, not sampler — temperature is not the
  acceptance rule; the existing Metropolis machinery is reused only as a
  restart/perturbation source if useful).
- **r_U at k = 2** in the candidate inner loop is a `residualForPeriods` recompute
  (a global eigendecomposition) — affordable on the **small** test hosts; the
  `@pytest.mark.slow` full-resolution event stays off CI.
- **Performance prerequisite (carried from T4):** the arbitrary-k `r_U` *gradient*
  used in Stage 2 currently rebuilds the chain complex per edge (~424 ms/call at k = 2
  scale). Before Stage 2 runs on anything but toy hosts it needs the precompute-once
  optimization (assemble `L_k`/weights/boundary once, per-edge `dM` only). Tracked as a
  follow-up; Stage 1 (recompute-based) does not depend on it.

## 8. Persistence (GraphML)

`Spacetime::save("*.graphml")` → `writeGraphML` (`Renderer.cpp:633`) currently exports
the 0- and 1-skeleton only: vertices (`id`, `time`, `degree`) + edges (`squared_length`,
`timelike`). Extend it with **per-hinge attributes** — the dual-Regge curvature the
viz colors — by emitting, alongside the edges, the genuine `(d−2)`-hinges with
`deficit` (`Simplex::lorentzianDeficitAngle`, Re and Im) and `dual_volume`
(`Simplex::dualVolume`). This is additive (a new `<hinge>`/attribute block); the
downstream force-directed primal+dual curvature visualization is the consumer (out of
scope here).

## 9. Build plan (follow-up PRs after this note signs off)

1. `EmergentOptimizer` scaffold + the closed-S⁴ host builder + pinned-state seeding;
   Stage-1 greedy loop over Pachner + gated surgical cone-pairs, ΔF from T4. (`@slow`
   end-to-end test on a tiny host.)
2. The frame-free color scorer (§4) + the exact-b₂ hardening guard (§5).
3. Stage-2 `relaxInterior` integration + the k = 2 `r_U`-gradient precompute optimization.
4. Random restarts + GraphML per-hinge export (§8) + the converged-run verdict readout.
5. Findings report (`docs/design/`, slug + commit hash).

Each keeps `tests/cobordism/test_epic410_invariants.py` green and the heavy event `@slow`.
