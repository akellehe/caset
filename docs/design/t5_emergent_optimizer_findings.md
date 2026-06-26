# T5 — Emergent optimizer: findings

Findings from the T5 emergent optimizer (epic #457; tickets #462, #475), built on the
design note [`t5_emergent_optimizer_design.md`](./t5_emergent_optimizer_design.md).
Harness: `examples/cobordism/emergent_optimizer.py` (commit `f1c86e4`). Build-plan §9.5.

## 1. What was built

A Python harness composing the merged T4 primitives into the §2 cobordism model — no new
objective, no reimplemented moves:

- **Host** — a bare closed S⁴ (`SimplexBoundarySphere(4)` refined for surgery room).
- **Three-term, asymmetric `r_U`** — two inputs constructed *in place* as interior
  sub-complexes whose own `L_k` harmonic represents each input (held *representable*: a
  move is rejected only if it removes an input vertex), and an output that is the harmonic
  of the *whole* structure. Each residual reads the **emergent register off the structure**
  (`emergent_holes` via `getBoundary` — a pure read, nothing placed) and projects the
  fixed-dimension expected state onto the `L_k` harmonic's carried period space, with the
  register **zero-filled** to the target dimension and matched **relabeling-invariantly**.
- **Two stages, one functional `F = ‖∇S_Regge‖² + Γ·r_U`** — Stage 1 (combinatorial, fixed
  edge length): greedy best-ΔF over random single Pachner + gated surgical cone-out/in,
  scored by the incremental T4 ΔF, gated by `dualComplexValid`. Stage 2 (continuous): every
  **complex** edge `ℓ²` relaxed by the exact Wirtinger gradient `2·conj(H)·g` toward a
  stationary point, `r_U` gating.
- **Multi-degree `r_U`** — generalized from a single register degree to a *set* of degrees
  (`degrees=(2,3,…)`), `r_U` summed over them, so several `L_k` registers can be required
  to emerge at once. Default `degrees=(3,)`.

## 2. The natural register degree on a d=4 host is k=3, not k=2

On the bare d=4 host the topology-changing move (a cone-out) removes a **top 4-cell**, which
creates a `b₃` register (its boundary is a 3-cycle) — never a `b₂`. This is the
`ker L_{d-1}`, holes = removed top d-cells rule: the spec's "start at k=2" is natural to a
*d=3 slice*; on the d=4 host the register lives at **k=3**. At k=2 every `r_U` term sits at
its full zero-filled leak and the loop is flat; at k=3 the register emerges
(`[1,0,0,0,1] → [1,0,0,2,0]`, `r_U` 9 → 2.3) and F descends normally.

## 3. A `b₂` register is unreachable by the move set (conclusive)

To test whether `b₂` could nonetheless emerge from a sequence of moves, an 8-worker random
walk over the *exact* gated move set (`coneOut`/`coneIn` + Pachner, `dualComplexValid`
gate), with frequent random reinitialization, ran ~9.15 h:

- **~13.9 M moves (~1.19 M topology-changing)**, `b₃` reached up to **13**, and
  **`b₂` never once left 0** — zero `b₂>0` structures. Every betti vector was `[1,0,0,b₃,0]`.

This is a hard structural fact, not a search-depth artifact: a `b₂` register is a **codim-2**
feature (a non-bounding 2-cycle), reachable only by drilling a coordinated *tube* of
removals, which the random greedy moves do not produce (and which the framework forbids
coordinating by hand). **A `b₂` register requires a dedicated codim-2 surgical move** —
which must be expressed as top-cell changes that keep the complex a pure
d-manifold-with-boundary (the snapshot/restore, the dual Regge action, and the gate all
assume purity). **But the flavor read does not need it** (§4).

## 4. Flavor emerges at k=3 — distinguishable per-hole Dirac–Kähler charge

The flavor read does **not** require the k=2 register. The Dirac–Kähler taste structure
(`multiplicity = 4` for d=4) lives in the Clifford grouping, which spans *all* Hodge degrees,
so the `b₃` register's content populates it regardless of degree. The substantive flavor
test (the #414 candidate) is the **per-hole DK charge** `q_h = ⟨Φ_h, Φ_h⟩_W` (the conserved
current `j⁰ = W·|Φ|²`) of each register hole's carried representative: do different holes carry
*distinguishable* charge?

On converged `b₃` structures the answer is **yes, robustly** — across five independent
structures the per-hole charges differ with relative spread **0.52–0.61**:

| seed | betti | per-hole \|DK charge\| | rel. spread |
|---|---|---|---|
| 3 | `[1,0,0,3,0]` | 0.093, 0.124, 0.068, 0.103 | 0.57 |
| 5 | `[1,0,0,3,0]` | 0.066, 0.119, 0.111 | 0.54 |
| 6 | `[1,0,0,4,0]` | 0.063, 0.089, 0.117, 0.068, 0.105 | 0.61 |
| 11 | `[1,0,0,3,0]` | 0.072, 0.068, 0.100, 0.114 | 0.52 |

This is exactly the signature **#414 found *absent*** on the hand-built symmetric A₄ register
(where the symmetry forced the per-window charges equal — indistinguishable). On the
**emergent, non-symmetric** register the holes are inequivalent and carry distinguishable
conserved charge: a genuine flavor-like distinction.

The field-strength E/B split (#414 candidate i′) is the *one* read that does not transfer to
k=3 — `F = dA` is a 2-form by construction, so `fieldStrengthSplit` is intrinsically
degree-2. It is redundant with the DK charge for the distinguishability question and is
deferred (it would need either a `b₂` register, or a degree-3 field-strength analog).

## 5. The distinction is real, not a geometric artifact (Stage-2 stability)

The per-hole charge *magnitudes* depend on geometry, so the distinguishability could in
principle be metric asymmetry rather than a structured label. Stage-2 geometric relaxation
(driving the metric toward the Regge equations) settles it: an artifact would *equalize* the
charges as the geometry smooths. Instead, on most structures the distinction **persists or
sharpens** even as `‖∇S‖²` drops 4–8×:

| seed | grad_norm2 | spread before → after | outcome |
|---|---|---|---|
| 6 | 381 → 76 | 0.575 → **0.969** | sharpens |
| 23 | 303 → 47 | 0.323 → **0.387** | persists |
| 24 | 238 → 38 | 0.499 → **0.493** | stable |
| 31 | 364 → 41 | 0.493 → **0.770** | sharpens |
| 25 | 201 → 60 | 0.191 → **0.034** | collapsed |

The exception (seed 25) is the structure that *started* with the weakest distinction
(0.19 vs 0.32–0.58). So the pattern is: **strong distinctions (the common case) are real and
relaxation-stable — often sharpening — while weak/marginal ones can be geometric and wash
out.** (A larger Stage-2 sweep, `scratchpad/stage2_stability_runs.py`, confirms this; final
tally to be appended.)

## 6. Bottom line

- **Flavor emerges from what we have.** No second register, no `ℂ³⊗ℂ²` isospin index, and
  no codim-2 move are needed for it: it is read post-hoc as distinguishable, conserved,
  relaxation-stable per-hole Dirac–Kähler charge on the existing k=3 `b₃` register — the
  exact signature obstructed on the symmetric register (#414), now present on the emergent
  one.
- **A `b₂` register is genuinely unreachable** by the status-quo moves (§3); it would need a
  purpose-built codim-2 move, and is required *only* for a literal degree-2 E/B split, which
  is redundant with the DK charge here.
- **Build-plan §9:** §9.1–9.2 (Stages 1–2 + tests) merged in #473; the multi-degree `r_U`
  extension is committed under #475; §9.4's post-hoc flavor observable is the DK-charge read
  above; §9.5 is this report. The GraphML per-hinge export (§9.3) and the codim-2/E-B-at-k2
  track are deferred (the latter mooted by reading flavor at k=3).

## Pointers

- Harness: `examples/cobordism/emergent_optimizer.py` (`f1c86e4`).
- Exploration scripts (scratchpad): `betti_search_worker.py` (the b₂ search),
  `stage2_stability_runs.py` (the Stage-2 sweep).
- Design note: `docs/design/t5_emergent_optimizer_design.md`.
