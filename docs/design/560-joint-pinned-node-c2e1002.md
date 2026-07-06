# #560 — the joint 3-pair formation node with the final-state knob

Findings note for issue #560 (part of epic #559, P1). Engine: `c2e1002` (main
at branch time — the #594 C++ observable battery included). Experiment layer:
`examples/cobordism/joint_pinned_proton.py` on branch `feat/joint-pinned-node`
(PR #596). **No core component was modified**: `Proton`, `ProtonIngredients`,
and `MultiCobordism` are composed through their public surface only.

## What this arm is

ONE co-optimized `MultiCobordism` — the #489 shape through the canonical
engine:

* **inputs** — the Z₃-orbit neutral triple `{1,−1,0} ⊔ {0,1,−1} ⊔ {−1,0,1}`
  (each Σ = 0), seeded at v0/v1/v2 of the single Δ⁴ seed, held representable
  through their r_U terms for the whole build;
* **outputs** — **pinned**: baryon `{1,ω,ω²}` seeded at v3, antibaryon
  `{1,ω̄,ω̄²}` at v4 (ω = exp(2πi/3)).

Exactly one variable differs from the canonical `Proton::build()` — the event
graph (joint vs two-step) — and exactly one from the campaign's inputs-only
`ProtonIngredients::jointNode` — the final-state knob (pinned vs empty
outputs). That makes this node the **control arm** between the two: it can
never claim an emergent singlet (the answer is pinned), but it measures
joint-shape **feasibility** independent of basin rarity, its A/B against the
canonical two-step isolates what the event graph alone changes, and its
specimens (dumped with output-block provenance) exercise the C++ observable
battery on real, singlet-driven joint geometry.

**Microcausality verdict** (the #559 decision-log criterion, recorded as
pre-registered): causal structure lives in the **move history** — every
accepted change is one gated local move — not in the boundary-block count.
The joint node is physically admissible; nothing in this experiment weakens
that verdict.

## Method

* **Construction** (`joint_pinned_node`): the engine's minimal seed rebuilt
  from the public surface (one Δ⁴ pentatope, uniform all-spacelike ℓ² = +1 —
  no causal structure initialized; the un-run node's r_U is the exact
  full-leak arithmetic `6 + w·6`, verified affine in the input weight to
  machine precision), then `MultiCobordism(host, pairs, [baryon, antibaryon],
  degrees=[3], …)` with `seed_inputs(v0,v1,v2)` / `seed_outputs(v3,v4)`.
  Blocks start as the seed vertex's cell-neighbourhood (the whole pentatope on
  a one-cell host) and differentiate as `runStage1` grows them — identical to
  the engine's own factories.
* **Drive**: the campaign worker's recipe verbatim — init pass
  (`grow_boundaries=True`) → evolution pass (∂W frozen) → stage-2 chunks to
  the engine's relTol=1e-9 stationarity test → persistence passes (holes, b₃,
  F stable). Battery-scale honesty cap: stage-2 attempts stopping on the
  iteration cap are recorded `stationary=False`, never promoted.
* **Faithful records**: the engine build is not process-deterministic, so
  every attempt writes a schema-1 geometry dump (the frozen campaign writer)
  whose metadata carries the output-block provenance
  (`{label, vertices, target}` per block) — `observe_proton_ingredients.py
  --geometry` runs the landed C++ battery (`BlockResiduals` included) on any
  attempt after the fact.

### Pre-registered criteria (from the ticket) and reads

1. **Convergence rate ≥ the two-step's** on the same seed list and budgets.
   Convergence is the answer-agnostic verdict (stationary ∧ persistent) for
   both arms; the canonical answer-shaped gate (whole singlet < 0.5 ∧ ≥ 3
   holes) is reported alongside for reference.
2. **≥ 3 emergent holes clustered on the baryon block's region** — the count
   of `emergent_holes` on the baryon block's own sub-complex (ambient top
   cells fully inside the block's vertex region, via the canonical
   `fromCells`, uniform metric — the `BlockResiduals` scoring rule).
3. **Baryon-block singlet residual < 0.5** — `r_state` of `{1,ω,ω²}` against
   the block sub-complex.
4. **Antibaryon block carries the conjugate** — `r_state` of `{1,ω̄,ω̄²}` < 0.5
   on its sub-complex (CPT pairing).
5. **Per-hole charge spread ≥ 0.1** — *documented substitution*: the ticket
   names the per-hole DK charge, whose Dirac–Kähler readout was retired in
   #509. The read here is the pairwise period fit — hole 0 as reference,
   hole j carries charge `m_j = argmin_m residualForPeriods([h₀,h_j],
   [1, ω^m])` (a single hole's phase is gauge; only relative phases are
   physical) — and the criterion is evaluated on the spread of the carried
   phases `2πm_j/3`, with the winner→runner-up residual margins recorded.

### Calibration (pre-registered choice rule)

Γ ∈ {20, 50, 100} × input weight w ∈ {5, 20} on 2 seeds, short budgets
(the r_U scale changes with 5 blocks). Choice: among configs with ≥ 1
stationary attempt, the lowest mean output-block residual sum; ties → lower Γ,
then lower w.

## Results

### Smoke (seed 42, reduced budgets: init 60 / evolve 30 / stage-2 cap 400 / 1 persistence pass)

The full path runs end to end — node → drive → verdict → schema-1 dump →
rehydration → the landed C++ battery — and the C++ `BlockResiduals` reproduces
the module's mirror read **to the printed digit** (1.233e-31, 49 cells per
block), with GAUGE/RELABEL gates green.

The physics of the attempt is the first finding:

* **All five blocks carried at machine precision on a b₃ = 2 whole** —
  `r_u = 7.8e-30` with the whole's singlet diagnostic at exactly 1.0 (two of
  three components). 54 cells, 800 stage-2 iterations (cap-stopped, recorded
  `stationary = False`), 1h44m single-threaded.
* **The two pinned output blocks did not separate — they merged into the
  IDENTICAL region**: vertex Jaccard 1.0, both blocks = the same 49 of 54
  cells (91% of the complex), whose own sub-complex carries **3** holes
  (while the whole has b₃ = 2), and whose carried period space fits the
  baryon target *and* the conjugate antibaryon target **exactly**
  (residuals 1.2e-31 each). Nothing in the objective penalizes overlap, so
  "baryon ⊔ antibaryon" is satisfiable by one delocalized fit.
* Consequence for the pre-registered criteria: 2–4 **pass as written** on
  such a fit while describing no separated baryon. The criteria are evaluated
  as pre-registered, and every record now carries the block overlap
  (`vertex_jaccard`, shared cells) and `region_fraction`, so a delocalized
  pass is visible as such. Separation statistics are a battery deliverable.
* **All-spacelike again**: `im_max = 0`, `re_min = 0.044` — no causal content
  emerged, consistent with the campaign's uniform finding on the inputs-only
  arm.
* The pairwise phase probe measured **degenerate** on rich carried spaces
  (winner→runner-up margins ~1e-29 on a b₃ = 2 campaign specimen): a 2-hole
  fit is exactly solvable for every Z₃ phase. It is recorded as a diagnostic
  with an explicit `degenerate` flag; the criterion-5 read is the landed
  `PairLoopFlavor` joint read (per-hole DK charges `q`), exactly as the
  ticket names it.

### Two-step smoke (seed 44, same reduced budgets)

The canonical arm's first battery-recipe attempt produced a **full 3-hole,
machine-precision singlet-carrying specimen**: b₃ = 3, whole singlet residual
6.7e-31, `canonical_converged` (singlet < 0.5 ∧ ≥ 3 holes), step B genuinely
stationary (persistence not yet — one continued pass still moved the summary;
110 cells, 2h37m for both nodes). Two reads on it matter beyond the A/B:

* **The full `PairLoopFlavor` read ran on a real specimen**: oriented per-hole
  weights exactly `[1, ω, ω²]` (root-fixed convention), pair-loop duality
  `[γ_ij] = −[k]` at ~2e-16, per-hole DK charges `q = [0.0111, 0.0181,
  0.0183]`, and the **2:1 multiplicity verdict TRUE** with ρ = 0.021 ≪ 0.5
  (odd loop [1,2], dual hole 0; odd-vs-diquark not evaluable — the two-step
  drive has no hole-level diquark provenance, honestly reported).
* **Criterion 5's threshold is mis-scaled for the read it names**: the q
  spread on this textbook specimen is 0.0071 — the 2:1 *structure* is
  unambiguous (ρ), but the ticket's `spread ≥ 0.1` fails on it. The battery
  reports the as-written verdict alongside ρ; the threshold's scale is a
  findings item, not something this experiment silently rewrites.
* **The whole-complex r_state is target-insensitive on full-rank registers**:
  `singlet` = `singlet_conj` = 6.7e-31 — with 3 holes and 3 independent
  harmonics, ANY 3-vector target fits exactly, so the whole-singlet diagnostic
  cannot distinguish the baryon from its conjugate there. Content lives in
  the orientation-fixed flavor read and the block reads.

### Calibration grid

*(pending — filled by `aggregate --by config` over the 12 grid attempts;
early returns: stationarity+persistence in as little as 75 stage-2 iterations
at Γ=100/w=5; Γ=20/w=20 budget-stopped with both blocks at full leak)*

### A/B battery

*(pending — filled by `aggregate --by arm` over the fixed-seed battery)*

### Criteria verdicts

*(pending)*

## Interpretation

*(pending — the smoke's delocalization finding stands regardless: with pinned
conjugate output targets and no separation term, the joint node can satisfy
both pins with one shared region; whether separated blocks EVER emerge under
this shape is exactly what the battery measures)*

## Reproduction

Attempts are **labeled** by seeds, never reproduced by them (the engine build
is not process-deterministic); the geometry dumps attached to #560 via the
issue-attachments release are the faithful records. Commands:

```bash
# one attempt (either arm)
python joint_pinned_proton.py attempt --arm joint-pinned --seed 101 \
  --out battery.jsonl --geometry-dir geometry/
python joint_pinned_proton.py attempt --arm two-step --seed 101 --out battery.jsonl

# criteria + rates
python joint_pinned_proton.py aggregate battery.jsonl --by arm

# the C++ observable battery on a dumped attempt
python observe_proton_ingredients.py --geometry geometry/joint_pinned_seed_101_geometry.json

# the animation harness (labels only)
python emergent_proton.py --joint-pinned --save joint_pinned.gif
```
