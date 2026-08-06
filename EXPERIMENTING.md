# Running experiments in this repository

This document records mistakes that have actually been made here, and the rules
that prevent them. Every rule below is followed by the specific incident that
motivated it, so the cost of ignoring it is concrete rather than hypothetical.

The failures collected here share one shape: **a number was produced, believed,
and reported before the thing it described had been established.** The rules are
mostly about closing that gap.

---

## 1. Re-read state after any operation that may replace it

`MultiCobordism::step` commits an accepted move by *replacing* the node's
complex (`spacetime_ = build(bestSnapshot)`), not by mutating it in place. A
handle captured from the `st` property before a drive call therefore continues to
describe the old complex while the node moves on. The failure is silent: every
read succeeds and returns self-consistent values for a complex the node no longer
holds.

```python
# Wrong: st describes the pre-drive complex
st = node.st
node.run_stage1(180, 8, 15, True)
holes = len(MultiCobordism.emergent_holes(st, 3))

# Right: re-read after each drive call
node.run_stage1(180, 8, 15, True)
st = node.st
holes = len(MultiCobordism.emergent_holes(st, 3))
```

`objective()` and `r_u(st)` are especially treacherous together, because
`objective()` reads the node's live complex internally while `r_u(st)` reads
whichever complex it is handed. Mixing them against a cached handle produces
figures that cannot be reconciled with each other — which is how this was
eventually noticed, rather than by the readings looking wrong.

> **Incident.** A cached handle produced the report "`r_U` pinned at the full-leak
> floor of 120 with zero emergent holes, host plateaued at 104 cells." The true
> values were `r_U` reaching `2.5e-30` with the host growing from 1 to 62 top
> cells. The claim reached a merged commit message before being corrected.

---

## 2. Report no number without its qualifying checks, in the same run

A measurement of a complex means nothing unless that complex is a valid manifold
and its register carries. Both must be established **in the same run that
produced the measurement**, not in a separate one and not from memory.

Report, on every row: manifold validity (`SurgicalCone::validate`), the register
residual `r_U`, and then the quantity of interest. If a row cannot carry its
checks, it is not a result.

> **Incident.** A figure of "43.82% of simplices carry a negative Hodge star
> ratio" was reported with no validity or carry check attached. It invited an
> explanation in terms of signature effects. When the checks and the cell shapes
> were added, the actual cause turned out to be malformed cells — a different
> conclusion entirely.

---

## 3. Establish what a quantity measures before drawing conclusions from it

Read the implementation of anything being measured. In particular, confirm that
the quantity is computed over the objects you believe it is computed over.

> **Incident A.** Negative dual volumes were investigated using
> `min(circumcenterBarycentric()) < 0` on each simplex — that simplex's *own*
> circumcentre. The dual-volume recursion walks its **cofaces'** circumcentres. At
> `k = 1` an edge's circumcentre is its midpoint and is always interior, so the
> column read zero while ratios went negative. This produced the finding
> "well-centeredness does not track the negative ratio," which was an artifact of
> counting the wrong circumcentres. Measured against cofaces, it tracks closely.
>
> **Incident B.** Even the corrected predicate was too weak. The sign is set by
> **one specific** barycentric coordinate — the vertex opposite the facet — and
> `min(barycentric) < 0` is implied by it but does not imply it. The weaker
> condition overcounts.

---

## 4. When a quantity is a product, check every factor

A compound expression can acquire a sign or a magnitude from more than one
place, and the causes may have entirely different meanings.

> **Incident.** Dual heights are formed as
> `h = oppositeVertexSign(cf, s) * signedSqrt(R²(cf) - R²(s))`. The barycentric
> factor going negative is a mesh defect; the radicand going negative is a
> timelike separation, which is correct physics. Only the first factor was
> examined, after which the negatives were asserted to be "not benign" — a claim
> with no evidence for the half that had never been looked at. The eventual
> measurement was 96 defects against 25 timelike separations: the conclusion
> survived, the reasoning had not supported it.

---

## 5. Use the construction the codebase uses; do not reimplement it

Node construction lives in factories — `ProtonIngredients::jointNode`,
`Proton::formationNode`, `Proton::recombinationNode`. They fix the host, the
targets, the seeding, and the residual weight. Reproducing their internals by
hand in an experiment script reliably gets one of those wrong.

If a factory does not expose a parameter an experiment needs, the correct fix is
to add the pass-through to the factory, not to rebuild the node externally.

> **Incident.** An experiment needed a constructor flag the factories do not
> expose, so their internals were copied into Python. The copy used the wrong
> host, and — more seriously — fed the colour singlet as an *input* with empty
> outputs, when `formationNode` pins it as an **output target** and reads it off
> the whole cobordism. That is a different node, and its residual measures a
> different thing.

---

## 6. Hold semantics fixed across anything you intend to compare

Two runs are comparable only if everything except the variable under study is
identical: the same drive schedule, the same tolerances, the same step budgets,
the same construction. Changing the schedule between runs invalidates the
comparison even when both runs are individually correct.

> **Incident.** A dual-volume decomposition and a gradient split were run with a
> single `run_stage2` call, while the convergence results they were compared
> against used a stage-pair loop at a different tolerance. Both measurements had
> to be discarded and repeated.

---

## 7. Converge before concluding, and say which convergence you mean

"Converged" is ambiguous and the distinction matters. A drive reaching a fixed
point of its own iteration (`|ΔF| = 0` because no move is accepted) is not the
same as reaching stationarity (`‖∇S‖² → 0`). Report which one occurred.

Measurements taken far from convergence describe transients, not the object of
study.

> **Incident.** Fifteen Lorentzian-inadmissible cells and two degenerate cells
> were measured at `stage2_max_iters = 10` and used to argue that a validity gate
> was needed. At a converged tolerance the same configuration produced zero of
> each. The evidence for the proposal was an artifact of stopping early.

---

## 8. Do not add a filter that removes the phenomenon under test

If the question is whether a process produces some structure on its own, adding a
mechanism that rejects the counterexamples makes the question unanswerable. Run
the experiment first; introduce the mechanism only if the result requires it.

> **Incident.** A gate rejecting Lorentzian-inadmissible cells was nearly wired
> into the move-acceptance path while the open question was whether convergence
> alone yields consistent cells. Had it been added, a clean result would have been
> indistinguishable from the filter doing the work. Two later measurements showed
> the gate was also unnecessary: the pathological cells were transients, and
> admissibility turned out to be independent of the property being investigated.

---

## 9. Distinguish the level at which a condition applies

Conditions stated about an operation are not conditions on each of its parts.

> **Incident.** The realizable gate actions are those whose holonomy block
> conserves total charge. This was checked per input block: the diquark `{1, ω}`
> sums to `1 + ω ≠ 0`, so the configuration was declared invalid and a valid run
> was discarded. The operator is what must conserve charge, and it does —
> `1 + ω + ω² = 0` in, singlet `0` out. The diquark is a coloured `SU(3) 3̄` and is
> *expected* to be individually charged.

---

## 10. Identify the cause of a difference before attributing it

When two runs differ, determine which changed variable produced the difference.
Attributing it to the most salient change is not the same as establishing it.

> **Incident A.** A host construction was changed and, separately, a step budget
> was reduced for a quick test. The resulting difference in cell counts was
> attributed to the host. Running both at full budget produced bit-identical
> results to six significant figures; the difference had come from the budget.
>
> **Incident B.** Continuous integration failures were attributed to wedged
> self-hosted runners, then to concurrency configuration. Two runners genuinely
> were wedged and were repaired, but the failures continued, and the actual cause
> was a platform outage. Some of the intermediate cancellations were caused by
> repeatedly re-triggering the workflow, which placed multiple runs in one
> concurrency group where they cancelled one another.

---

## 11. Correct the record where the claim was made

Incorrect results propagate into commit messages, pull request descriptions and
issue comments, which later work then builds on. When a reported result turns out
to be wrong, post the correction where the original claim lives, state plainly
what was wrong, and say what still stands.

Distinguish three cases explicitly, because they have different consequences for
work already done:

- the conclusion was wrong;
- the conclusion holds but the reasoning did not support it;
- the measurement was valid but described something other than what was claimed.

---

## Checklist

Before reporting any experimental result:

- [ ] State was re-read after every operation that could replace it.
- [ ] Manifold validity and register carry are reported on the same row.
- [ ] The implementation of each measured quantity has been read.
- [ ] Every factor of a compound quantity has been examined.
- [ ] Node construction came from the factories, not from a reimplementation.
- [ ] Anything being compared differs only in the variable under study.
- [ ] The convergence state is reported, and which kind of convergence it is.
- [ ] No filter has been added that would remove the phenomenon under test.
- [ ] Conditions are applied at the level at which they are stated.
- [ ] Any difference is attributed to a cause that has been established.
