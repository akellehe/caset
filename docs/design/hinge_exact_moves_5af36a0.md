# Hinge-exact, exactly-invertible Pachner moves + cone primitives (T1)

*Ticket: #458 (epic #457, "Emergent Color Topology"). Commit: `5af36a0`.*

## What this delivers

T1 is the load-bearing prerequisite for the emergent-color-topology optimiser: the
existing CDT Pachner move suite is made **hinge-exact** and **exactly invertible**,
and the stellar `1↔(d+1)` refinement cone is validated, so that

> `move ∘ move⁻¹` leaves both the complex *and* the complex (Lorentzian / Sorkin)
> `dualReggeAction` — real **and** imaginary parts — invariant to machine precision.

Without this a greedy `ΔF` move would chase orphan-hinge / lossy-rollback / build-order
artifacts rather than the geometry. The moves themselves are **reused, not
reimplemented**; only their bookkeeping is fixed.

Everything here is `r_U`-independent: no register, no `residualForPeriods`, no
`k`-degree, no topology-changing surgery (that is #459/#460). The cone in this ticket
is the **topology-preserving** stellar refinement only.

## The four defects, and why each fix is bookkeeping (not physics)

The dual Regge action is *defined* as the geometric invariant
`S = Σ_h |★h|·ε_h` over the `(d-2)`-hinges of the triangulation. Each fix below makes
the implementation compute that invariant faithfully; none changes the definition.

### 1. Orphan hinges leaked a bare `2π` into the action (#365/#371)

`Spacetime::removeSimplex` of a top cell leaves the `(d-1)`/`(d-2)` sub-faces it had
lazily materialised (`Simplex::getFacets`) registered in `getSimplices()`. Once their
last top coface is gone they are *orphans*: `lorentzianDeficitAngle` returns a bare
`2π` for them (no top cell to subtract a dihedral from) while their gradient maps are
empty — so `dualReggeAction` counted them and `actionGradientExact` did not.

**Fix:** `ReggeSolver::collectHinges` now keeps only genuine `(d-2)`-faces of a current
top cell, via the new `Simplex::hasTopCoface`. `dualReggeAction` (and `reggeAction`,
`actionGradientExact`, the Hessians — all routed through `collectHinges`) is now a pure
function of the current top-cell set, hence exactly equal to a from-scratch rebuild.

### 2. `dualVolume` walked the wrong ambient dimension

`Simplex::dualVolume`/`Gradient`/`Hessian` found the ambient dimension `n` by walking
`getCofaces()[0]` up to a top cell. A move can leave a stale orphan facet at index `[0]`,
truncating the walk and giving a hinge the wrong `n` (and an empty gradient).

**Fix:** `Simplex::ambientTopDimension` reads `n` straight off the metric signature
when an owning spacetime is present (falling back to the coface walk for coordinate-free
fixtures). `dualVolRec` then sums correctly over the genuine up-closure; orphan facets
in a coface list contribute zero because their own recursion dead-ends.

### 3. The Lorentzian deficit was vertex-order dependent

The dihedral-cosine cofactor formula applies a sign fix `if (Cii < 0) denom = -denom`
that stands in for the `(-1)^d` diagonal-cofactor parity. With the **signed** (non-Wick)
Cayley–Menger matrix of a Lorentzian cell, `C_ii` and `C_jj` can carry *different* signs,
so the result depended on which of the two opposite vertices the cell happened to store
first. A Pachner move stores a cell's vertices in causal (not sorted) order, so the same
geometry built by a move vs. built sorted gave **different** deficits — e.g. on a fixed
`S⁴ = ∂Δ⁵` geometry, sorted/reversed/shuffled cell orders gave action `-3.11 / -1.87 /
-2.62`.

**Fix:** `lorentzianDihedralAngle` now evaluates in the canonical sorted-by-id frame
(the ChainComplex reference orientation) via the new `Simplex::cayleyMengerCanonical`,
and anchors the asymmetric sign fix on the lower canonical position. The deficit — and
the whole action — is now a true relabelling / vertex-order invariant. **Sorted-build
values are unchanged**, so every existing (sorted-build) test keeps its exact numbers;
only non-canonical orderings are brought into agreement.

### 4. `RemoveMove` was not exactly invertible (the stale-vertex strand)

`RemoveMove` deletes a vertex; `rollback` recreates it with the same id but as a **fresh
`Vertex` object** (`createVertex(id)` after `removeIfIsolated` freed the original). Any
facet/hinge materialised *before* the deletion still pointed at the **old** object. A
later materialisation reuses those sub-simplices by fingerprint (fingerprints are id-based,
so old- and new-object versions collide), so the restored star's dual/coface walk ran
over a stale, empty-list vertex — yielding a spurious bare-`2π` and a wrong dual volume.
This is exactly the documented "`RemoveMove.rollback` drifts `O(1)`". The CDT
add/flip/iflip/shift moves were unaffected because they never remove-and-recreate a vertex.

**Fix (two layers):**

- `RemoveMove::apply`/`applyPreGeometric` now drop the orphaned sub-simplices incident to
  the deleted vertex (`removeIncidentSubSimplices`) before removing it, so rollback's
  re-materialisation builds clean facets/hinges that reference the recreated vertex.
- Defence in depth: `lorentzianDeficitAngle`, its gradient, and `hasTopCoface` now scan
  **all** of a hinge's vertices for incident top cells (`Simplex::incidentTopCells`,
  deduped by fingerprint, membership tested by id), so a single stale pointer can no
  longer mask a genuine top coface. Identical to the single-vertex scan whenever no
  vertex is stale.

## Cone primitives

The stellar `1↔(d+1)` refinement cone already exists as the **pre-geometric**
`AddMove` (cone-in: `1→(d+1)`) and `RemoveMove` (cone-out: `(d+1)→1`). They are reused,
not reimplemented; new cells are built in the canonical orientation. Topology-changing
surgical coning is out of scope (#460). Round-trip tests exercise both
`AddMove(PreGeometric).apply`/`.rollback` (deterministic cone-in/out) and the standalone
pre-geometric `RemoveMove` (cone-out) inverted by its rollback.

## New API

| Symbol | Role |
| --- | --- |
| `Simplex::hasTopCoface()` | genuine-hinge predicate (a face of some current top cell) |
| `Simplex::incidentTopCells()` | deduped top cells containing this simplex, stale-pointer-robust |
| `Simplex::ambientTopDimension()` | ambient `n` from the metric (robust dual-volume walk) |
| `Simplex::cayleyMengerCanonical()` | bordered CM in the sorted-by-id reference frame |
| `Spacetime::pruneOrphanedSimplices()` | restore the registry to the exact top-cell closure |
| `RemoveMove::removeIncidentSubSimplices()` | drop a deleted vertex's orphaned sub-faces |

## Round-trip residuals achieved

`tests/cobordism/test_hinge_exact_moves.py` (22 tests, all green):

- **Action invariance** `|A(move∘move⁻¹) − A|`: `≤ ~1e-15` on the minimal spheres,
  `≤ ~1e-9` on the larger CDT(250) builds (summation reassociation over more hinges).
  Both Re and Im are asserted (the CDT toroid fixture has `Im S ≈ −35`, so Im-sign
  stability under coning is a live test, not vacuous).
- **Complex invariance:** the top-cell set and the genuine-hinge set are restored
  identically; after `pruneOrphanedSimplices` the *raw* registered simplex set is
  bit-identical.
- **Coverage:** every move type (add/remove/flip/iflip/shift) + stellar cone-in/out, on
  `S³ = ∂Δ⁴` (the proton's spatial slice), `S⁴ = ∂Δ⁵` (the epic host), `S²×S¹`, and CDT
  toroids of sizes 40/120/250; a 5-deep stack of cones inverted LIFO with the action
  retraced at *every* level; `Im S` preserved under coning and flips.

## Verification

- Round-trip suite: 22/22 pass.
- Existing suites stay green (the canonical-frame and `incidentTopCells` changes leave
  sorted-build values bit-unchanged): analytic action gradient, retriangulation
  consistency, Lorentzian Regge, Regge solver, the five Pachner move suites, and
  `test_epic410_invariants.py`.

## Notes / follow-ups

- `pruneOrphanedSimplices` is offered as an explicit utility; the moves do not auto-prune
  the wider region, so a long optimiser run (T5) may want to call it periodically. The
  *action* never needs it (orphans are already excluded), it is only for raw-set identity.
- `incidentTopCells` made the deficit and its gradient stale-pointer-robust; the
  `dualVolume` family is robust via `ambientTopDimension` plus the per-vertex cleanup in
  `RemoveMove`. No move currently leaves a stale coface that `dualVolRec` cannot absorb.
