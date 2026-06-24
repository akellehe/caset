# Gated surgical cone-out / cone-in: the topology-changer (T3, #460)

Part of the **Emergent Color Topology** epic (#457). T1 (#458) made the cone
primitives hinge-exact and exactly invertible; T2 (#459) made the
topology-**preserving** refinement cone orientation-safe and gated. T3 adds the
genuine **topology-changer** — the move that actually grows `b_k` — and gates it
on the full manifold check.

## The sharp edge (#457's "sharp edge #1")

Pachner moves and the stellar refinement cone are topology-**preserving**: none
of the emergent color holes come from them. The genuine hole-creator is a
**surgical cone-out** — remove a top cell *without* refilling — and its inverse a
**surgical cone-in**. Surgery is powerful and dangerous: an ungated cut can
pinch the complex into a non-manifold (exactly what broke the #353 weld). The
rule is therefore: **surgery is allowed *because* it is gated**, on the full
`dualComplexValid` manifold check; never bypass it.

## What the moves do

### `cobordism::SurgicalCone` (sibling of T2's `OrientedCone`)

- **`coneOut(cell)`** — remove the top cell whose sorted vertex ids equal `cell`:
  drop the `d`-simplex, then every edge of it that no surviving top cell still
  covers (the ticket's "if the edge multiplicity > 1 decrement it, else remove
  it" — an edge's multiplicity *is* its surviving-coface count), then any vertex
  left with no incident edge. Refuses to remove the last top cell (that would
  drop the complex dimension).
- **`coneIn(targetVerts)`** — the literal "add a vertex, draw edges to `d`
  others": create a fresh apex vertex and the single top cell on
  `targetVerts ∪ {apex}`.
- **`rollback()` / `rollbackAll()`** — undo accepted moves LIFO. A cone-out's
  inverse re-creates the exact removed cell *and restores every removed edge's
  length and phase bit-exactly* — this is the surgical **cone-in of that cell**,
  and it is what makes the round trip action-exact.
- **`bettiNumbers()` / `validate()`** — read-outs the tests assert on.

The class lives in the **cobordism** layer (which depends on `spacetime`) so it
can call `ChainComplex` directly, mirroring `OrientedCone`. It reuses the
existing `Spacetime` mutators (`removeSimplex`, `removeEdge`, `removeIfIsolated`,
`createVertex`, `createSimplexTracked`) — no move machinery is reimplemented.

### The gate

Every accepted move must leave the complex a valid combinatorial
**manifold-with-boundary**: `ChainComplex::dualComplexIsValid` over the current
top cells (facet coface counts in `{1,2}` — a removed cell's facets become
boundary at count 1; ridge links single paths/cycles; the #429 recursive
`n ≥ 4` vertex-link validation). A move whose result fails the gate (e.g. a
cone-in onto an interior facet, which would give that facet 3 cofaces) is rolled
back and the complex left bit-identical; the caller sees `(False, reason)`.

## Why a single cone-out raises `b_{d-1}`, and where

Removing one open `d`-ball from a closed connected `d`-manifold kills the top
class `b_d` but leaves the result contractible-rel-boundary — `b_{d-1}` is
**unchanged**. A `b_{d-1}` hole appears only when a *second* ball, **disjoint**
from the first, is removed (the complement gains an `S^{d-1}` factor). So on
`S^3` (where `b_2` is the color register's degree) the surgical degree is `b_2`,
and the empirical sequence is:

| step | complex | Betti `[b0,b1,b2,b3]` |
| --- | --- | --- |
| start | refined `S^3` | `[1,0,0,1]` |
| cone-out cell A | punctured | `[1,0,0,0]` |
| cone-out cell B (disjoint from A) | two punctures | `[1,0,1,0]` ← `b_2` **+1** |
| inverse (cone B back in) | one puncture | `[1,0,0,0]` ← `b_2` **−1** |
| inverse (cone A back in) | refined `S^3` | `[1,0,0,1]` (restored) |

**The minimal `S^3 = ∂Δ⁴` (5 tetrahedra) has *no* disjoint cell pair** — every
facet pair shares a ridge — so no single removal can open a `b_2` hole there.
The test refines `S^3` with a dozen topology-preserving stellar subdivisions
(T1 `AddMove(PreGeometric)`) to 41 tetrahedra (still Betti `[1,0,0,1]`), where
127 disjoint pairs exist, then cones one out.

`coneIn` with a fresh apex is **homotopy-neutral** when it caps a boundary disk
(it cones a disk, adding no cycle), so it does not by itself lower `b_2`; the
clean `b_2`-lowering inverse is the exact cone-out inverse (`rollback`). On a
*closed* manifold `coneIn` always rejects (every facet already has two cofaces),
so it is only ever accepted after a cone-out has opened a boundary — the two
moves are genuinely complementary.

## Verification

Build clean (`pip install -e ".[dev]"`, exit 0). Tests run under a 16-thread cap,
parallelized with `pytest -n 16`.

**New suite — `tests/cobordism/test_surgical_cone_topology.py` (7 tests, all
pass):**

- A surgical **cone-out raises `b_2` by exactly 1** on a refined `S^3` (removing
  a cell disjoint from a first puncture), and the **inverse lowers it by 1**;
  unwinding fully restores the top-cell set and Betti vector.
- The **round trip restores the dual Regge action — Re *and* Im** — on a
  genuinely Lorentzian CDT toroid (`Im S ≈ −35`): cone-out then inverse leaves
  the complex action invariant to `< 1e-6` (observed Re drift `~1e-15`, Im drift
  `0`).
- The **gate rejects a non-manifold attempt** (a cone-in onto an interior facet
  → 3 cofaces) and leaves the complex unchanged.
- **cone-in needs a boundary** (always rejects on a closed manifold) and is
  exactly reversible where accepted.
- The **`n ≥ 4` recursive gate** accepts and round-trips a cone-out on `S^4`.
- Degenerate inputs (unknown cell, wrong arity) are rejected cleanly with no
  mutation.

**No regression:** `tests/cobordism/test_epic410_invariants.py` and the full
cobordism suite re-run with `pytest -n 16` stay green.

## Scope guard honored

T3 implements *only* the move + its gate + reversibility. **When** to fire a
surgery (the optimizer loop, T5/#462) and **whether a hole should emerge from the
physics** (T6/#463) are out of scope — here we only prove the move correctly and
reversibly changes `b_k`. Nothing hand-identifies "the hole": the move + gate are
the sole sanctioned mechanism, never an answer inserted by fiat.
