# Orientation-safe, `dualComplexValid`-gated stellar coning (T2, #459)

Part of the **Emergent Color Topology** epic (#457). T1 (#458) made the stellar
cone primitives hinge-exact and exactly invertible. T2 makes them
**orientation-safe** and **`dualComplexValid`-gated**, so a cone can never flip a
local induced orientation and inject a spurious sign into the **complex**
(causal / oriented) deficit — i.e. into `Im S`
(`ReggeSolver::dualReggeAction`), which is real physics.

## The sharp edge

The Lorentzian dual Regge action `S = Σ_h |★h|·ε_h` is **complex**: timelike
hinges contribute a real angle deficit, spacelike hinges a boost (`acosh`,
imaginary). A stellar cone builds `d+1` fresh top cells. If a cone quietly
produced a **non-manifold** or **non-orientable** local configuration, the
induced orientation could flip on a shared hinge and flip the sign of its
contribution to `Im S` — a corruption invisible to `|S|` or `Re S` alone. The
program's rule is to reason about the full complex action; never `Re S` alone.

## What was already safe (T1), and what T2 adds

- **Orientation convention.** The cone's new cells already take the *standard
  simplicial orientation* from `TemporalOrientation::orientationOf` (the
  Ambjørn–Loll CDT convention, computed canonically from vertex **times** — not
  an ad-hoc vertex-label sort) in the `Simplex` constructor. The
  `sortByVertexId` used by the moves is the deliberate **chain reference
  orientation** (so `∂²=0` glues), not an orientation heuristic. T1 fix #3 made
  `lorentzianDihedralAngle` evaluate in the canonical sorted-by-id frame, so the
  per-hinge deficit — and hence `Im S` — is a true relabelling invariant. For a
  *valid* refinement the deficit is therefore already order-independent.
- **What was missing.** Nothing rejected a cone whose *result* was a non-manifold
  or non-orientable complex. T2 adds that gate and the test net proving the
  primitives are orientation/`Im`-sign safe.

## Design: the gate lives in the cobordism layer

`AddMove` / `RemoveMove` live in the **`spacetime`** layer; `ChainComplex`
(`dualComplexValid`, orientation propagation) lives in **`cobordism`**, which
*depends on* `spacetime`. Putting the `dualComplexValid` gate *inside* the moves
would invert that dependency. So the gate is a thin **cobordism-layer wrapper**
that *reuses* the T1 cone primitives rather than reimplementing them — exactly
the additive path the ticket's "don't touch the action definition or the T1 move
machinery" guard points to.

### `cobordism::OrientedCone`

A propose/apply/gate/rollback wrapper over the T1 primitives
(`AddMove`/`RemoveMove` in `PachnerMode::PreGeometric`, the `1↔(d+1)` stellar
subdivision):

- `coneIn(seed)` / `coneOut(seed)` apply the underlying move, then accept it
  **only if** the resulting complex passes the gate; on rejection the move is
  rolled back and the complex is left bit-identical (a rejected cone is a no-op).
- `rollback()` delegates to the T1 move's exact inverse.
- `validate()` exposes the gate verdict on the current complex.
- `orientationCovector()` exposes the induced orientation read-out.

### The gate

A candidate complex is accepted iff **both** hold:

1. **Manifold** — `ChainComplex::dualComplexIsValid` (the #429 check: facet
   coface counts in `{1,2}`, ridge links single paths/cycles, and for `n ≥ 4`
   a recursive validation of every vertex link as an `(n-1)`-manifold).
2. **Orientable** — `ChainComplex::orientationCovector` propagates a consistent
   global orientation without contradiction.

`dualComplexIsValid` alone is **insufficient**: a non-orientable manifold (e.g.
an `RP³` region) is a valid manifold yet can reverse induced orientation. The
orientability check is the second, independent leg of the gate.

### `ChainComplex::orientationCovector(topCells)`

A new static read-out: the per-cell induced-orientation sign `ε_t ∈ {±1}` of a
whole top-cell complex, by facet-sharing sign propagation (component roots = the
lexicographically smallest cell, `+1`; across an interior facet the two induced
signs cancel, `ε_b = −ε_a·s_a·s_b`, with facet `j` of a sorted cell carrying
boundary sign `(−1)^j` — the same rule `endSignCovector` uses). Aligned to the
canonical sorted-unique (`C_d`) cell order. **Unlike `fundamentalClass`** it does
not require the complex to be *closed* (boundary facets impose no constraint), so
it reads the orientation of an open refinement region (a single cone star, a
Lorentzian CDT slab). Throws on a non-pseudomanifold or a non-orientable
contradiction — precisely the verdicts that must make the gate reject a
sign-flipping cone. It is implemented standalone (it does not refactor
`endSignCovector`, per the project's preference to leave the load-bearing
orientation read-out cores alone).

## Scope guard honored

T2 keeps the cone **topology-preserving** (refinement). The topology-changing
surgical variant (open-star removal without refill) is **T3/#460** and is not
implemented here. `r_U` / the register / the optimizer (T5) are untouched. The
action definition and the T1 move internals are unchanged — the work is purely
additive (one wrapper class, one static read-out, bindings, tests, docs).

## Verification

Build clean (`pip install -e ".[dev]"`, exit 0). Tests run under a 16-thread cap.

**New suite — `tests/cobordism/test_orientation_safe_coning.py` (14 tests, all
pass):**

- The gate **accepts** a topology-preserving cone-in on `S³` / `S⁴` / `S²×S¹`,
  and a gated cone-out (`(d+1)→1` weld of a raised apex) on `S³`; each round trip
  restores the dual Regge action (**Re and Im**) and the top-cell set.
- The induced orientation (`orientationCovector`; closed-manifold
  `fundamentalClass`) is well-defined after cone-in, and `move∘move⁻¹` restores
  **every** sign (per-cell covector bit-identical before/after).
- `endSignCovector` of a fixed hole pair is stable across a cone applied away
  from them — stated in its global-sign-invariant form (the *relative*
  induced orientation of the holes is preserved; the covector itself is defined
  only up to one overall sign per component, which a cone that removes the
  lexicographic root cell may flip).
- `Im S` sign consistency across a **gated** cone on a genuinely Lorentzian CDT
  toroid (`Im S ≈ −35`): the gate accepted the refinement and the round trip
  restored `Im S` (asserted on Re **and** Im). (Not skipped — the strict gate
  accepts a valid CDT refinement.)
- The orientability gate has teeth: `orientationCovector` **raises** on a
  non-orientable Möbius complex and returns a clean `±1` covector (lex-root `+1`)
  on an orientable `S²`.

**No regression:** `tests/cobordism/test_hinge_exact_moves.py` (T1) +
`tests/cobordism/test_epic410_invariants.py` — 30 passed; the full cobordism
suite was re-run in parallel (`pytest -n 16`) and stays green. The
orientation-safety requirement was met entirely within the existing primitives,
with no change to the action definition or the T1 move machinery — the work is
purely additive (one wrapper class, one static read-out, bindings, tests, docs).
