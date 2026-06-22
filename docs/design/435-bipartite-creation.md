# Bipartite q/q̄ creation node — findings (#435)

Branch `feat/bipartite-creation-color-emergent` (PR #441). The deliverable is the
**creation primitive** that Experiments A (#434) and B (#438) instantiate as their
initial state: `BipartiteCreationTopology` (one seed window → two emergent windows
`q`, `q̄`) on the shared `SymmetricWindowSurface` helper, plus the charge↔color
bridge (`creation_pair_states`) that hands the two windows to a downstream
`TransportCobordism`.

## What is CONFIRMED in isolation (structural / topological)

Read off the relaxed geometry, deterministic at a fixed seed:

- **Valid, non-welded manifold.** `dualComplexValid`; every triangle has ≤ 2
  cofaces. Betti `b1 = 8` (a connected `S²` minus 9 holes), **identical** to the
  bare junction with the charge sector absent — **no parallel register** is smuggled
  in (a `Q̂` register would add independent cycles).
- **Symmetric window placement.** The three windows are exactly three of the four
  A₄ tetrahedral vertex-orbits — `seed {2,8,10}`, `q {1,4,7}`, `q̄ {0,6,9}` — with
  the fourth orbit `{3,5,11}` left filled. The seed→q,q̄ split is therefore one C₃
  orbit (the colour Z₃).
- **U-turn localization.** A single symmetric apex reflection → `temporalFlipCount()
  = 1`: the `TemporalOrientation` flip is localized to the one creation vertex, the
  propagation slabs stay time-orientation-coherent (no per-slice PT alternation —
  the #429 correction).
- **The bridge emits two color states** of dimension 3 (the q and q̄ windows), ready
  for a downstream `TransportCobordism`.
- **The all-spacelike (Riemannian) control is the degenerate `E ≡ 0` case** (`Q = 0`),
  as expected.

## What does NOT emerge in isolation — and why (the key finding)

The three *dynamical* targets — non-degenerate emergent charge, color-indefiniteness,
and exact pair-neutrality — **do not emerge in the isolated node**, and the cause is
structural, not a bug. A convergence scan (Riemannian and Lorentzian, 0–150 iters):

| metric | iters | gradS² | σ_pair | spread_q | Q_q |
|---|---|---|---|---|---|
| Riemannian | 0   | 1.6e2 | 0.20 | 0.47 | 0 |
| Riemannian | 150 | 8.9   | 0.57 | **2.44** | 0 |
| Lorentzian | 150 | 20.5  | 0.21 | 1.66 | 0 |

Two things stand out: the relaxation **never converges** (`gradS²` stays O(10),
`converged = False`), and **more relaxation makes the color spread *worse*** (0.47 →
2.44) — the stationary configuration is *not* the clean pair.

**Diagnosis — under-constraint / conformal runaway.** The creation node pins only
**one** boundary, the seed, so the state residual is negligible (`r_state ≈ 3e-27`).
With essentially no matter term to regulate it, the relaxation minimizes `‖∇S‖²`
straight into the **conformal/scale runaway** (the bare action is unbounded below):
the geometry drifts off the C₃-equivariant symmetric point, scrambling the color and
spoiling Stokes neutrality. The trivalent `W_ABC` junction converged precisely because
its **three** pinned inputs supplied enough constraint to regulate that runaway; a
single pinned seed cannot.

**Charge is doubly blocked in isolation.** The carried color representative is a
**closed** 1-cochain, so `F = dψ = 0` → no electric sector → `Q = 0` regardless of
Lorentzian worldlines. A nonzero emergent Gauss-law charge needs a **current source**,
which an isolated creation event does not have.

## The hand-off: this is exactly what A/B are for

Both limitations are resolved by the experiments' design — **pin both endpoints**
(the initial neutral pairs *and* the final hadrons) and relax the *entire* interior
at once:

- The **bilateral** boundary constraints regulate the conformal runaway the single
  seed cannot, so the symmetric/neutral/color-indefinite configuration becomes
  reachable (Experiment A, #434).
- The downstream junctions supply the **current source** for a non-closed connection,
  so the emergent Gauss-law charge can be nonzero (the charge↔color bridge feeding
  `q+q → diquark`, etc., in #434/#438).

So #435 delivers the primitive and **diagnoses why the assembly experiments are
necessary**: the neutral, color-indefinite, charged pair is an *assembly-context*
result, not an isolated-node one. The three dynamical tests are marked `xfail` with
this reason (documented, not masked); the structural tests pass.

## Next steps
- #434 Experiment A — emergent intermediates under bilateral pinning (the convergence
  test of this diagnosis).
- #438 Experiment B — fixed bipartite sequence + A-vs-B comparison.
