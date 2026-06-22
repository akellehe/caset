# Experiment A — emergent intermediates (#434)

**Epic:** Flavor and Charge Sectors (#410). **Companion:** Experiment B (#438, pins a
known intermediate sequence; reuses this structure). **Prerequisite:** the bipartite
creation primitive (#435).

## The question

Build the whole `q/q̄ → proton (& anti-proton)` color event as **one** connected,
tube-connected (#378, never welded) cobordism over several **temporal slices**; pin
**only** the initial state (three color-symmetric quark inputs) and the final state
(the proton color singlet); relax the **entire interior at once**; and read the
**emergent** intermediates off the relaxed geometry. This is the direct test of the
#435 finding: the **isolated** creation node pinned only ONE boundary, so
`r_state ≈ 3e-27` gave the conformal runaway no restoring force, `‖∇S‖²` stayed stuck
~O(10), and the emergent charge was degenerate (`Q = 0`). The epic predicts that
**bilateral** pinning (both endpoints) supplies the missing constraint. Experiment A
tests that prediction.

## What was built (the reusable builder)

`EmergentEventTopology` (`include/cobordism/EmergentEventTopology.h`,
`src/cobordism/EmergentEventTopology.cpp`) — a `TopologyBuilder` whose `readoutHoles`
pins **both** endpoints of a deep temporal stack:

- **Base:** the shared `SymmetricWindowSurface` (`S²` minus the four `A₄`-tetrahedral
  `C₃`-symmetric windows A,B,C,R of three color holes each) — the same validated
  symmetric-window geometry as the proton's `TripartiteRegisterTopology`.
- **Temporal depth:** the holed surface is stacked over `nLayers` layers by the
  dimension-generic staircase (`Spacetime::prismCells`), giving ONE connected
  3-complex whose layers are **tube-connected** through the shared hole-tube walls
  (#378). A window's three color holes at temporal layer `ℓ` are the base holes
  shifted by `ℓ·stride`, so each window's period is readable at **every** slice.
- **Bilateral pin:** `readoutHoles` pins the three input windows A,B,C at the
  **bottom** layer (`ℓ=0`) AND the result window R at the **top** layer (`ℓ=nLayers`),
  both over the EXACT `residualForPeriods` signed by the induced-orientation covector.
  The **middle** layers are pinned nowhere — the variable interior whose intermediates
  emerge. The canonical emergent intermediate (window R at the middle slice) is
  returned as `resultHoles` and read directly; `windowHolesAtLayer` exposes any window
  at any slice for the per-slice read-out.
- **Lorentzian / charge:** `setLorentzianWorldlines` makes the cross-layer worldline
  edges timelike (`Im S ≠ 0`, the electric sector non-empty). `setUTurnTwist` reverses
  every window's induced-orientation covector (the anti-baryon / anti-proton sector,
  opposite charge — the readout-level form #435 uses for the antiquark window).

No relaxer change was needed: bilateral pinning falls straight out of the existing
`CobordismRelaxer::relaxInterior` by returning both endpoints as pinned `inputHoles`.
The dynamics stay `δS = 0` (no sampler), matter is not imposed
(`MatterConfiguration()` empty), the complex action is kept, the dimension is not
reduced, no weld is introduced (`dualComplexValid` throughout), and color is never
painted (the inputs are the frame-symmetric `ω`-rep, the singlet emerges).

The worked example is `examples/cobordism/emergent_intermediates.py`; the falsifiable
tests are `tests/cobordism/test_emergent_intermediates.py`.

## Results (fixed seed; pre-registered thresholds)

Primary configuration: the **minimal** temporal depth `nLayers = 2` (slices 0, 1, 2;
slice 1 is the one emergent middle slice). Inputs: the `ω`-representation colored
quarks on A,B,C; the color singlet `[1, ω, ω²]` on R.

| quantity | proton sector | meaning / threshold |
|---|---|---|
| `‖∇S‖²` | **71.4** | **< 100** (pre-registered carriable floor) — **MET** |
| `r_state` (`r_U`) | 0.53 | < 1e-3 — **not met** (the bilateral colored→singlet residual) |
| top (final) singlet | **1.000** | ≥ 0.95 — **MET** (the proton emerges) |
| emergent charge `|Q_e|` | **0.297** | > 0.05, non-degenerate — **MET** (#435 got 0) |
| closed-surface flux `|Q_f|` | 1.7e-16 | < 1e-6 — **MET** (topological protection) |
| total charge `|Q_p + Q_p̄|` | 0.000 | < 1e-3 — **MET** (CPT, anti-proton sector) |
| all-spacelike control `|Q_e|` | ~0 | the degenerate `E=0` case the Lorentzian beats |
| middle-slice color | singlet 0.995, `σ` 0.096 | a small colored component, hosted in bulk |
| `b₁` / `dualComplexValid` | 11 / true | no smuggled register/holes; no weld |
| null edges | 0 | no spontaneous photons (a real photon needs a source, #413) |

**`‖∇S‖²` is extensive in the temporal volume** (a sum over interior edges):

| depth | `‖∇S‖²` | interior edges | per-edge | top singlet | `|Q_e|` |
|---|---|---|---|---|---|
| proton baseline (single-end, #400) | 44.9 | 354 | 0.127 | 1.000 | — |
| `nL=2` | 71.4 | 264 | 0.270 | 1.000 | 0.297 |
| `nL=3` | 158.5 | 438 | 0.362 | 1.000 | 0.260 |
| `nL=4` | 267.6 | 612 | 0.437 | 1.000 | 0.281 |

So the absolute floor is met at minimal depth and crossed by `nL=3`; the per-edge
strain (~0.27) is ~2× the working single-end junction's (0.127) and grows with depth.

## Pass / fail against the ticket's criteria

1. **Connectivity / convergence — PARTIAL.** The stationary-action floor `‖∇S‖² < 100`
   is **met** at minimal depth (71 < 100): bilateral pinning **regulates the conformal
   runaway** that left the singly-pinned #435 node stuck. The *full* realizability floor
   (also `r_U < 1e-3`) is **not met** — `r_U ~ 0.3–0.5`. Per the ticket's rule the floor
   is **not loosened**: meeting the full floor is an honest **negative**. The residual is
   physical, not numerical (see Interpretation).
2. **Charge / Stokes conservation — PASS.** The closed-surface flux is protected to
   round-off (`|Q_f| = 1.7e-16`); the proton (+) and anti-proton (−, U-turn sector)
   emergent charges cancel to `|Q_p + Q_p̄| = 0.000` (CPT).
3. **Emergent intermediate — PARTIAL.** The middle (unpinned) slice carries a
   well-defined emergent color content read off the relax (never hand-placed), with a
   nonzero colored component `σ ≈ 0.096` hosted in the bulk (`r_U` low, not floored).
   But it is **singlet-dominated** (overlap 0.995): the geometry does **not** host a
   strong transient `3̄` diquark. The colored component is small.
4. **Final singlets — PASS.** The pinned top slice overlaps the color singlet `1.000`
   for both the proton and the anti-proton.
5. **Color crystallization — WEAK.** Singlet overlap is measurable per slice and the
   top is maximal, but the emergent intermediate is **already** ~99.5% singlet, so the
   increase (0.995 → 1.000) is small — not a strong `≪1 → 1.0` crystallization curve.
6. **Photons — none observed.** Zero null edges in the neutral symmetric propagation
   (consistent with #413: a real photon needs a symmetry-breaking source).
7. **Validity / determinism — PASS.** `dualComplexValid` throughout; `b₁ = 11` (the
   same as the tripartite junction, no smuggled holes/registers); no welds; bit-for-bit
   deterministic at the fixed seed.

The epic invariants `tests/cobordism/test_epic410_invariants.py` (G1–G5) stay green
(5 passed).

## Interpretation — the headline, and the honest negative

**The headline positive:** bilateral pinning does what the #435 finding predicted on the
two observables the isolated node failed.

- The **conformal runaway is regulated**: `‖∇S‖²` reaches the carriable floor (71 < 100)
  at minimal depth, where the singly-pinned node stayed stuck.
- The **emergent Gauss-law charge is non-degenerate** (`|Q_e| = 0.297`) and conserved
  (flux protected, CPT total 0), where the isolated node got `Q = 0`. The all-spacelike
  control stays degenerate (`E = 0`), so the charge is genuinely the Lorentzian electric
  sector, read off the relaxed connection — not a parallel register.
- The **proton singlet emerges** at the pinned top slice (1.000) from frame-symmetric
  colored-quark inputs — color is not painted.

Crucially, the bilateral constraint is **genuine**, not the degenerate singly-pinned
case: with the physical `ω`-rep colored input the state residual is `r_U ~ 0.5`, far
above the #435 node's `~3e-27`. That nonzero residual is the **source** of the
non-degenerate emergent charge — and it is also why the *full* realizability floor is
not met: three **colored** quarks pinned at the bottom cannot be period-carried into a
**singlet** pinned at the top with zero residual. The colored→singlet gap is the
discrete confinement strain of the bilateral constraint, not an optimizer failure
(`‖∇S‖²` plateaus, flat across 15→80 iterations).

**The honest negatives, plainly:**

- The full realizability floor (`‖∇S‖² < 100` **and** `r_U < 1e-3`) is not met; `r_U`
  stays `~0.3–0.5` at every depth.
- `‖∇S‖²` is extensive in the temporal volume and crosses the floor by `nL=3`, with a
  per-edge strain ~2× the single-end junction.
- The emergent intermediate is **singlet-dominated** (0.995) with only a small colored
  component (`σ ≈ 0.096`): the construction transports three colored quarks toward the
  singlet, and the singlet dominates the carried representative almost immediately — it
  does **not** stage a strong transient `3̄` diquark in the bulk. Color crystallization
  is correspondingly weak.

## Open questions handed forward (to #438 / the epic)

- **The `r_U` residual is intrinsic to colored→singlet bilateral pinning.** Driving it
  to the realizability floor needs a lever the period residual does not provide (`r_U`
  is the period non-harmonicity; the colored→singlet gap is a real obstruction). A
  scale/charge-sensitive interior term, or a symmetric-apex temporal stack (vs the
  `prismCells` staircase, which carries ~2× the per-edge strain), are the candidate
  levers — but neither should be adopted without checking it does not paint the answer.
- **A strong transient `3̄` diquark** would need the literal staged event graph
  (`q+q → 3̄`, then `3̄+q → singlet`) laid out as windows at distinct slices, rather than
  the four symmetric windows carried as vertical tubes. The reusable `EmergentEventTopology`
  exposes `windowHolesAtLayer` for any window at any slice, the hook #438 can use to
  pin a known intermediate sequence and compare against this emergent result (the A-vs-B
  comparison).
- **Proton + anti-proton on ONE connected manifold.** The `A₄` symmetric-window orbit
  caps at four windows (one baryon). Here the two sectors are the untwisted and
  U-turn-twisted builds, with the charge cancellation measured across them; a single
  8-window manifold is a follow-up.

## How to reproduce

```
git submodule update --init third_party/itensor
python3 -m venv .venv-build
OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 .venv-build/bin/pip install -e ".[dev]"
.venv-build/bin/python examples/cobordism/emergent_intermediates.py
.venv-build/bin/python -m pytest tests/cobordism/test_emergent_intermediates.py tests/cobordism/test_epic410_invariants.py
```
