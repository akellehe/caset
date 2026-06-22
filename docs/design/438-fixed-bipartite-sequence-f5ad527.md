# Experiment B — the fixed bipartite sequence (pinned intermediates) (#438)

Part of the **Flavor and Charge Sectors** epic (#410). Experiment B reuses
Experiment A's (#434) connected, tube-connected (#378, never welded) event
cobordism **verbatim** and additionally **pins the intermediate state** to the
known bipartite sequence — the colored `3̄` diquark — then relaxes the whole
interior at once and asks whether that known path is the one the geometry wants.

All numbers below are read **at convergence** off the relaxed geometry (the
relaxation is a fixed point: `‖∇S‖²`, `r_state`, the diquark `σ` are bit-stable
from 40 through 300 iterations). The verdict is never read off a preliminary
checkpoint — `‖∇S‖²` is extensive in temporal volume, so it is judged against a
**per-depth** floor at the minimal depth `nL=2`.

Commit: `f5ad527`. Reproduce: `python examples/cobordism/fixed_bipartite_sequence.py`;
test: `pytest tests/cobordism/test_fixed_bipartite_sequence.py` (`@pytest.mark.slow`).

## Construction

`FixedBipartiteSequenceTopology` is a **subclass** of `EmergentEventTopology`
(#434): the `build` — the shared `SymmetricWindowSurface` (S² minus the four A₄
windows A,B,C,R) stacked over `nLayers` temporal slices by the staircase prism,
tube-connected through the hole walls — is **inherited unchanged**. Only
`readoutHoles` is overridden: it pins the same bilateral endpoints as A (the
three color-indefinite ω-rep quark inputs A,B,C at the bottom slice, the proton
color singlet `[1,ω,ω²]` at the top slice) **and additionally** pins the result
window R at every strictly-interior temporal layer to the colored `3̄` diquark.

**The diquark color.** The textbook antisymmetric `3 ⊗ 3 → 3̄` diquark is the
wedge `q_A ∧ q_B` of two definite-color quarks (`q_r ∧ q_g =` the complementary
"anti-blue", a color basis state). It **cannot** be derived by wedging the two
*pinned* inputs: the color-symmetric (ω-rep, #414 no-go) inputs are color-Z₃
phase-copies of one common direction, so `q_A ∧ q_B = 0` — two color-**indefinite**
quarks carry no definite relative color to antisymmetrize. The diquark color is
therefore the **one** thing this experiment pins: a definite, strongly-colored
anti-triplet (the canonical anti-color axis; by the A₄/color-Z₃ window symmetry
the three axes are equivalent, so the hosted-vs-floored verdict is axis-
independent), normalized to the singlet norm `√3`. Its conjugate (`3̄` /
antisymmetric) character is carried by the orientation-reversing **#416 twist**
(the induced-orientation covector is negated on the diquark window — exactly the
within-hole sign reversal A uses for the U-turn sector, never a re-welded
geometry). The endpoints stay color-emergent (no painted color at creation).

**Control — the structure is reused verbatim.** With the intermediate pin OFF
(`set_pin_intermediate(False)`), B's subclass reproduces A **bit-for-bit**:
`‖∇S‖² = 71.357`, `r_state = 0.5328` — identical to Experiment A. The only
physics difference between A and B is the pinned intermediate.

## Results (converged, `nL=2` minimal depth)

| quantity | Experiment A (#434) | Experiment B (#438) |
|---|---|---|
| `‖∇S‖²` (per-depth floor 100) | 71.4 | **49.1** ✓ regulated |
| whole-path `r_state` | 0.533 | 19.23 |
| connected-bulk carry `r_U` (3 quarks → intermediate, 12 holes) | 0.533 (→ singlet) | **0.137** (→ colored `3̄`) |
| intermediate colored content `σ` | 0.096 (weak) | **0.577** (strong, imposed) |
| intermediate singlet overlap | 0.995 | 0.577 |
| top (proton) singlet | 1.000 | **1.000** ✓ |
| `b1` / `dualComplexValid` | 11 / true | **11 / true** ✓ |
| `|Q_e|` (Lorentzian, emergent) | 0.30 | **0.117** ✓ (> 0.05) |
| `|Q_e|` (all-spacelike control) | 0.000 | **0.000** ✓ |
| `|Q_f|` (closed-surface flux) | ~1e-16 | **~1.9e-16** ✓ |
| `|Q_p + Q_p̄|` (CPT) | 0.000 | **0.000** ✓ |
| anti-proton singlet (U-turn) | 1.000 | **1.000** ✓ |

### Question 1 — is the known bipartite path realizable? **YES.**

At the minimal depth `nL=2`, the **converged** `‖∇S‖² = 49.1` is below the
pre-registered per-depth carriable floor (100) — in fact **below A's 71**. Pinning
the colored diquark in the bulk keeps the conformal runaway regulated; the known
bipartite path is geometrically supported. As predicted, `‖∇S‖²` is **extensive
in temporal volume** — the depth sweep is `49.1 (nL=2) → 158.4 (nL=3) → 268.4
(nL=4)`, so the floor is met only at minimal depth, exactly as A's
`71 → 158 → 268`. (The verdict is read at `nL=2`; the deeper points are not
failures, they are the same extensive scaling.)

### Question 2 — is the pinned colored `3̄` diquark hosted? **HOSTED.**

The decisive measure is the **connected-bulk joint carry**: the three quark
inputs carried into the colored `3̄` diquark in one carry (12 holes — the **same
count** as A's three-quarks → singlet whole path, so apples-to-apples). It is
`r_U = 0.137`, **at or below** A's connected-bulk `0.3–0.5` and far from any
free-quark-like floor (pre-registered hosted ceiling 1.0). The connected bulk
**hosts** the strong colored diquark that A produced only weakly.

> **Methodology note (a degenerate measure caught and replaced).** The ticket's
> literal phrasing — `residualForPeriods` over the diquark holes **alone** — is
> **degenerate**: a lone 3-hole window has enough edge DOF to carry any three
> target periods exactly, so it is `~1e-25` for A's quark windows, A's singlet,
> and B's diquark **alike** — non-discriminating. The reported `diquark_rU`
> (`5e-25`) is kept for completeness but the **connected-bulk joint carry**
> (`connected_bulk_rU`, `0.137`) is the meaningful hosting measure and the one the
> test asserts on.

The high **whole-path** `r_state = 19.23` (15 holes) is **not** a hosting failure:
it is the over-constrained joint demand that the *same* three quark inputs
period-carry into **both** a colored `3̄` at the middle **and** the color singlet
at the top simultaneously. Each sub-carry is individually clean (quarks → `3̄` =
`0.137`; `3̄` → singlet `≈ 0`); the tension is the genuine confinement pull
between a colored intermediate and a neutral endpoint, and `‖∇S‖²` stays regulated
through it.

### Question 3 — does A's *emergent* intermediate match B's *imposed* one? **NO — A's path diverges.**

In a common (induced-orientation, `endSignCovector`) frame the normalized overlap
is `|⟨A_emergent, B_imposed⟩| = 0.519`, well below the pre-registered agreement
threshold `0.90`. A's emergent intermediate is **singlet-dominated** (singlet
overlap `0.995`, colored `σ = 0.096` — a weak `3̄`); B's imposed intermediate is a
**strong `3̄`** (`σ = 0.577`). So left to itself the geometry does **not** crystallize
the textbook strong-`3̄` bipartite diquark — it prefers a nearly-neutral
intermediate and only weakly colors it. The bipartite sequence is therefore *a*
geometrically consistent path (it relaxes cleanly, the diquark is hosted, the
proton emerges), but **not the path A's free relaxation selects**.

### Question 4 — final singlets, charge, photons, validity, determinism.

Unchanged from A and all green: proton and anti-proton singlets `1.000`; emergent
Gauss-law charge `|Q_e| = 0.117` (non-degenerate vs the all-spacelike control's
`0.000`); closed-surface flux protected (`|Q_f| ~ 1.9e-16`); CPT total charge
exactly `0`; one emergent null edge counted; `b1 = 11`, `dualComplexValid`, no
welds; the relaxation is deterministic (a fixed point, not a sampler).

## Verdict

The known bipartite `q+q → diquark(3̄) → proton` path **is** geometrically
realizable (`‖∇S‖² = 49 < 100` at convergence) and the connected bulk **hosts**
the strong colored `3̄` (joint carry `r_U = 0.137`, at A's connected-bulk scale).
But A's *emergent* intermediate **diverges** from this imposed strong `3̄` (overlap
`0.52`; A is singlet-dominated, only weakly colored). Together A and B answer the
epic's question: the bipartite sequence is **a** consistent path the geometry will
accept, **not the** path the geometry spontaneously chooses — the free relaxation
keeps the intermediate nearly neutral rather than strongly colored.

## Criteria vs the ticket

| # | criterion | pre-registered threshold | result | pass |
|---|---|---|---|---|
| 1 | known path realizable | `‖∇S‖² < 100` @ `nL=2`, converged | 49.1 | ✓ |
| 2 | colored diquark hosted | connected-bulk `r_U < 1.0` (~A's 0.3–0.5) | 0.137 | ✓ |
| 2′ | imposed diquark is strong `3̄` | `σ > 0.30` | 0.577 | ✓ |
| 3 | geometry wants the bipartite path | overlap `≥ 0.90` | 0.519 | ✗ (xfail; documented divergence) |
| 3′ | quantify the divergence | A weakly / B strongly colored | A `σ=0.10`, B `σ=0.58` | ✓ |
| 4 | final singlets | `≥ 0.95` | 1.000 / 1.000 | ✓ |
| 4 | total charge CPT | `|Q_p+Q_p̄| ≤ 1e-3` | 0.000 | ✓ |
| 4 | emergent charge non-degenerate | `|Q_e| > 0.05`; control `≤ 1e-9` | 0.117 / 0.000 | ✓ |
| 4 | flux protected | `|Q_f| ≤ 1e-6` | 1.9e-16 | ✓ |
| 4 | validity / determinism | `dualComplexValid`, `b1=11`, deterministic | yes | ✓ |
| — | epic G1–G5 invariants | green | green | ✓ |

Criterion 3 is a **documented honest negative** (an *answer* to question 3, not a
code failure): the pre-registered agreement threshold is never loosened, the
divergence is marked `xfail` with the reason, and the quantified relationship (A
weakly vs B strongly colored, overlap `0.52`) is asserted positively. Test status:
**17 passed, 1 xfailed**; `test_epic410_invariants.py` green.

## Faithfulness (epic #410 ethos)

Emergent-first everywhere except the deliberately-pinned intermediate diquark (the
independent variable). NO parallel registers (charge = the emergent Gauss-law
holonomy `Q = ∮_S E`); the dynamics are the relaxation's `δS = 0` (not a sampler);
matter not imposed; NO dimension reduction (the full symmetric stack, #429); NEVER
welds (one connected manifold, `dualComplexValid`, `b1=11`); standard orientation
(#412); the complex action is kept (`Im S`, the Lorentzian worldlines), so the
conformal runaway is regulated by the constraint, not sidestepped. The endpoints
stay color-emergent — only the intermediate color is pinned, to test hosting.
