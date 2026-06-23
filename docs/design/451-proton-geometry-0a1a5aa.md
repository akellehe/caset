# Geometric proton validation on the Experiment-B emergent interior (#451)

**Epic:** Flavor and Charge Sectors (#410). **Builds on:** the bilateral-pin event
(`EmergentEventTopology`, #444), the fixed-bipartite-sequence event
(`FixedBipartiteSequenceTopology`, #438/#445), and the final-t quantum-number pass
(#449/#450). **Example:** `examples/cobordism/event_proton_geometry.py`.
**Test:** `tests/cobordism/test_event_proton_geometry.py`.

## The question

PR #450 (#449) established that the Experiment-B event puts a real proton on its **top
slice** by its *quantum numbers* (color singlet → 1, color σ → 0, one baryon, charge
CPT-conjugate to its antiproton). It deliberately left the *metric* mass, radius, and
curvature **out of scope**, because the top slice is a **pinned Dirichlet boundary**:
its spatial edges are frozen at the uniform seed `l² = 1` and carry no relaxed
information. A prior pass quoted `r·m ≈ 8.8` (within ~2× of the physical proton's
`m_p·r_p/ħc ≈ 938·0.84/197 ≈ 4.0`) but read it off that frozen slice — the radius was
`√⟨l²⟩` over edges that are mostly pinned at 1, i.e. not a measurement at all.

This note measures mass, radius, and curvature **the right way**: on the **relaxed
emergent worldtube**, the strictly-interior simplices between the frozen bottom (quark)
and top (proton) slices — the only cells the relaxation actually moves — at the
converged carriable depth **nL = 2** (nL = 3/4 sit above the carriable floor; not used
for the verdict).

## What is even measurable (the geometry)

The event is a **3D cobordism**: a 2D `SymmetricWindowSurface` (S² minus the four A₄
windows A,B,C,R) stacked over temporal slices by the staircase prism. At nL = 2 there
are 3 slices (0,1,2), 42 vertices/slice, 408 tetrahedra, 684 edges.

In 3D the Regge **hinges are edges** (the (d−2)-simplices); curvature is their deficit
angle. An edge carries an honest curvature **only if its coface fan closes** — every
triangle around it shared by two tetrahedra. The edges on the frozen top/bottom slices
and the edges bordering the window holes have **open fans**; their "deficit" is a
boundary artefact near 2π, not curvature. The clean **interior set is 264 of the 684
edges** (= the 84 closed L1-spatial edges + 90+90 closed timelike edges); the 280
boundary triangles vs 676 interior triangles confirm the cut.

**Skeleton handling.** `dualVolume()`/`lorentzianDeficitAngle()` walk a hinge up through
its cofaces, so the facet/coface lattice must exist *and be built in C++*: the
`ReggeSolver(st, MatterConfiguration())` ctor does this. Driving `materializeFacets()`
from Python registers detached **copies** of the sub-simplices (copy-semantics bindings),
corrupts the coface lists, and `dualVolume()` then sees half the cofaces. The prior pass
used the Python `materializeFacets()` path; this one does not.

**Determinism.** The symmetric A₄ windows + uniform seed metric make the relaxation fully
deterministic — every number below is **identical across seeds 0, 1, 2** (not merely
within a statistical band).

## Results (nL = 2, residual 68.3 at the carriable floor; A's ~71)

Top-slice **proton sector confirmed**: singlet overlap `1.0000`, σ `8e-17`.

| quantity | value | reference |
|---|---|---|
| interior (closed-fan) hinges | **264** of 684 | — |
| **r·m** (task's literal: `V_dual^{1/3}` × #352 shell Re-deficit) | **1.48** | physical 4.0; prior 8.8 |
| m_shell (#352 shell Re-deficit, interior) | 0.294 | — |
| &nbsp;&nbsp;extensive alternatives | sum Re ε = 57.3; sum \|★h\|Re ε = 24.1 | — |
| **dual radius** r = `V_dual^{1/3}` (V_dual = 127.0) | **5.03** | prior frozen r = 1.29 |
| &nbsp;&nbsp;cross-check r = `V3^{1/3}` (primal tets, V3 = 177.2) | 5.62 | — |
| **mean Re(deficit)** | **+0.217** (net positive) | round sphere: any sign, uniform |
| **participation ratio** of \|Re ε·★h\| | **0.414** | round sphere: 1.0 |
| &nbsp;&nbsp;concentration vs equal-volume sphere | **2.4× more concentrated** | — |
| curvature std/mean | 2.95 | round sphere: 0 |
| window isotropy (min/max quark share) | 0.60 | 1.0 = color-symmetric |
| curvature-weighted RMS shell radius | 0.95 shells | — |
| fraction of \|curvature\| within shell ≤ 1 of quarks | 0.98 | — |
| \|Q_e\| (Gauss-law holonomy, **unnormalized**) | 0.117 | — |
| m/Q (**mixed units**) | 2.52 | only Q_p/Q_p̄ = −1 is clean |

## Honest verdict — how proton-like is it?

**Clean and robust (deterministic across seeds):**

- **The radius is now genuinely emergent.** `r_dual ≈ 5.0` comes from the relaxed
  interior's signature-aware circumcentric dual volume, with primal/dual agreement
  (V3^{1/3} = 5.6 vs V_dual^{1/3} = 5.0). The prior `r ≈ 1.29` was an artefact of
  averaging mostly-frozen `l² = 1` edges — it measured the seed, not the proton. This is
  the substantive fix.
- **The curvature is a localized, net-positive lump.** Mean Re(deficit) = **+0.217 > 0**
  (positive curvature, the bound-state / sphere-like sign); participation ratio **0.414**
  → **2.4× more concentrated** than a round sphere of equal dual volume; **98%** of the
  curvature sits within one BFS shell of the quark windows. Qualitatively this is a bound
  lump, not a spread-out unbound configuration.
- **It is the proton sector**, not a colored object: singlet 1.0, σ ~ 0.

**Soft / honest caveats:**

- **r·m is not a sharp validator at nL = 2.** Across reasonable definitions it spans
  ~0.3–320: the *intensive* #352 shell-mean mass (0.29) × the dual radius (5.0) gives
  **r·m ≈ 1.48**, while *extensive* masses (sum Re ε = 57, or dual-weighted 24) push it
  to 120–320. The task's literal combination (1.48) is the same **O(1–10)** as both the
  physical 4.0 and the prior 8.8 — and removing the boundary pollution moves the literal
  ratio from 8.8 toward (in fact just below) 4.0 — but the definitional spread means the
  agreement with 4.0 should be read as *order-of-magnitude*, not a hit.
- **The lump is not a *smooth* round sphere.** It has the right *sign* (positive) and is
  localized, but its curvature std/mean ≈ 3 and window isotropy ≈ 0.60 — it is lumpy and
  only roughly color-balanced, not the uniform curvature of a round S³.
- **The worldtube is shallow** (only ~2 BFS shells deep at nL = 2), so the radial
  localization profile is coarse; the participation ratio (over the 264 hinges) is the
  more reliable concentration measure than the shell profile.
- **m/Q is mixed-units.** Q is in unnormalized register-holonomy units (|Q_e| ≈ 0.12 at
  the bottom reference; it is surface-dependent). Only the CPT ratio Q_p/Q_p̄ = −1 is
  normalization-free; pinning Q to the elementary +1 is not done here.

**Bottom line.** Measured the right way — on the relaxed emergent interior, with the
boundary artefacts excluded and the skeleton built in C++ — the Experiment-B top-slice
object is a **confined color singlet** sitting on a **localized, net-positive-curvature
bound lump** with a **genuine emergent dual radius (~5)**. That is structurally
**proton-like**. The dimensionless `r·m ≈ 1.5` is the same order as the physical 4.0 and
below the boundary-polluted prior 8.8, but `r·m` is too definition-sensitive at nL = 2 to
serve as a sharp quantitative validator; the robust evidence is the localization and the
positive curvature, not the precise ratio.
