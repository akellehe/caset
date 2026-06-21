# SPIKE — A 3+1 D cobordism over a triangulated S³ (the home for full E, B and 4×4 Dirac)

> **This is a SPIKE.** The deliverable is this design note plus an *optional*,
> isolated `S³` smoke prototype (`tests/cobordism/test_s3_smoke.py`). It adds **no**
> production C++ and changes **no** 2+1 D proton path. It scopes — it does not
> build — the 3+1 D sector. Tracks epic #410; ticket #418.

## TL;DR — recommendation: **BUILD THE FULL DIMENSION (generalize; do not reduce)**

Build the `S³` sector at full dimension. **Do not** reduce to a 2+1 D / reduced
sector as a shortcut: the program is fully emergent, so dropping a dimension
fundamentally changes the physics — it is not a faithful "first cut." The
dimensional question is answered by the **generalization**, not by a reduced
slice: ticket **#429** ("Iterated apex-reflection cobordism (dimension-generic)
and emergent twists") generalizes `symmetricStackCells` via coface-mirroring,
builds `S³` over a tetrahedral base, and measures an emergent twist.

- **Almost all the machinery is already dimension-generic** and runs on a
  triangulated `S³` / `S³ × I` as-is: `Spacetime::fromCells`, `getBoundary`,
  `getTopVertexCount`, `CombinatorialDimension`, the `HodgeLaplacian` spectrum /
  kernel, the dual Regge action, `CobordismRelaxer`, and `TemporalOrientation`.
  The isolated smoke prototype confirms this end-to-end: it stands up the closed
  `S³` (`∂Δ⁴`) and the 4D `S³ × I` and reads the `S³` Betti vector `[1,0,0,1]`
  straight off the Hodge spectrum.
- **Exactly one construction has a current implementation gap:**
  `Spacetime::symmetricStackCells` is triangle-only *today* (the `t.size() != 3
  -> continue` guard; `src/spacetime/Spacetime.cpp:437`). This is a temporary
  gap, **not** a dimensional wall. The interior connectivity is defined
  dimension-generically from the base layer's coface/dual structure (cone each
  top `d`-simplex to an apex vertex; a facet shared by two cofaces has those
  cofaces' apex-connectivity reflected across it — see §0). #429 lifts the
  triangle special-case to this coface-mirroring rule: cone tetrahedra to a
  cell-apex, split the codim-1 gap cells over shared *triangles* on the canonical
  dual edge. Two supporting changes ride along: the `dualComplexValid` vertex-link
  check is hardcoded to `n=3` (`src/cobordism/ChainComplex.cpp:845`), and the
  register/window read-out assembles connections on *triangles*
  (`src/cobordism/EigenstateSynthesis.cpp:1016,1251`).
- **The cost is resolution-driven, and accepted.** A *minimal* `S³ × I` is
  actually *cheaper* than the current `S² × I`; a *like-resolution* `S³ × I`
  (icosahedral-symmetry class → the 600-cell) costs ≈ **10³–10⁴× the current
  `S² × I`** per iteration in dense Hodge linear algebra alone. This is a real
  number to budget for — not a reason to reduce dimension. The cliff is the
  *resolution* needed to host 3D windows / plaquettes, not the extra dimension
  per se.
- **The full 3-component E and B vectors and the 4×4 Dirac–Kähler structure
  require `S³`** — and so does every other observable, faithfully, because the
  physics is emergent at the full dimension. There is no faithful reduced 2+1 D
  surrogate; the reduced sectors are diagnostics, not the target.

---

## 0. The dimension-generic apex rule and the emergent-twist hypothesis (#429)

The symmetric apex interior is **not** fundamentally 2D. Its connectivity is
defined from the base layer's **coface / dual** structure, dimension-generically:
cone each top `d`-simplex to an apex vertex; a facet (a `(d−1)`-simplex) shared
by **two** cofaces has those cofaces' apex-connectivity **reflected** across it,
and lower faces shared by `k` cofaces mirror analogously. In 2D this is the
dual-edge octahedron split; in 3D it cones tetrahedra to cell-apexes across
shared *triangles*. The current `symmetricStackCells` triangle-only special case
(`t.size() != 3 -> continue`) is a temporary **implementation gap**, not a wall —
#429 ("Iterated apex-reflection cobordism, dimension-generic") replaces it with
the coface-mirroring rule above, gains an `nApexSlices` parameter (default `1` =
today's single layer), and builds `S³` over a tetrahedral base.

**The emergent-twist hypothesis.** Each apex is a centre of point-reflection: its
neighbour edges are inverted through it to make the next slice, plus a "cap."
Iterating reflect-and-cap over many slices, and since composing reflections is a
**rotation**, the stack may accumulate a net screw / monodromy along time — a
**twist**. This twist is *emergent* (read off the relaxed geometry), unlike the
hand-supplied `prismCells` twist. A nontrivial twist makes `W` a **mapping torus**
(a twisted product). Candidate physical meanings to test in #429: a `Z₃` twist =
the colour singlet's `ω`-phases; a `2π → −1` twist = the spinor sign (spin); a
`U(1)` twist = a gauge holonomy (charge). #429 measures this emergent twist
directly; it is the proper way to answer the dimensional question — by
generalizing, not by reducing to a smaller sector.

---

## 1. Machinery generalization audit

Per component: **generalizes as-is** / **needs change X**, with a `file:line`
citation. "Generic" = keys off the combinatorial dimension / vertex sets with no
2-base assumption; "assumes-2D" = hardcodes triangles, a surface, or `k=2`.

| Component | Verdict | Evidence (`file:line`) |
|---|---|---|
| `Spacetime::fromCells` | **Generalizes as-is** | Builds one top simplex per cell of *any* arity via `createSimplex`; the metric-signature dimension is a parameter. `src/spacetime/Spacetime.cpp:300-353`. `fromCells(3, …)` ⇒ tetrahedral `S³`; `fromCells(4, …)` ⇒ 4-simplex `S³ × I`. The optional `vertexTimes` arg already auto-wires timelike legs for a Lorentzian extrusion (`:305,:321-329`). |
| `Spacetime::getTopVertexCount` | **Generalizes as-is** (one precondition) | Returns `signature.dimensions + 1` — `src/spacetime/Spacetime.cpp:569-574`. Precondition: the signature dimension must match the triangulation's dimension, else the top set is empty (`:587`). Build `S³` with a `Signature(3,…)` and `S³×I` with `Signature(4,…)` — which `fromCells(3,…)` / `fromCells(4,…)` do. |
| `Spacetime::getBoundary` | **Generalizes as-is** | Pure facet-counting from the top simplices' vertex sets (a codim-1 face is boundary iff owned by exactly one top cell). `src/spacetime/Spacetime.cpp:576-617`. Dimension-agnostic; the smoke prototype confirms `∂(S³)=∅` and `∂(S³×I)=S³⊔S³`. |
| `CombinatorialDimension::compute` | **Generalizes as-is** | `max(|vertices|) − 1` over all simplices — `src/cobordism/CombinatorialDimension.cpp:14-20`. Returns `3` for the `S³` slice and `4` for `S³×I`; the latter is the first time the `n=4` top-cell path is exercised here. |
| `HodgeLaplacian` spectrum / kernel | **Generalizes as-is** | `spectrum/eigenvalues/eigenvectors/harmonics/harmonicMatrix(k, metric)` are valid for every `k` up to the top dimension, "empty above the top dimension" — `include/cobordism/HodgeLaplacian.h:149-191`. `k=2` (E/B) and the higher-`k` ladder are reachable in principle; `k=4` activates on `S³×I`. The harmonic count `#{|λ|<tol} = b_k` is read off generically. |
| Dual Regge action | **Generalizes as-is** | Sums `dualVolume()·lorentzianDeficitAngle()` over hinges = `(n−2)`-simplices, no dimension-specific branch — `src/simulations/ReggeSolver.cpp:129-137`. In 4D the hinges are triangles; the Lorentzian/null handling is hinge-local. (`Characteristic.cpp:27` gates the Pontryagin number to `n=4`, but that is an Euler/signature characteristic-number path, not the action.) |
| `CobordismRelaxer::relaxInterior` | **Generalizes as-is (cost driver)** | Bounded GN/LM descent of `β‖∇_I S‖² + r_state` over interior edge squared-lengths via the analytic gradient + sparse Hessian of the dual Regge action — `include/cobordism/CobordismRelaxer.h:54-61`. No 2-base assumption; cost scales with interior-edge count × per-iteration Hodge/Regge solve (see §2). |
| `TemporalOrientation::orientationOf` | **Generalizes as-is** | Counts vertices on the min/max time slices (the Ambjørn–Loll initial/final split) — works for any vertex count, `src/mesh/TemporalOrientation.cpp:99-118`. Causal type via `Edge::isTimelike()/isSpacelike()` — `include/mesh/Edge.h:118-119`. |
| **`Spacetime::symmetricStackCells`** | **CURRENT GAP — the core build (#429)** | Triangle-only *today* via the temporary `if (t.size() != 3) continue;` guard (`src/spacetime/Spacetime.cpp:437`) — an implementation gap, **not** a dimensional wall. It cones *triangles* to face-apexes and splits the octahedron over a shared *edge* along the dual edge `f1–f2` (`:451-466`). The connectivity is defined dimension-generically from the coface/dual structure (§0): #429 lifts the special-case to the coface-mirroring rule — cone each top `d`-simplex to a cell-apex (the `(4,1)`/`(1,4)` cells in 3D), and split the codim-1 gap cells over each shared `(d−1)`-face (a *triangle* in 3D) along the canonical dual edge (the two cell-centres). The canonical-dual split is what keeps it `g`-equivariant; the dual edge `f1–f2` of two cells sharing a codim-1 face is canonical in any dimension, so the split survives the extra codimension — to be built and its equivariance re-measured. (Not Python-bound today, so even the smoke prototype cannot call it — it uses the generic `prismCells` for the stack.) |
| `dualComplexValid` (manifold validity) | **NEEDS CHANGE for `n=4`** | The ridge-link "pinch" check is generic over `(n−2)`-faces (`src/cobordism/ChainComplex.cpp:826-843`), but the vertex-link sphere test is hardcoded to `n=3`: `if (dim == 2) return ok;` then "n == 3: vertex links must be 2-spheres" with a `χ=2` test on *triangular* link faces — `src/cobordism/ChainComplex.cpp:845-889`. At `n=4` a vertex link is a 3-complex (tetrahedral faces) whose validity is "is it an `S³`?", not a surface `χ`; the existing test misapplies. **Caveat:** this is a *validation* helper — it does not block building or running the Hodge solve (the smoke prototype never calls it), so it is a correctness add for the full-dimension build (#429), not a blocker. |
| Register / window read-out (`EigenstateSynthesis`) | **Assumes-2D (for the *period* read-out only)** | The kernel-only harmonic read is generic (`harmonicMatrix(k,…)`, `src/cobordism/EigenstateSynthesis.cpp:906,1043`), but the *connection/window* assembly hardcodes triangles: `curvatureFromConnection` requires degree-2 cells of 3 vertices (`:1016`) and the loop read-out requires triangular holes (`:1251`). The color-singlet/confinement read-out is therefore triangle-hardcoded *today*. **Path:** generalize the connection/window assembly to 3-cell windows via the coface-mirroring read-out (§0, #429) so colour reads at the full emergent dimension. (The E/B and Dirac–Kähler sectors do *not* use the window/period read-out — they read the field-strength cochain and the operator — so they need only the mesh, independent of the window port.) |
| `EigenstateSynthesis::fieldStrengthSplit` (E/B, #417) | **Generic code, 2D-limited physics** | The split runs at `k=2` in any dimension — it reads each plaquette's causal type from its edges — but it *requires* a 2-cochain: `if (k_ != 2) throw` (`src/cobordism/EigenstateSynthesis.cpp:1038`). `F` is a 2-form in every dimension, so the *code* is generic; the *physics* deficit (a 2-component E and a single-B pseudoscalar on an `S²` slice vs. the full 3-vectors) is the slice, not the code. See §3. |
| `DiracKähler` operator (`(d+δ)`, #415/#424) | **NEEDS `S³×I` to realize; assembles generically** | The operator `D=d+δ` and the `D²=L` check iterate `0..n` and run on any mesh, but the Clifford framework is fixed at 4D: `constexpr int kFrameworkDim = 4` (`src/cobordism/DiracKahler.cpp:29`), `gammaDimension()=2⁴=16` (`:181`), `multiplicity()=2²=4` (`:184`). The full `16 = 4×4` module needs `Ω⁰…Ω⁴`, i.e. a genuine 4-complex. The current `S²×I` cobordism is only a **3-complex** (`Ω⁰…Ω³`), so the 4×4 spinor image and the 4-fold taste do not materialize there. The header says exactly this (`include/cobordism/DiracKahler.h:97-102`). No code change is needed in `DiracKähler` itself — it needs the `S³×I` *mesh* to feed it `Ω⁴`. |
| `MergeCobordism::operatorU()` / `choiState()` | **Deferred — `S³` *may* help, not assessed to resolve here** | Empty pending the interior-handle operator-topology rework: on the current topology `ker L₁(W−∂W)` is `d²−1`-dim with no basis-independent map to the `d×d` operator (needs "distinguished interior Choi-cycles") — `include/cobordism/MergeCobordism.h:58-65,150-159`. An `S³` interior has genuine 3-handles (`S²×S¹`-type cycles) that *could* supply distinguished `L₁` cycles, which a 2D interior cannot. **Verdict for this spike:** an `S³` interior is the *natural* place to resolve the frame ambiguity, but confirming it does is its own ticket — out of scope here (the ticket only asks to assess `S³` impact, and the impact is "plausibly enabling, unverified"). |

**One-line summary.** Generic and ready: `fromCells`, `getBoundary`,
`getTopVertexCount`, `CombinatorialDimension`, `HodgeLaplacian`, dual Regge action,
`CobordismRelaxer`, `TemporalOrientation`. Needs a build (#429):
`symmetricStackCells` (the dimension-generic coface-mirroring apex stacking, §0),
the `dualComplexValid` `n=4` link check, and the `S³` window read-out generalized
via coface-mirroring. Needs only the mesh: `DiracKähler`.

---

## 2. Cost estimate

**Operator size = Hodge `L_k` matrix dimension = `|C_k|`** (number of `k`-simplices).
The dense symmetric eigensolve behind `HodgeLaplacian::spectrum` is `O(|C_k|³)`;
the relaxation's per-iteration cost is dominated by the largest `L_k` solve plus
the sparse Regge gradient/Hessian over interior edges.

### Closed spatial slices (`|C_0|, |C_1|, |C_2|, |C_3|`)

| Triangulation | role | V | E | tri | tet | Betti |
|---|---|---:|---:|---:|---:|---|
| Icosahedron `S²` | current proton base | 12 | 30 | 20 | — | `[1,0,1]` |
| **`∂Δ⁴`** (5-cell) | minimal `S³` (smoke) | 5 | 10 | 10 | 5 | `[1,0,0,1]` |
| 16-cell (4-orthoplex ∂) | small symmetric `S³` | 8 | 24 | 32 | 16 | `[1,0,0,1]` |
| **600-cell** | like-resolution `S³` (icosahedral class) | 120 | 720 | 1200 | 600 | `[1,0,0,1]` |

The 600-cell is the `S³` analogue of the icosahedron: its symmetry group `H₄`
contains the binary icosahedral group, so it is the natural fine geodesic 3-sphere
for a symmetric-window (A₄/H₄) construction — the 3D counterpart of the icosahedral
`S²` the proton uses.

### Extruded cobordisms (`× I`, one layer)

| Cobordism | dim | top cells | `|C_2|` (the eigensolve driver) | rel. to `S²×I` |
|---|---:|---:|---:|---|
| `S²×I` (current `W_ABC`) | 3 | — | `~10²` (ticket; `b₁=11`) | 1× (baseline) |
| `∂Δ⁴ × I` | 4 | 20 | 60 | **< 1×** (too coarse to use) |
| 16-cell `× I` | 4 | 64 | ~200 | ~1× (still coarse for windows) |
| 600-cell `× I` | 4 | 2400 | thousands | **~10³–10⁴×** |

(`∂Δ⁴×I` is *measured* by the smoke prototype: f-vector `[10, 35, 60, 55, 20]`
— V=10, E=35, tri=60, tet=55, 20 four-simplices — Betti `[1,0,0,1,0]`, χ=0, with
`∂(S³×I)` = 10 boundary tetrahedra in 2 components. The `16-cell×I` top-cell count
is the `(m=4)` Freudenthal four-simplices per base tetrahedron × 16 base tets =
`64`.)

### The headline number

Comparing **like resolution** (the icosahedral-symmetry class the symmetric-window
construction needs): `S²` `|C_2| = 20` → 600-cell `|C_2| = 1200`. The dense Hodge
eigensolve alone scales `(1200/20)³ = 60³ ≈ 2×10⁵`; on the edge operator
`(720/30)³ ≈ 1.4×10⁴`. With the interior-edge count (LM unknowns) up roughly an
order of magnitude on top, a like-resolution `S³×I` relaxation is **≈ 10³–10⁴×
the current `S²×I` per iteration**. The cliff is the *resolution* required to host
3D windows / plaquettes, not the extra dimension per se — a minimal `S³×I` is
actually cheaper than `S²×I` but carries nothing.

---

## 3. Observable requirement table

| Observable | Verdict | One-line reason |
|---|---|---|
| **Full E-vector** (3 components) | **Requires `S³`** | `E` = plaquettes with a timelike leg; 3 independent spacelike directions need a 3D spatial slice. An `S²` slice has 2 → only a 2-component reduced E. The split *code* runs generically (`EigenstateSynthesis.cpp:1038`); the deficit is the slice. |
| **Full B-vector** (3 components) | **Requires `S³`** | `B` = purely spacelike plaquettes; an `S²` slice has a single spacelike 2-plane orientation → `B` is one pseudoscalar, not a 3-vector. 3 independent spacelike 2-planes need `S³`. |
| **4×4 Dirac–Kähler** (16-component `Ω⁰…Ω⁴`) | **Requires `S³×I`** | The full `16 = 4×4` Clifford module needs `Ω⁴ ≠ 0`, i.e. a 4-complex. The `S²×I` cobordism is a 3-complex (`Ω⁰…Ω³`); `DiracKähler` reports the 4D framework (`DiracKahler.cpp:29,181,184`) but the 4×4 image / 4-fold taste only materialize on `S³×I`. |
| **Gauss-law charge density `j⁰`** (`∇·E = ρ`) | **Full emergent dimension** | `∇·E = ρ` is read off the full E field on the `S³` slice; the divergence of a *reduced* E on an `S²` slice is a different (lower-dimensional) physics, not a faithful surrogate. Read the holonomy / triality (`Q = I₃ + Y/2`) link at full dimension — see the emergent twist of #429. |
| **Binary flavor** (proton-vs-neutron) label | **Full emergent dimension** | A `Z₂` label is emergent from the full geometry (the genuine 4-fold DK taste, the temporal/spatial split). A reduced `S²×I` multiplicity changes the physics rather than approximating it; read the label off the full-dimension relaxation. |
| Color singlet / confinement `σ` | **Full emergent dimension** | Period/window read-out (kernel-only, `EigenstateSynthesis.cpp:906`); on `S³` the window assembly is generalized via the coface-mirroring read-out (§0, #429), not kept on a 2D sub-slice as a shortcut. |

**The contract this table encodes:** every observable targets the **full
emergent dimension** — the physics is emergent, so a reduced slice is a different
theory, not a faithful first cut. The **3-vector E/B** and the **4×4
Dirac–Kähler** make this most visible (they cannot even be expressed below `S³`),
but the Gauss-law charge (#411) and the flavor label (#414) are equally
full-dimension observables. The dimensional question is settled by the
generalization (#429), not by standing up a reduced slice.

---

## 4. Go / no-go + minimal prototype path + sequencing

### Decision: **BUILD THE FULL DIMENSION — generalize via #429, do not reduce**

The machinery is almost entirely ready, and the physics is emergent — so a
reduced 2+1 D sector is a *different* theory, not a faithful first cut. Answer the
dimensional question by **generalizing**: #429 ("Iterated apex-reflection
cobordism, dimension-generic") lifts `symmetricStackCells` to the coface-mirroring
rule (§0), builds `S³` over a tetrahedral base, and measures the emergent twist.
The 10³–10⁴× like-resolution cost is accepted and budgeted, not a reason to drop a
dimension.

### Where it sequences in epic #410

```
#412 ─▶ #413 ─┬─▶ #417 E/B (full E, B) ─▶ #411 charge (full j⁰) ─┐
              └─▶ #415 Dirac–Kähler (full Ω⁰…Ω⁴) ────────────────┼─▶ #414 flavor (full)
                                                                  │
#418 (this spike) ─ recommends: build at full dimension ─────────┤
                    via the dimension-generic generalization:     │
                                                                  ▼
   #429 apex-reflection cobordism (coface-mirroring + emergent twist)
        ─▶ S³ sector ─▶ full 3-vector E/B + 4×4 Dirac–Kähler
```

The spike's standing recommendation for #417 / #415 / #411 / #414: **build at the
full emergent dimension**; the apex generalization and the emergent-twist
measurement are owned by #429.

### Minimal prototype path for the full-dimension `S³` build (#429)

In order; all C++ on existing classes / static utils, **no free functions**:

1. **`S³` base builder** — a static on `Spacetime` returning a tetrahedral cell
   list, mirroring `fromCells`/`prismCells`: e.g. `Spacetime::crossPolytopeCells(4)`
   (the 16-cell) and a geodesic 3-sphere subdivision for the 600-cell-class
   resolution. (The smoke prototype needs none — `∂Δ⁴` is 5 combinatorial
   tetrahedra fed straight to the existing `fromCells`.)
2. **The core build — dimension-generic apex stacking (#429).** Generalize
   `Spacetime::symmetricStackCells` (`src/spacetime/Spacetime.cpp:414-468`) via the
   coface-mirroring rule (§0): cone each top `d`-simplex up/down to a cell-apex,
   split the codim-1 gap cells over each shared `(d−1)`-face (a *triangle* in 3D)
   along the canonical dual edge (the two cell-centres `f1–f2`); add the
   `nApexSlices` parameter (default `1`). Re-measure the `g`-equivariance residual
   (`‖M P_in − P_out M‖/‖M‖`) — the #413 result was `4.5e-14` in 2D; confirm the
   dual split holds in the extra codimension. Iterate reflect-and-cap over many
   slices and read off the emergent twist (§0). Bind it to Python.
3. **`dualComplexValid` `n=4` link check** — replace the `n=3` surface `χ=2`
   vertex-link test (`src/cobordism/ChainComplex.cpp:845-889`) with an `S³`
   (3-sphere) link test, so the 4-complex passes the manifold gate.
4. **`S³` window read-out** — generalize the triangle-hardcoded connection/window
   assembly (`EigenstateSynthesis.cpp:1016,1251`) to 3-cell windows via the
   coface-mirroring read-out (§0), so colour is read at the full emergent
   dimension rather than on a reduced 2D sub-slice. (The E/B + Dirac–Kähler
   sectors bypass the window read-out and need only the mesh.)
5. **First `S³×I` relaxation smoke** — one `CobordismRelaxer::relaxInterior` pass
   on a 16-cell `× I`, asserting convergence and a finite `dualReggeAction`; then
   the full 3-vector E/B (`fieldStrengthSplit` already runs at `k=2`) and the
   `Ω⁰…Ω⁴` Dirac–Kähler (`DiracKähler` already reports the 4D framework).

### This spike's optional prototype (delivered here)

`tests/cobordism/test_s3_smoke.py` — isolated, deterministic, **no production C++**.
It builds the closed `S³` (`∂Δ⁴`) via `fromCells` and the 4D `S³×I` via the
dimension-generic `prismCells`, and asserts (F1) `CombinatorialDimension == 3`
(slice) / `4` (`S³×I`); (F2) `HodgeLaplacian.spectrum(k)` runs `k=0..3` and
`#{|λ|<1e-9} = [1,0,0,1]` (volume *and* unit weights), cross-checked against the
combinatorial `ChainComplex.bettiNumbers()`; (F3) the `n=4` top path runs without
raising; (F4) `getBoundary` is empty for closed `S³` and `S³⊔S³` (10 tets, 2
components) for `S³×I`; (F5) two in-process builds give bit-for-bit identical
spectra. It shares no state with the proton pipeline, so it cannot perturb the
golden 2+1 D results (`tests/cobordism/test_epic410_invariants.py` is the guard).

---

## Paste-ready recommendation for the #410 checklist

> **#418 (S³ spike) — BUILD THE FULL DIMENSION; do not reduce.** Almost all the
> cobordism machinery (`fromCells`, `getBoundary`, `getTopVertexCount`,
> `CombinatorialDimension`, `HodgeLaplacian`, dual Regge action, `CobordismRelaxer`,
> `TemporalOrientation`) is already dimension-generic and runs on a triangulated
> `S³`/`S³×I` as-is — confirmed by an isolated smoke prototype that reads `S³`
> Betti `[1,0,0,1]` off the Hodge spectrum. The single core build is the
> dimension-generic generalization of `symmetricStackCells` via coface-mirroring
> (triangle-only *today*, `Spacetime.cpp:437` — a gap, not a wall), with a
> `dualComplexValid` `n=4` link check and an `S³` window read-out riding along;
> `DiracKähler` needs only the 4-complex mesh, no code change. The program is
> **emergent**, so reducing to a 2+1 D sector changes the physics and is *not* a
> faithful first cut — the dimensional question is answered by **#429** ("Iterated
> apex-reflection cobordism, dimension-generic and emergent twists"), which
> generalizes the apex rule, builds `S³` over a tetrahedral base, and measures an
> emergent twist (candidate carriers: `Z₃` colour `ω`-phases, a `2π→−1` spinor
> sign, a `U(1)` charge holonomy). A like-resolution `S³×I` (600-cell class) costs
> ≈ 10³–10⁴× the current `S²×I` per iteration — accepted and budgeted, not a
> reason to reduce. Every observable — the 3-vector E/B and 4×4 Dirac–Kähler, the
> Gauss-law charge (#411), the flavor label (#414) — targets the full emergent
> dimension.
