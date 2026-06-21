# SPIKE — A 3+1 D cobordism over a triangulated S³ (the home for full E, B and 4×4 Dirac)

> **This is a SPIKE.** The deliverable is this design note plus an *optional*,
> isolated `S³` smoke prototype (`tests/cobordism/test_s3_smoke.py`). It adds **no**
> production C++ and changes **no** 2+1 D proton path. It scopes — it does not
> build — the 3+1 D sector. Tracks epic #410; ticket #418.

## TL;DR — recommendation: **STAGE**

Build the `S³` sector **later**, gated on a reduced-sector observable demonstrably
hitting the 2+1 D ceiling — not now, and not never.

- **Almost all the machinery is already dimension-generic** and runs on a
  triangulated `S³` / `S³ × I` as-is: `Spacetime::fromCells`, `getBoundary`,
  `getTopVertexCount`, `CombinatorialDimension`, the `HodgeLaplacian` spectrum /
  kernel, the dual Regge action, `CobordismRelaxer`, and `TemporalOrientation`.
  The isolated smoke prototype confirms this end-to-end: it stands up the closed
  `S³` (`∂Δ⁴`) and the 4D `S³ × I` and reads the `S³` Betti vector `[1,0,0,1]`
  straight off the Hodge spectrum.
- **Exactly one construction does *not* generalize:** `Spacetime::symmetricStackCells`
  is 2D-only (it cones triangles; `src/spacetime/Spacetime.cpp:437`). The 3D
  analogue — cone tetrahedra to a cell-apex, split the codim-1 gap cells over
  shared *triangles* on the canonical dual edge — is the single core build the
  staged ticket owns. Two supporting changes ride along: the
  `dualComplexValid` vertex-link check is hardcoded to `n=3`
  (`src/cobordism/ChainComplex.cpp:845`), and the register/window read-out
  assembles connections on *triangles* (`src/cobordism/EigenstateSynthesis.cpp:1016,1251`).
- **The cost cliff is resolution, not dimension.** A *minimal* `S³ × I` is
  actually *cheaper* than the current `S² × I`, but is far too coarse to host the
  color windows or a meaningful E/B plaquette set. A *like-resolution* `S³ × I`
  (icosahedral-symmetry class → the 600-cell) costs ≈ **10³–10⁴× the current
  `S² × I`** per iteration in dense Hodge linear algebra alone — squarely a
  "stage it" number.
- **The full 3-component E and B vectors and the 4×4 Dirac–Kähler structure
  genuinely require `S³`.** Their *reduced* sectors (a 2-component E, a single-B
  pseudoscalar, the Gauss-law charge density, the binary flavor label) are
  faithful to explore first in 2+1 D — so the track should not over-build.

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
| **`Spacetime::symmetricStackCells`** | **NEEDS CHANGE — the core build** | 2D-only: `if (t.size() != 3) continue;` skips any base cell that is not a triangle — `src/spacetime/Spacetime.cpp:437`. It cones *triangles* to face-apexes and splits the octahedron over a shared *edge* along the dual edge `f1–f2` (`:451-466`). Fed a tetrahedral `S³` base it returns an **empty** interior. The `S³` analogue: cone *tetrahedra* to a cell-apex (the `(4,1)`/`(1,4)` cells), and split the codim-1 gap cells over each shared *triangle* along the canonical dual edge (the two cell-centres). The canonical-dual split is what keeps it `g`-equivariant; the open question the spike answers structurally — *does the dual split survive the extra codimension?* — is **yes in principle** (the dual edge `f1–f2` of two cells sharing a codim-1 face is canonical in any dimension), but it must be built and its equivariance re-measured. (Not Python-bound today, so even the smoke prototype cannot call it — it uses the generic `prismCells` for the stack.) |
| `dualComplexValid` (manifold validity) | **NEEDS CHANGE for `n=4`** | The ridge-link "pinch" check is generic over `(n−2)`-faces (`src/cobordism/ChainComplex.cpp:826-843`), but the vertex-link sphere test is hardcoded to `n=3`: `if (dim == 2) return ok;` then "n == 3: vertex links must be 2-spheres" with a `χ=2` test on *triangular* link faces — `src/cobordism/ChainComplex.cpp:845-889`. At `n=4` a vertex link is a 3-complex (tetrahedral faces) whose validity is "is it an `S³`?", not a surface `χ`; the existing test misapplies. **Caveat:** this is a *validation* helper — it does not block building or running the Hodge solve (the smoke prototype never calls it), so it is a correctness add for the staged build, not a blocker. |
| Register / window read-out (`EigenstateSynthesis`) | **Assumes-2D (for the *period* read-out only)** | The kernel-only harmonic read is generic (`harmonicMatrix(k,…)`, `src/cobordism/EigenstateSynthesis.cpp:906,1043`), but the *connection/window* assembly hardcodes triangles: `curvatureFromConnection` requires degree-2 cells of 3 vertices (`:1016`) and the loop read-out requires triangular holes (`:1251`). The color-singlet/confinement read-out is therefore 2D-specific. **Mitigation:** the E/B and Dirac–Kähler sectors do *not* use the window/period read-out (they read the field-strength cochain and the operator), so an `S³` build need not port the color windows up first — it can keep color on a 2D sub-slice. |
| `EigenstateSynthesis::fieldStrengthSplit` (E/B, #417) | **Generic code, 2D-limited physics** | The split runs at `k=2` in any dimension — it reads each plaquette's causal type from its edges — but it *requires* a 2-cochain: `if (k_ != 2) throw` (`src/cobordism/EigenstateSynthesis.cpp:1038`). `F` is a 2-form in every dimension, so the *code* is generic; the *physics* deficit (a 2-component E and a single-B pseudoscalar on an `S²` slice vs. the full 3-vectors) is the slice, not the code. See §3. |
| `DiracKähler` operator (`(d+δ)`, #415/#424) | **NEEDS `S³×I` to realize; assembles generically** | The operator `D=d+δ` and the `D²=L` check iterate `0..n` and run on any mesh, but the Clifford framework is fixed at 4D: `constexpr int kFrameworkDim = 4` (`src/cobordism/DiracKahler.cpp:29`), `gammaDimension()=2⁴=16` (`:181`), `multiplicity()=2²=4` (`:184`). The full `16 = 4×4` module needs `Ω⁰…Ω⁴`, i.e. a genuine 4-complex. The current `S²×I` cobordism is only a **3-complex** (`Ω⁰…Ω³`), so the 4×4 spinor image and the 4-fold taste do not materialize there. The header says exactly this (`include/cobordism/DiracKahler.h:97-102`). No code change is needed in `DiracKähler` itself — it needs the `S³×I` *mesh* to feed it `Ω⁴`. |
| `MergeCobordism::operatorU()` / `choiState()` | **Deferred — `S³` *may* help, not assessed to resolve here** | Empty pending the interior-handle operator-topology rework: on the current topology `ker L₁(W−∂W)` is `d²−1`-dim with no basis-independent map to the `d×d` operator (needs "distinguished interior Choi-cycles") — `include/cobordism/MergeCobordism.h:58-65,150-159`. An `S³` interior has genuine 3-handles (`S²×S¹`-type cycles) that *could* supply distinguished `L₁` cycles, which a 2D interior cannot. **Verdict for this spike:** an `S³` interior is the *natural* place to resolve the frame ambiguity, but confirming it does is its own ticket — out of scope here (the ticket only asks to assess `S³` impact, and the impact is "plausibly enabling, unverified"). |

**One-line summary.** Generic and ready: `fromCells`, `getBoundary`,
`getTopVertexCount`, `CombinatorialDimension`, `HodgeLaplacian`, dual Regge action,
`CobordismRelaxer`, `TemporalOrientation`. Needs a build: `symmetricStackCells`
(the core 3D apex stacking), the `dualComplexValid` `n=4` link check, and an `S³`
window read-out (only if color is ported up). Needs only the mesh: `DiracKähler`.

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
| **Gauss-law charge density `j⁰`** (`∇·E = ρ`) | **Explorable first in 2+1 D** | The divergence of the *reduced* E on the `S²` slice is a genuine 0-cochain charge density — a faithful 2+1 D Gauss law. Only the E it diverges is reduced. Explore the holonomy / triality (`Q = I₃ + Y/2`) link in 2+1 D before paying for `S³`. |
| **Binary flavor** (proton-vs-neutron) label | **Explorable first in 2+1 D** | A `Z₂` label needs only 2-fold multiplicity. Candidate carriers — the temporal/spatial split, or a reduced DK multiplicity — are testable on `S²×I`. Commit to `S³` only if the binary label provably needs the genuine 4-fold taste (which *is* 4D). |
| Color singlet / confinement `σ` | **2+1 D (already in hand)** | Period/window read-out on the `S²` base (kernel-only, `EigenstateSynthesis.cpp:906`); `S³` would only refine resolution, not enable anything new. |

**The contract this table encodes:** only the **3-vector E/B** and the **4×4
Dirac–Kähler** force `S³`. Everything the track explores *next* — the Gauss-law
charge (#411), the flavor label (#414) — has a faithful 2+1 D first cut. Later
tickets must not stand up a 4D relaxation where a reduced slice suffices.

---

## 4. Go / no-go + minimal prototype path + sequencing

### Decision: **STAGE**

Neither "build `S³` now" (the cost is 10³–10⁴× and the next two charge/flavor
tickets have faithful 2+1 D first cuts) nor "defer indefinitely" (the full E/B
3-vectors and 4×4 Dirac–Kähler genuinely need it, and the machinery is almost
entirely ready). Stage it: explore the reduced sectors in 2+1 D, and trigger the
`S³` build when a reduced-sector observable demonstrably hits the 2D ceiling.

### Where it sequences in epic #410

```
#412 ─▶ #413 ─┬─▶ #417 E/B (reduced, 2+1 D) ─▶ #411 charge (2+1 D) ─┐
              └─▶ #415 Dirac–Kähler (reduced, 2+1 D) ────────────────┼─▶ #414 flavor (2+1 D first)
                                                                     │
#418 (this spike) ─ recommends: do the above in 2+1 D first ────────┘
                    then STAGE the S³ build below, gated on a 2D-ceiling hit:

   [staged] S³ sector ─▶ full 3-vector E/B + 4×4 Dirac–Kähler
```

The spike's standing recommendation for #417 / #415 / #411 / #414: **build the
reduced 2+1 D version first** (the table's right column). The `S³` build is a
*new, later* ticket, not a dependency of those four.

### Minimal prototype path for the staged `S³` build (when triggered)

In order; all C++ on existing classes / static utils, **no free functions**:

1. **`S³` base builder** — a static on `Spacetime` returning a tetrahedral cell
   list, mirroring `fromCells`/`prismCells`: e.g. `Spacetime::crossPolytopeCells(4)`
   (the 16-cell) and a geodesic 3-sphere subdivision for the 600-cell-class
   resolution. (The smoke prototype needs none — `∂Δ⁴` is 5 combinatorial
   tetrahedra fed straight to the existing `fromCells`.)
2. **The core build — tetrahedral symmetric apex stacking.** Generalize
   `Spacetime::symmetricStackCells` (`src/spacetime/Spacetime.cpp:414-468`) to a
   tetrahedral base: cone each base tetrahedron up/down to a cell-apex, split the
   codim-1 gap cells over each shared *triangle* along the canonical dual edge
   (the two cell-centres `f1–f2`). Re-measure the `g`-equivariance residual
   (`‖M P_in − P_out M‖/‖M‖`) — the #413 result was `4.5e-14` in 2D; confirm the
   dual split holds in the extra codimension. Bind it to Python.
3. **`dualComplexValid` `n=4` link check** — replace the `n=3` surface `χ=2`
   vertex-link test (`src/cobordism/ChainComplex.cpp:845-889`) with an `S³`
   (3-sphere) link test, so the 4-complex passes the manifold gate.
4. **`S³` window read-out (only if color is ported up)** — generalize the
   triangle-hardcoded connection/window assembly
   (`EigenstateSynthesis.cpp:1016,1251`) to 3-cell windows, *or* keep color on a
   2D sub-slice and use `S³` only for E/B + Dirac–Kähler (which bypass the window
   read-out).
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

> **#418 (S³ spike) — STAGE.** Almost all the cobordism machinery (`fromCells`,
> `getBoundary`, `getTopVertexCount`, `CombinatorialDimension`, `HodgeLaplacian`,
> dual Regge action, `CobordismRelaxer`, `TemporalOrientation`) is already
> dimension-generic and runs on a triangulated `S³`/`S³×I` as-is — confirmed by an
> isolated smoke prototype that reads `S³` Betti `[1,0,0,1]` off the Hodge
> spectrum. The single core build is the tetrahedral generalization of
> `symmetricStackCells` (2D-only today, `Spacetime.cpp:437`), with a
> `dualComplexValid` `n=4` link check and (only if color is ported up) an `S³`
> window read-out riding along; `DiracKähler` needs only the 4-complex mesh, no
> code change. A like-resolution `S³×I` (600-cell class) costs ≈ 10³–10⁴× the
> current `S²×I` per iteration. **Only the full 3-vector E/B and the 4×4
> Dirac–Kähler require `S³`**; the Gauss-law charge (#411) and the binary flavor
> label (#414) have faithful 2+1 D first cuts — do those reduced sectors first,
> then trigger the staged `S³` build when one provably hits the 2D ceiling.
