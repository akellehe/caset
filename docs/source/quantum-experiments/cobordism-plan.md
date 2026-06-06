# Cobordism correspondence — Stage 1 (algebraic layer) implementation plan

Implementation plan for **Stage 1** of the experiment specified in
[`cobordism.md`](cobordism.md) (the State–Operation–Cobordism correspondence).
Tracked under the **Cobordism Correspondence v0.1** milestone in the *Cobordism
Extension* project.

## Context

The spec tests whether a quantum operation between two states can be realized as a
**cobordism** whose boundary is the two states and whose TQFT value is their
transition amplitude. **Stage 1** is the pure-linear-algebra "algebraic layer"
(n=1): it validates the map–state correspondence and the gauge structure and, per
the spec, is the **correctness oracle** that runs first, before any mesh dynamics.
**This plan covers Stage 1 only**; Stage 2 (Dijkgraaf–Witten state-sum, interior
bistellar moves, Lorentzian variant) is a follow-up milestone.

Already built (the topological substrate, in `cobordism/`): `ChainComplex` (integer
boundary maps, ℚ/GF(2) Betti, torsion, intersection form, signature), `Cobordism`
(boundary verification), `Characteristic`/`CombinatorialDimension`, `IntegerLinalg`,
and exact `Topology` fixtures. The mesh (`Spacetime`/`Edge`/`Vertex`), the
`WilsonLoop` observable (curvature/causal holonomy on the dual graph), and a
partly-built operator/state layer in `quantum/` (Choi via
`ChoiState`/`InteractionSimulation`, Schmidt, von Neumann entropy) also exist.
Unbuilt: the spec's Hermitian-weighted Laplacian, harmonic states, and the
Choi–Jamiołkowski bending.

## Architecture — reuse the two existing worlds, build the seam

The experiment sits at the seam between the **mesh/topology** world (`Spacetime`,
`Edge`, `ChainComplex`, `WilsonLoop`) and the **operator/state** world (`quantum/`).
Work splits along that seam. All new code is class-organized and namespaced (no free
functions), and reuses existing machinery rather than re-creating it.

### Data model: the U(1) connection lives on `Edge`

- **`Edge` gains one member** (`Edge.h` already anticipates "storing state on the
  Edge"): `double phase` — the U(1) connection on the stored src→tgt orientation
  (negated on reversal). Magnitude is the existing signed `squaredLength` (its sign
  encodes signature, matching the spec and seeding the §5.6 Lorentzian variant). The
  default (`phase = 0`) leaves every existing CDT edge unchanged.
- **Complex weight** of an edge = `squaredLength · e^{i·phase}`. Assembly:
  `A_ij = Σ squaredLength · e^{i·phase}`, `D_ii = Σ |squaredLength|` (magnitude
  convention), `L = D − A`.
- **No `ComplexEdge`, no side tables, no `EdgeList` refactor** (`EdgeList` stores
  `Edge` by value → a subclass would slice; unnecessary once `phase` is a base
  member).
- **Deferred to Stage 2** (not exercised by Stage 1): `Edge::multiplicity`
  (Δ-complex identifications on surface fixtures); `Spacetime::fromAdjacency`; and a
  **U(1)-connection mode on `WilsonLoop`** (the general holonomy `Σ phase` around a
  loop — needed by T4, "bulk holonomies = Stage-1 fluxes"). Stage-1 cycles are tiny
  and known, so flux there is computed test-side, not via a library call.

### `SpacetimeType::HERMITIAN_WEIGHTED`

A new enum value. Unlike CDT (fixed edge lengths), this type carries variable
complex edge weights — a real semantic distinction. Nothing branches on
`SpacetimeType` today; this records intent and is the natural hook for future
weight-aware behavior. Stage-1 fixtures are constructed with this type.

### `cobordism::HodgeLaplacian` — operator on a `Spacetime`

An operator object in the spirit of `ReggeSolver` (wraps a `Spacetime`, does the
edge linear algebra), **degree-parameterized** so the same class serves Stage 1
(`k=0`, assembled from edge weights) and Stage 2 (`L_k` from `ChainComplex` boundary
matrices). Stage 1 implements `k=0`, reading `squaredLength`/`phase` off edges. It
does **not** compute flux, cycle bases, or Betti numbers — those are
`WilsonLoop`/`ChainComplex`'s job.

```cpp
namespace tessera::cobordism {
class HodgeLaplacian {
 public:
  explicit HodgeLaplacian(std::shared_ptr<Spacetime> st);

  std::vector<std::complex<double>> adjacency() const;     // N*N flat, Hermitian
  std::vector<double>               degree()    const;     // magnitude convention
  std::vector<std::complex<double>> laplacian(int k = 0) const;  // D - A at k=0

  bool   isHermitian(double tol = 1e-12) const;
  double unitarityResidual(double t = 1.0) const;          // ||e^{-iLt}(...)^H - I||

  std::vector<double>               eigenvalues(int k = 0) const;   // real, ascending
  std::vector<std::complex<double>> eigenvectors(int k = 0) const;  // columns
  std::vector<std::complex<double>> harmonics(int k = 0, double tol = 1e-9) const;

  HodgeLaplacian gauge(const std::vector<double>& alphas) const; // phase -> + a_i - a_j
 private:
  std::shared_ptr<Spacetime> st_;                          // lazy Eigen cache
};
}
```

Spectra via `Eigen::SelfAdjointEigenSolver<MatrixXcd>` (Hermitian ⇒ real
eigenvalues, `e^{-iLt}` unitary by construction). `gauge` rephases edge phases;
spectrum and every cycle flux must be invariant (C3). `b₁` and any flux are read
from `ChainComplex::bettiNumbers()` and (test-side) `Σ phase` around the known
cycle.

### `cobordism::SpectralGap` / `HarmonicDimension` — scalar Observables

The *scalar* spectral measurements are `Observable`s (`compute(spacetime) → double`),
mirroring how `EulerCharacteristic`/`Signature` wrap `ChainComplex`
(`Characteristic.cpp` delegates to `ChainComplex::fromSpacetime`). Because the
connection lives on `Edge::phase`, a `Spacetime` carries everything they need.

- **`SpectralGap`** — `λ₁ − λ₀` of the Hermitian Laplacian; collapses to 0 at
  `Φ = π` on the triangle (C4).
- **`HarmonicDimension`** — `dim ker L` (zero-mode count); the U(1) flux *lifts* the
  zero-mode, so it drops 1 → 0 as `Φ` leaves 0, while the topological
  `ChainComplex.bettiNumbers()` is unchanged (C4/C5).

`HodgeLaplacian` itself stays a rich operator (returns matrices/eigenvectors, like
`ChainComplex`), not an `Observable`.

### `quantum::ChoiJamiolkowski` — map–state duality (bending)

Bending *is* the Choi–Jamiołkowski construction, so it is a class in `quantum/`
(beside `ChoiState`), reusing Eigen + the existing `vonNeumannEntropy` helper.

```cpp
namespace tessera::quantum {
class ChoiJamiolkowski {
 public:
  ChoiJamiolkowski() = delete;  // static utility (cobordism::Cobordism pattern)
  static std::vector<std::complex<double>> vectorize(const std::vector<std::complex<double>>& U, int dA, int dB);
  static std::vector<double> singularValues(const std::vector<std::complex<double>>& U, int dA, int dB);
  static int schmidtRank(const std::vector<std::complex<double>>& U, int dA, int dB, double tol = 1e-10);
  static std::vector<std::complex<double>> transitionOperator(  // U_T = |psiA><psiB| (rank 1)
      const std::vector<std::complex<double>>& psiA, const std::vector<std::complex<double>>& psiB, int dA, int dB);
  static std::complex<double> transitionAmplitude(             // <psiA|U|psiB>
      const std::vector<std::complex<double>>& psiA, const std::vector<std::complex<double>>& U,
      const std::vector<std::complex<double>>& psiB, int dA, int dB);
};
}
```

**Locked conventions:** `vec(|a><b|) = a ⊗ conj(b)` (separable, Schmidt rank 1);
**C1 is the HS/duality identity** `<psiA|U|psiB> = <vec(U_T)|vec(U)> = Tr(U_T^H U)`;
Schmidt rank of `vec(U)` = #nonzero singular values of `U`. **The same class
replaces `InteractionSimulation`'s inline 256-dim Choi.**

## Fixtures (existing + inline; no new graph classes)

| Fixture | Source | Laplacian spectrum (closed form) | Role |
|------|------|------|------|
| **triangle** = `SimplexBoundarySphere(1)` | existing `Topology` (`S¹ = ∂Δ²`, 3 verts/3 edges) | `λ_k = 2 − 2cos((Φ + 2πk)/3)` with flux Φ via `Edge::phase`; harmonic iff Φ≡0 | C4/C5 (b₁=1) |
| **path** `0-1-2` | inline (`createVertex`/`createSimplex`) | `2 − 2cos(kπ/n)` (`{0,1,3}`); phase-independent (tree) | C5 tree (b₁=0) |
| **testbed** square `00-01-11-10` + diag `00-11` | inline | numpy oracle; b₁=2 | C3 (spec's representative cyclic fixture) |

**Provenance.** The triangle's flux spectrum is the tight-binding/Hückel
**Aharonov–Bohm ring** result `E(k) = −2t·cos((2πk+φ)/N)` (here `L = 2I − A`, so
`λ_k = 2 − 2cos((Φ+2πk)/N)`); see *Quantum rings for beginners*
([arXiv:cond-mat/0310064](https://arxiv.org/abs/cond-mat/0310064)) and *Transport
through quantum rings* ([arXiv:1403.1154](https://arxiv.org/abs/1403.1154)). The
path/cycle Laplacian spectra are standard spectral graph theory. `b₁` for every
fixture is cross-checked against `ChainComplex.bettiNumbers()[1]`.

## Mapping to the spec checks (C1–C5)

| Check | Fixture | Quantity | Expected | Tol | Refutes |
|------|------|------|------|------|------|
| **C1** value = amplitude | bare 2×2 `U`, unit `psiA,psiB` (seeded) | `transitionAmplitude` vs `<vec(U_T)|vec(U)>` & `Tr(U_T^H U)` | equal | `1e-12` | P1 (alg. half) |
| **C2** rank=Schmidt=connectivity | `U_T=|psiA><psiB|`; `I₂` (cup); `σ_x` | `schmidtRank`, matrix rank | 1 ⇒ separable/disconnected; 2 ⇒ entangled/connected | `1e-10·σ_max` | P4 |
| **C3** gauge invariance | testbed (b₁=2), random `w,θ,α` | `eigenvalues`, eigvec rephasing, `Σ phase` per cycle | spec(L) & all Φ_γ unchanged; `v→Gv` | `1e-12`/`1e-10` | P5 |
| **C4** flux in spectrum | triangle = `SimplexBoundarySphere(1)`, flux sweep | `SpectralGap`, `HarmonicDimension` | `2−2cos((Φ+2πk)/3)`; harmonic iff Φ≡0; gap→0 at Φ=π | `1e-12` | P5 |
| **C5** tree vs cycle | path (b₁=0); triangle/testbed (b₁≥1) | `SpectralGap`/`HarmonicDimension` over a θ sweep | tree θ-independent; cycle varies only through Φ_γ | `1e-12` | P5 |

Pass criterion: **all C1–C5 pass** ⇒ P4, P5 supported and the amplitude half of P1.

## Tickets (Cobordism Correspondence v0.1; issue-first, one per PR)

- **[#88]** mesh/spacetime: `Edge::phase` + `SpacetimeType::HERMITIAN_WEIGHTED`.
- **[#90]** cobordism: `HodgeLaplacian` (k=0) operator — assembly/spectra/harmonics/
  `gauge` (+ bindings, + C3 + numpy-oracle spectrum tests; fixtures =
  `SimplexBoundarySphere(1)` + inline).
- **[#95]** cobordism: `SpectralGap` + `HarmonicDimension` Observables (delegate to
  `HodgeLaplacian`; mirror `EulerCharacteristic`/`Signature` over `ChainComplex`) — C4/C5.
- **[#91]** quantum: `ChoiJamiolkowski` class (+ bindings, + C1/C2 tests).
- **[#92]** quantum: refactor `InteractionSimulation`'s inline Choi → `ChoiJamiolkowski`.
- **[#93]** examples: `examples/cobordism/algebraic_correspondence.py` — C1–C5
  pass/fail table + flux sweeps (§7/§8); figures not committed.
- **[#94]** docs: wire `cobordism.md` + this plan into the quantum-experiments
  toctree; clean `sphinx-build -E`.

*(Closed: #89 graph-fixture classes — the triangle already exists as
`SimplexBoundarySphere(1)`, and `CycleGraph`/`PathGraph`/`CompleteGraph` would
duplicate `Spacetime`. Flux is a Wilson-loop quantity, handled test-side in Stage 1
and via a `WilsonLoop` U(1) mode in Stage 2.)*

## Build & verification

- **Bindings:** `Edge::phase` in `src/mesh/Bindings.cpp`; enum in
  `src/spacetime/Bindings.cpp`; `HodgeLaplacian` in `src/cobordism/Bindings.cpp`;
  `ChoiJamiolkowski` in `src/quantum/Bindings.cpp`. `#include <pybind11/complex.h>`
  where complex crosses. No CMake change (auto-glob; Eigen `PUBLIC` on `tessera_core`;
  `quantum/` already links Eigen).
- **Verify:** `pip install -e ".[dev]"`; `pytest tests/cobordism tests/quantum -v`
  (C1–C5 green; `InteractionSimulation` tests unchanged); run the report; `sphinx-build
  -E` 0 warnings. Heavier sweeps honor the 10-CPU cap (precautionary; Stage-1
  matrices are tiny).

## Risks

1. **Convention bugs pass for the wrong reason** — assert against numpy, not the C++
   output alone; lock `D_ii=Σ|w|`, `vec(|a><b|)=a⊗conj(b)`, the C1 HS form.
2. **Kernel/degeneracy tolerance** — gap-ratio check; zero-count stable across
   `tol∈[1e-6,1e-10]`; the triangle's Φ=π degeneracy is expected.
3. **Complex over pybind** — `<pybind11/complex.h>`; round-trip test first.
4. **Coordinate-free magnitudes** — ensure the `Metric` doesn't overwrite
   explicitly-set squared lengths; set via `setSquaredLength` and assert.
5. **`InteractionSimulation` refactor** — keep the 256-dim Choi byte-identical; gate
   on its existing tests.

## After Stage 1 (Stage 2 milestone)

Stage 2 reuses the *same* `HodgeLaplacian` at `k≥1` (harmonic 1-forms on surface
fixtures via `ChainComplex` boundary matrices), and is where `Edge::multiplicity`
(Δ-complex identifications), `Spacetime::fromAdjacency`, and a **`WilsonLoop`
U(1)-connection mode** (general `Σ phase` holonomy, for T4) finally earn their keep.
It adds the Dijkgraaf–Witten ℤ₂ state-sum (T1/T3), interior bistellar moves for
triangulation-independence (T2, make-or-break), gluing/functoriality (T5),
cross-layer consistency (T4), and the Lorentzian variant (§5.6). Note **T3 must use
T³, not S²×S¹** (the triple cup product vanishes on S²×S¹ — a negative control).
