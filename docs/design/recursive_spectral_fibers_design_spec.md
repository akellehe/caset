# Recursive Spectral-Fiber Simulation — Design Specification

## 1. Purpose

Implement the formulation in
[`recursive_spectral_fibers_whitepaper.md`](recursive_spectral_fibers_whitepaper.md)
as a complete, falsifiable Tessera simulation. The implementation must discover
particles as persistent modular spectral components, derive color transport from
the Hodge data, enforce fermion statistics through simplicial grading, grow the
state space through graded tensor products, and identify a proton only from
post-optimization observables.

The simulation must scale. Exact algebraic and structure-exact reductions are the
default. Iterative numerical algorithms are permitted only with residual, spectral
gap, leakage, and conditioning certificates. Dense global diagonalization and
finite-difference gradients are not production paths.

Delivery is tracked by GitHub epic
[#763](https://github.com/akellehe/tessera/issues/763) and its dependency-ordered
child tickets.

## 2. Goals

- Replace “quark = hole” with “quark candidate = persistent high-modularity
  component with an odd, rank-three localized spectral fiber.”
- Represent a connected component as a coarse vertex using exact Schur/Kron
  response reduction.
- Build a recursive hierarchy of components and effective complexes.
- Realize the color `3`, `3̄`, `1`, and `8` sectors from three oriented edge qubits.
- Derive `SU(3)` transport and Wilson loops from neighboring spectral frames.
- Implement label-independent fermionic exchange and Pauli exclusion from the
  exterior grading.
- Represent the growing finite stages of the Fock expansion without allocating the
  full tensor product when its exact state is factorized or block sparse.
- Read quark, antiquark, gluon, meson, baryon, and proton candidates from the same
  underlying complex.
- Preserve current amplitude, Hodge, Regge, topology, and spectral-dimension tests.
- Provide an end-to-end deterministic simulation, checkpoint format, benchmark,
  analysis report, and animation.

## 3. Non-goals

- Do not insert holes, quark regions, a color singlet, a `uud` pattern, or a proton
  target in emergence mode.
- Do not introduce an independently sampled gauge connection. Fiber transport is a
  derived readout.
- Do not replace the current Regge-Hodge objective with a particle classifier.
- Do not call a perimeter constraint a Hilbert-space normalization.
- Do not identify a nonzero Hodge band with cohomology or assign it a period without
  an additional closedness certificate.
- Do not claim continuum QCD, a physical mass prediction, or a spin-statistics
  theorem from a finite numerical match.
- Do not make global modularity maximization appear exact. It is NP-hard; the
  selected partition must be identified as a deterministic discovery heuristic
  with exact score evaluation.
- Do not silently truncate the Fock state in certification mode.

## 4. Required modes

The public simulation interface exposes three modes.

### 4.1 Emergence mode

The production scientific mode. Optimize only the existing geometry/state
functional and permitted scale regulation. All particle and gauge quantities are
post-hoc observables. A proton either appears or does not.

### 4.2 Synthesis mode

Pin a specified carrier or spectral sector to establish existence, measure a
residual floor, or construct an oracle fixture. Output must be stamped `synthesis`;
its result is never counted as unforced emergence.

### 4.3 Replay mode

Load a checkpoint, recompute all derived hierarchies and certificates, and verify
that no cached or serialized choice changes the result.

## 5. Mathematical invariants

Every frame must certify the applicable invariants below.

### 5.1 Chain and orientation

$$
\partial_{k-1}\partial_k=0
$$

exactly over integers, with simplex orientation changing sign under odd vertex
permutations and no observable changing under a global relabeling.

### 5.2 Hodge metric

In the positive metric regime,

$$
\|L_k-L_k^{\dagger_W}\|\leq\epsilon_{\mathrm{Hodge}}.
$$

In the signed regime, report left/right eigen residuals and the biorthogonal
condition number. Do not apply a self-adjoint solver to a non-self-adjoint operator.

### 5.3 Coarse response

For every coarse component and random compatible interface vector `b`,

$$
\left|
\min_i [b;i]^\dagger L[b;i]-b^\dagger L_{\mathrm{eff}}b
\right|
\leq\epsilon_{\mathrm{Schur}}\|b\|^2.
$$

### 5.4 Fiber isometry

$$
\epsilon_G=\|\Phi^\dagger W\Phi-I\|,
\qquad
\epsilon_{\mathrm{eig}}=\|L\Phi-\Phi\Lambda\|.
$$

For a non-normal band, use the corresponding biorthogonal projector and report both
left and right residuals.

### 5.5 Transport leakage

$$
M_{AB}=\Phi_A^\dagger WT_{AB}\Phi_B,\qquad
\eta_{AB}=\|M_{AB}^\dagger M_{AB}-I\|.
$$

No `SU(3)` Wilson value is accepted unless `η_AB` and the two endpoint band gaps are
below configured thresholds.

### 5.6 Exterior algebra

The creation/annihilation matrices satisfy the CAR, and the sign of any bit-level
operation matches the wedge sign exactly. Duplicate complete modes wedge to zero.

### 5.7 Inductive compatibility

For the vacuum embedding `ι_M`, report

$$
\epsilon_\iota=\|\iota_MU_M-U_{M+1}\iota_M\|
$$

on the active carried subspace.

## 6. Data model

All new C++ public types live on classes in an existing Tessera namespace. No new
free-function API is introduced.

### 6.1 Edge quantum data

Extend or adapt the existing edge-attached state so one canonical record exposes:

```cpp
struct EdgeQuantumData {
  std::complex<double> squaredLength;
  Eigen::Vector2cd qubit;
  std::int8_t orientationSign;
};
```

`qubit` is normalized on construction. Reversing the edge applies one documented
conjugation/permutation convention and flips `orientationSign`. Serialization must
round-trip it exactly within floating-point representation.

### 6.2 Stable component identity

```cpp
class ComponentId {
 public:
  std::string canonicalHash() const;
  std::size_t level() const;
};
```

The hash is derived from the oriented incidence structure and parent lineage, not
raw vertex numbers. It is used for persistence matching and deterministic
tie-breaking, never as a physical observable.

### 6.3 Spectral band and fiber

```cpp
struct SpectralBandCertificate {
  int degree;
  std::size_t rank;
  double lowerGap;
  double upperGap;
  double localization;
  double projectorResidual;
  double gramDefect;
  double conditionNumber;
  bool selfAdjoint;
};

class SpectralFiber {
 public:
  Eigen::MatrixXcd rightFrame() const;
  Eigen::MatrixXcd leftFrame() const;
  Eigen::MatrixXcd projector() const;
  SpectralBandCertificate certificate() const;
};
```

Degenerate bands are represented by their projector. Individual eigenvectors are a
gauge choice and must not determine a particle identity.

### 6.4 Recursive component

```cpp
class SpectralComponent {
 public:
  ComponentId id() const;
  std::vector<std::size_t> simplexIds(int degree) const;
  ModularityRead modularity() const;
  PersistenceRead persistence() const;
  std::vector<SpectralFiber> fibers() const;
  Eigen::MatrixXcd effectiveOperator(int degree) const;
};
```

### 6.5 Derived transport

```cpp
struct FiberTransportRead {
  ComponentId from;
  ComponentId to;
  Eigen::Matrix3cd rawMap;
  Eigen::Matrix3cd su3Map;
  double leakage;
  double polarResidual;
  double determinantResidual;
  bool accepted;
};
```

### 6.6 Particle reads

```cpp
struct QuarkRead {
  ComponentId component;
  int exteriorParity;
  int colorRank;
  double baryonFlux;
  std::optional<double> isospin;
  std::optional<double> electricFlux;
  double confidence;
  std::vector<std::string> failedCertificates;
};

struct BaryonRead {
  std::array<ComponentId, 3> quarks;
  ComponentId boundComponent;
  double colorGramDeterminant;
  double colorFlux;
  double baryonFlux;
  std::optional<double> electricFlux;
  std::optional<double> totalJ2;
  std::optional<std::complex<double>> rotationCharacter;
  double persistence;
  std::vector<std::string> failedCertificates;
};
```

Unknown or uncertified values are `null`, not zero.

## 7. Proposed source layout

| Concern | Public header | Implementation | Focused tests |
|---|---|---|---|
| Recursive quotient | `include/cobordism/RecursiveQuotient.h` | `src/cobordism/RecursiveQuotient.cpp` | `tests/cobordism/test_recursive_quotient_python.py` |
| Intrinsic components | `include/observables/PersistentModularity.h` | `src/observables/PersistentModularity.cpp` | `tests/observables/test_persistent_modularity_python.py` |
| Spectral fibers | `include/observables/SpectralFiber.h` | `src/observables/SpectralFiber.cpp` | `tests/observables/test_spectral_fiber_python.py` |
| Color algebra | `include/observables/ColorFiber.h` | `src/observables/ColorFiber.cpp` | `tests/observables/test_color_fiber_python.py` |
| Derived connection | `include/observables/FiberConnection.h` | `src/observables/FiberConnection.cpp` | `tests/observables/test_fiber_connection_python.py` |
| Exterior/Fock state | `include/quantum/GradedFock.h` | `src/quantum/GradedFock.cpp` | `tests/quantum/test_graded_fock_python.py` |
| Particle classification | `include/observables/ParticleClusters.h` | `src/observables/ParticleClusters.cpp` | `tests/observables/test_particle_clusters_python.py` |
| Total-space exchange/spin | extend `include/observables/DiracKahler.h` or add a class beside it | matching source | `tests/observables/test_exchange_holonomy_python.py` |
| Orchestration | extend `include/cobordism/MultiCobordism.h` | matching source | `tests/cobordism/test_recursive_fiber_simulation.py` |
| User experiment | — | `examples/cobordism/recursive_baryon_simulation.py` | smoke/replay tests |

All public headers receive Python bindings and a `{doxygenfile}` entry in
[`cpp_api.md`](../source/cpp_api.md).

## 8. Algorithm A — intrinsic multiscale component discovery

### 8.1 Input graph

Derive a nonnegative similarity graph from the current complex. The first
implementation uses the existing weighted `1`-skeleton, with a documented monotone
map from complex edge magnitude to similarity. Optional qubit overlap may be exposed
as an ablation, but it must not be silently mixed into the default metric.

### 8.2 Partition objective

For resolution `γ`, evaluate generalized modularity exactly:

$$
Q_\gamma(P)=\frac1{2m}\sum_{ij}
\left(A_{ij}-\gamma\frac{k_ik_j}{2m}\right)
\mathbf1[c_i=c_j].
$$

Build on `ModularityOptimizer`, but add label-free partition discovery. Each local
move uses cached community degree and internal-weight totals, so its `ΔQ` is exact
and `O(deg(v))`. One complete sparse sweep is `O(|E|)` up to revisits.

The global optimum is not promised. Run deterministic multilevel aggregation from a
fixed seed sequence, retain the best exact score, and report the spread across
restarts. Canonical component hashes break equal-score ties.

### 8.3 Persistence

Scan a configured resolution sequence. Match components across adjacent resolutions
using maximum weighted overlap of their simplex support and spectral projectors.
A persistent component must pass minimum lifetime, support-overlap, and conductance
thresholds. Thresholds are analysis parameters, recorded in every checkpoint.

### 8.4 Acceptance tests

- relabeling leaves the partition hierarchy isomorphic;
- disconnected planted components are recovered exactly;
- a ring without a stable scale does not manufacture a persistent quark;
- exact modularity recomputation equals all cached `ΔQ` accumulations;
- current fixed-partition modularity behavior remains available and unchanged.

## 9. Algorithm B — spectral-band extraction

For every persistent component and configured form degrees:

1. assemble its restricted weighted Hodge operator;
2. request the smallest set of eigenpairs covering all candidate isolated bands;
3. group eigenvalues by a relative gap rule;
4. build a projector for each whole degenerate band;
5. calculate inverse participation/localization on the component;
6. reject a band with insufficient gap, excessive residual, or poor conditioning;
7. track the accepted projector through time and scale by principal angles.

The detector enumerates band ranks. It does not request rank three. The quark
classifier later selects persistent rank-three reads, avoiding a built-in color
answer.

For the self-adjoint case, use a sparse block eigensolver. For a small component,
use the exact dense self-adjoint solve. For the signed/non-normal case, compute
matched left and right subspaces and use the biorthogonal Riesz projector. Every
path reports residual and condition number.

## 10. Algorithm C — exact recursive quotient

For each accepted partition and Hodge degree:

1. classify cells as component interior or interface;
2. compute an exact integer nullspace basis for topological interior zero modes;
3. project the numerical operator onto the supported interior complement;
4. factor `L_II` once with a sparse rank-revealing factorization;
5. solve `L_II X=L_IB` without forming an inverse;
6. form `L_eff=L_BB-L_BI X`;
7. retain harmonic interior coordinates as the component fiber;
8. verify the quadratic response identity on deterministic probe vectors.

Nested reductions reuse child factorizations. A local topology or metric move
invalidates only the affected component ancestry.

## 11. Algorithm D — exact color kernel

`ColorFiber` operates on three oriented edge modes and supplies:

- the `N=0,1,2,3` exterior-sector projectors;
- creation and annihilation matrices;
- the eight normalized Gell-Mann generators on `N=1`;
- the exact Fourier color frame `F_3` built from `ω`;
- the `3`, `3̄`, singlet, and adjoint-octet projectors;
- `det(C)` and `det(C†C)` singlet certificates; and
- perimeter and Hilbert normalizers as distinct methods.

The constant algebra is generated once and checked at startup in debug builds. The
production operation count is constant.

Required exact tests:

$$
F_3^\dagger F_3=I,
\quad
[E_{ij},E_{k\ell}]=\delta_{jk}E_{i\ell}-\delta_{i\ell}E_{kj},
\quad
\operatorname{Tr}(\lambda_a\lambda_b)=2\delta_{ab}.
$$

Use algebraic expected values containing `√3` and `ω`; compare floating
representations only at the final boundary.

## 12. Algorithm E — spectral transport and Wilson observables

Given two accepted rank-three fibers:

1. construct the existing chain transfer `T_AB` from their connecting simplices;
2. compute `M_AB=Φ_A†WT_ABΦ_B`;
3. calculate leakage before normalization;
4. reject if leakage, gap, or condition thresholds fail;
5. take the `3×3` polar factor;
6. choose the determinant cube-root branch continuously from the previous frame;
7. store the resulting `SU(3)` map and all certificates;
8. multiply accepted maps around a loop and take normalized trace.

Gauge tests apply independent random `SU(3)` frame changes at every component and
require the Wilson trace to remain invariant. A deliberately leaking fixture must
be rejected even though polar normalization can produce a unitary matrix.

## 13. Algorithm F — graded Fock engine

### 13.1 Exact occupation representation

For up to the machine-word threshold, an exterior basis state is a bitset. Creating
mode `i` multiplies by

$$
(-1)^{\operatorname{popcount}(b\ \&\ ((1\ll i)-1))}.
$$

Above that threshold, use a chunked bitset with the same prefix-popcount rule. Mode
order comes from oriented component lineage. Relabeling rebuilds the canonical order
and applies the corresponding permutation parity.

### 13.2 Lazy exact state

Represent a state as an expression DAG with nodes:

- vacuum;
- sparse occupation block;
- graded tensor product;
- local unitary/cobordism map;
- direct sum by conserved occupation/parity sector; and
- antisymmetrized wedge.

Do not expand a tensor node until an operation crosses that partition. Memoize exact
subexpressions. Certification mode allows algebraically lossless compression only.
An optional approximation mode may use a stated singular-value truncation, but its
discarded norm is accumulated and printed in every amplitude result.

### 13.3 Gluon candidates

Even traceless quark-antiquark bilinears supply the `8` sector. A gluon candidate is
a persistent transported octet excitation with zero baryon flux and even parity.
Unbounded occupation is approached by adding more microscopic edge modes, not by
changing a qubit into an oscillator.

## 14. Algorithm G — exchange and spin holonomy

Track an isolated odd component subspace through a closed motion in configuration
space. Consecutive frames use polar overlap transport. The determinant of the loop
holonomy is the exchange character.

Required fixtures:

- one exchange of two identical odd clusters gives `-1`;
- two exchanges give `+1`;
- exchanging an odd cluster with an even composite gives `+1`;
- duplicate one-particle modes produce zero wedge norm;
- a vertex relabeling or in-band frame rotation changes no character;
- closing the spectral gap invalidates the read instead of emitting a sign.

The physical rotation path is separate from a label permutation. Build a geometric
`2π` loop of the cluster frame, evaluate its total-space holonomy, and require `-1`
before reporting spin `1/2`. This extends the total-space readout demanded by
[`joint_proton_spin_findings.md`](joint_proton_spin_findings.md); products of
per-hole Bloch vectors are insufficient.

## 15. Algorithm H — quark and baryon discovery

### 15.1 Quark classifier

Run on all persistent components, without feeding back into the optimizer. A quark
candidate requires:

- odd exterior parity;
- an accepted rank-three color band;
- bounded transport leakage;
- sufficient persistence and localization;
- an oriented world-tube flux near `+1/3` or `-1/3`; and
- refinement stability.

Flavor is reported only if an unlabeled, transported two-state spectral subclass
passes its own gap and persistence tests. Charge is reported only if the existing
Gauss-flux read is consistent across enclosing surfaces. Otherwise both fields are
unknown.

### 15.2 Bound-supercomponent search

At the next modular scale, enumerate components containing exactly three persistent
quark candidates. Require their lifetimes to overlap and their mutual transport to
remain inside the supercomponent.

### 15.3 Color singlet

Compute `s_color=det(C†C)` from the three normalized color columns. Report both the
complex determinant and its squared magnitude. Require vanishing net color flux as
an independent confinement check.

### 15.4 Proton classifier

A baryon may be called a proton only when all required values are certified:

| Observable | Required proton value |
|---|---:|
| persistent quark count | 3 |
| color Gram determinant | `1` within certificate |
| baryon flux | `+1` |
| flavor occupation | `uud` |
| electric Gauss flux | `+1` |
| total `J²` | `3/4` |
| `2π` character | `-1` |
| composite parity | odd |

A partial match is a “baryon candidate” with an explicit list of missing or failed
certificates, never a proton.

## 16. Optimizer and refinement integration

The base emergence objective remains the joint stationary functional already used
by `MultiCobordism`:

$$
F_{\mathrm{base}}=
\beta_R\|\nabla_zS_{\mathrm{Regge}}\|^2+
\eta_H\sum_k\|\nabla_zS_{\mathrm{Hodge},k}\|^2,
$$

with the existing, explicitly selected scale regulator where required. Particle
confidence, modularity, color determinant, Wilson loops, flavor, charge, and spin do
not enter this functional in emergence mode.

After an accepted geometry move or a configurable analysis cadence:

1. update affected Hodge/Regge caches;
2. update the local component hierarchy;
3. update affected spectral projectors and transports;
4. update the lazy Fock expression for the interaction;
5. evaluate particle reads; and
6. checkpoint the raw state plus certificates.

Refinement is driven by a posteriori error estimators: coarse-response residual,
band-gap loss, transport leakage, curvature concentration, and amplitude Gram
defect. It is not driven by a request for a hole or quark. Pachner/refinement moves
reuse the existing implementations and manifold/orientation gates.

## 17. Analytic-first performance contract

| Kernel | Preferred exact/structured path | Scaling target | Prohibited default |
|---|---|---:|---|
| topology | Smith normal form / integer boundary maps | sparse, component-local | Betti number from eigenvalue threshold alone |
| component score | cached exact `ΔQ` | near `O(|E|)` per sweep | recompute all communities per move |
| coarse operator | sparse Schur solves | affected component factorization | explicit dense inverse |
| uncoupled product spectrum | Künneth/Minkowski sums | output-sensitive | diagonalize full Kronecker matrix |
| local topology update | Woodbury/secular low-rank update | affected rank and star | rebuild every global operator |
| color algebra | fixed `3×3`/`8×8` formulas | `O(1)` | generic symbolic solver at runtime |
| singlet | `3×3` determinant/Gram determinant | `O(1)` | sampling color permutations |
| exchange sign | bit parity / orientation | `O(1)` per local action | phase estimation |
| Regge/Hodge derivatives | analytic complex/Wirtinger gradients | affected stars | finite differences |
| fiber transport | `r×r` overlap and polar factor, `r=3` | `O(nr²+r³)` | global gauge optimization |
| Fock state | lazy graded tensor DAG, sparse sectors | active support | eager `2^M` vector |
| eigenspace | sparse block solve + residual/gap | `O(iter·nnz(L)·r)` | dense global eigensolve |

Low-rank updates are accepted as exact only when applied to the full affected
subspace. Truncated Krylov or contour methods are certified numerical paths and must
report their residuals. A performance optimization may not change a topology,
amplitude, parity, or singlet verdict outside its declared tolerance.

## 18. Cache and invalidation design

Cache entries are keyed by a geometry revision and a canonical component ID:

- boundary/incidence blocks by degree;
- simplex metric weights and analytic derivatives;
- sparse factorization of `L_II`;
- integer nullspace/Betti data;
- localized spectral projectors;
- component sufficient statistics for modularity;
- fiber transport and Wilson products; and
- lazy Fock subexpressions.

Every accepted move publishes its touched simplices, affected stars, created/deleted
cells, and changed edge data. Invalidation walks from touched leaf components to the
root. Siblings remain valid. Replay mode can disable all caches and compare results
against the incremental path.

## 19. Checkpoint and analysis schema

Each frame stores:

```json
{
  "mode": "emergence",
  "geometry_revision": 0,
  "raw_complex": {},
  "edge_quantum_data": {},
  "objective": {},
  "hierarchy": [],
  "fibers": [],
  "transports": [],
  "fock": {
    "active_modes": 0,
    "exact": true,
    "discarded_norm": 0.0
  },
  "particles": {
    "quarks": [],
    "gluons": [],
    "baryons": []
  },
  "certificates": {},
  "provenance": {
    "seed": 0,
    "config_hash": "",
    "commit": ""
  }
}
```

Matrices too large for JSON use a versioned binary sidecar with content hashes.
Checkpoint filenames follow the repository's per-run suffix convention. Readers
must reject an unknown schema version.

## 20. Verification matrix

### 20.1 Exact unit fixtures

1. Chain boundary squared is exactly zero.
2. A hand-solvable path and triangle match their analytic Kron reductions.
3. Product-complex spectra equal pairwise eigenvalue sums.
4. `F_3` is unitary and its determinant has unit modulus.
5. Gell-Mann commutators and trace normalization are exact within representation
   rounding.
6. CAR and Pauli determinant identities hold for every three-mode basis state.
7. The `Λ^3 C^3` state is invariant under random `SU(3)` matrices.
8. Wilson traces are invariant under independent local frame rotations.
9. A leaking transfer is rejected before polar normalization.
10. Single/double exchange characters are `-1/+1`.
11. Vacuum embedding preserves all existing amplitudes.
12. Cached low-rank updates equal cold recomputation.

### 20.2 Property tests

- random vertex relabeling changes no hierarchy, particle verdict, Wilson value, or
  exchange character;
- random orientation-preserving retriangulation changes amplitudes only by the
  measured Gram/coarse-response defect;
- all accepted bands remain accepted under in-band basis rotations;
- thresholds near a closing gap return “uncertified,” not a discontinuous particle
  label;
- adding a disconnected vacuum component does not change prior observables;
- quark-antiquark pair creation preserves total parity and baryon flux.

### 20.3 Existing regression gates

- all `tests/cobordism/` tests remain green;
- amplitude/isometry evidence in `cobordism-results.md` is reproduced;
- joint Regge-Hodge gradient and performance tests remain green;
- the current proton animation continues to run with its old behavior unless the
  new analysis overlay is selected; and
- spectral-dimension analysis has no statistically significant regression at its
  pinned fixtures.

### 20.4 End-to-end acceptance

The epic is complete when one command can:

1. start from a documented neutral initial complex and seed;
2. run unforced joint stationarity with refinement;
3. build and persist the recursive component hierarchy;
4. maintain an exact or certified Fock state through interactions;
5. report all quark, gauge, exchange, and baryon certificates;
6. distinguish “no baryon,” “baryon candidate,” and “certified proton” without a
   target-dependent code path;
7. replay the checkpoint with cold caches and reproduce the verdict;
8. render the hierarchy, color transport, Wilson loops, and particle world tubes;
9. emit scaling data for at least three problem sizes; and
10. keep the analytic/structured path faster than the dense reference on the
    crossover fixture while agreeing within certificate.

An unforced proton is a scientific success condition, not a software completion
condition. The software is complete if it can return a rigorous negative result.

## 21. Delivery waves and dependencies

### Wave 0 — exact foundations

- analytic/structured solver contract and benchmark harness;
- recursive Schur quotient;
- exterior algebra and graded tensor primitives.

### Wave 1 — intrinsic fibers

- label-free persistent modularity;
- localized spectral-band/projector tracking;
- exact triangle color algebra.

### Wave 2 — transport and statistics

- derived `SU(3)` connection and Wilson loops;
- exchange determinant-line holonomy and physical `2π` loop;
- lazy inductive-limit Fock engine.

### Wave 3 — particles and interactions

- quark/antiquark classification, baryon/charge/flavor flux reads;
- even color-octet/gluon sector;
- three-quark singlet and proton total-space readout.

### Wave 4 — unforced complete simulation

- optimizer/refinement integration;
- multiscale continuum and spectral-dimension validation;
- deterministic campaign, checkpoint/replay, benchmark, and animation.

## 22. Merge discipline

Every implementation ticket must include:

- the exact identity or certified approximation it implements;
- an explicit statement of whether it affects ontology, dynamics, or readout;
- focused exact fixtures and relabeling tests;
- a cold-recompute comparison for any cache;
- a benchmark before and after;
- no finite-difference fallback where an analytic derivative exists;
- no particle observable added to the emergence objective; and
- a findings report under `docs/design/` that records positive and negative results.

This discipline is what keeps a creative geometric program scientifically sharp:
the implementation may discover an unexpected structure, but it may not hide how
that structure was selected or how accurately it was computed.
