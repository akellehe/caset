# Recursive Spectral-Fiber Simulation — Design Specification

## 1. Purpose

Implement the formulation in
[`recursive_spectral_fibers_whitepaper.md`](recursive_spectral_fibers_whitepaper.md)
as a complete, falsifiable Tessera simulation. The implementation must discover
particles as persistent modular spectral components, derive rank-dependent
transport from the Hodge data, enforce fermion statistics through exterior
grading, grow the state space by second-quantizing the glued one-particle
operator, and identify a proton only from post-optimization observables. The
quasi-free covariance layer is the exact primary state representation of
the current theory; the programme's organizing open question is whether
an exact covariance-only proton exists, or a genuinely non-Gaussian,
geometry-mediated interaction is required.

The simulation must scale. Exact algebraic and structure-exact reductions are the
default. Iterative numerical algorithms are permitted only with residual, spectral
gap, leakage, and conditioning certificates. Dense global diagonalization and
finite-difference gradients are not production paths.

Delivery is tracked by GitHub epic
[#763](https://github.com/akellehe/tessera/issues/763) and its dependency-ordered
child tickets.

The corrections prompted by external review, including the findings that were
qualified or rejected, are recorded in
[`recursive_spectral_fibers_referee_response.md`](recursive_spectral_fibers_referee_response.md).

## 2. Goals

- Replace “quark = hole” with “quark candidate = persistent certified
  component with an odd, rank-three, triangle-anchored localized spectral
  fiber.”
- Represent a connected component as a coarse response vertex using exact static
  Schur/Kron reduction and shifted Feshbach or certified AMLS reduction for
  nonzero spectral bands.
- Build a recursive hierarchy of operator-valued response networks; emit a
  simplicial or cellular-sheaf realization only when its incidence/restriction
  factorization is certified.
- Realize the color `3`, `3̄`, `1`, and `8` sectors from three oriented edge-mode
  factors and anchor every abstract rank-three color band to oriented triangles.
- Derive `U(r)` transport from neighboring spectral frames. At rank three retain
  the determinant line, projective `SU(3)/Z3` class, and explicit center lift.
- Implement label-independent fermionic exchange and Pauli exclusion from the
  exterior grading.
- Represent the growing finite stages of the Fock expansion without allocating the
  full tensor product when its exact state is factorized or block sparse.
- Make the covariance matrix the exact primary representation of the
  quasi-free sector: evolve `Γ` by `iΓ̇=[h,Γ]` and evaluate every polynomial
  certificate — occupations, parity, Pauli/Gram, `|S_ABC|²`, `<J²>`, and
  `Var(J²)` — by Wick contraction; materialize a Fock vector only for oracle
  tests or explicitly non-Gaussian boundary data.
- Decide the covariance-only dichotomy empirically: either an exact
  covariance-only proton exists, or a genuinely non-Gaussian,
  geometry-mediated interaction is required; a certified quasi-free
  sharp-spin obstruction is a first-class outcome, not a failure.
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
- Do not serialize one normalized pure state per edge and infer that the global
  state is their product. Each edge supplies a two-level mode; the state is a
  generally entangled global Fock vector or density operator.
- Do not claim that a plain Schur complement preserves nonzero spectrum, that a
  generic Schur-reduced Hodge block is another simplicial complex, or that every
  response network is automatically a cellular-sheaf Laplacian.
- Do not discard the determinant phase of a rank-three polar factor or choose a
  cube-root branch without recording its `Z3` center sector.
- Do not require a Kasteleyn orientation for the abstract CAR/Fock algebra. A
  spin-structure certificate is required only for a continuum physical-spin lift.
- Do not infer Kähler-Dirac taste multiplicity from occupation-number exterior
  algebra unless the one-particle operator is actually promoted to `d-d*` on all
  cochain degrees.
- Do not present classical or mean-field geometry backreaction as a source of
  non-Gaussian correlations; it is Gaussian-closed.
- Do not adopt a quartic interaction, quantized geometry entangled with the
  fermions, beyond-mean-field geometry integration, a cobordism map that is
  not the second quantization of a one-particle map, or
  measurement/postselection without an explicit scope decision: these are
  the only named routes to genuine non-Gaussianity, and none is part of the
  present model.

## 4. Required modes

The public simulation interface exposes three modes.

### 4.1 Emergence mode

The production scientific mode. Optimize only the existing geometry/state
functional and permitted scale regulation. All particle and gauge quantities are
post-hoc observables. A proton either appears or does not.

Emergence has two labeled, Gaussian-closed sub-modes, recorded in
provenance: `strict`, in which the state never acts back on the geometry,
and `meanfield`, certificates-blind backreaction in which only the carried
state's energy density enters the base objective through the
covariance-layer loop `h=h(Γ,g)`. Every particle certificate is firewalled
from both sub-modes.

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

In a Hermitian signed regime, report the inertia of `Φ†WΦ` and normalize it to a
signature matrix `J=diag(I_p,-I_q)`. Negative signature is not automatically an
antiparticle; existing pair-creation experiments do exhibit an
opposite-signature selection rule with conserved real part, which is
recorded as second-tier measured evidence for that reading while the
identification itself stays third-tier. In a non-normal regime, report left/right eigen residuals and the
biorthogonal condition number. Do not apply a self-adjoint solver to a
non-self-adjoint operator.

### 5.3 Coarse response

For every positive self-adjoint coarse component and compatible interface vector
`b`, the static certificate is

$$
\left|
\min_i [b;i]^\dagger L[b;i]-b^\dagger L_{\mathrm{eff}}b
\right|
\leq\epsilon_{\mathrm{Schur}}\|b\|^2.
$$

In a Hermitian indefinite sector replace `min` by stationarity. In a non-normal
sector test block elimination and the compatibility condition

$$
L_{IB}b\perp\ker L_{II}^{\dagger}.
$$

For a nonzero frequency window `Ω`, use the exact response pencil

$$
F_B(\lambda)=L_{BB}-\lambda I-
L_{BI}(L_{II}-\lambda I)^{-1}L_{IB},\qquad \lambda\in\Omega,
$$

or a Craig-Bampton/AMLS linear surrogate. Report the window, discarded-mode gap,
and resolvent/eigen residual. No nonzero-spectrum claim is attached to `F_B(0)`.
Report multiplicity honestly: for `λ` outside `spec L_II`,
`det(L−λI)=det(L_II−λI)·det F_B(λ)`; the order of the zero of `det F_B(·)`
at `λ` is the algebraic multiplicity and `dim ker F_B(λ)` the geometric
multiplicity, equal only in the self-adjoint/semisimple setting.

### 5.4 Fiber isometry

$$
\epsilon_G=\|\Phi^\dagger W\Phi-J\|,
\qquad
\epsilon_{\mathrm{eig}}=\|L\Phi-\Phi\Lambda\|,
$$

where `J=I` in the positive regime and is the reported signature matrix in an
indefinite Hermitian regime.

For a non-normal band, use the corresponding biorthogonal projector and report both
left and right residuals.

### 5.5 Transport leakage

$$
M_{AB}=\Phi_A^\dagger WT_{AB}\Phi_B,\qquad
\eta_{AB}=\|M_{AB}^\dagger M_{AB}-I\|.
$$

For non-normal frames replace the left factor by `Ψ_A†` with `Ψ_A†W_AΦ_A=I`;
such transport is generally `GL(r,C)` and is not silently unitary-normalized. No
unitary Wilson value is accepted unless rank, `η_AB`, endpoint band gaps, metric
signatures, and frame conditioning pass their configured thresholds. At rank three,
the determinant-line and center-lift data are mandatory.

### 5.6 Exterior algebra

The creation/annihilation matrices satisfy the CAR, and the sign of any bit-level
operation matches the wedge sign exactly. Duplicate complete modes wedge to zero.

### 5.7 Inductive compatibility

For the vacuum embedding `ι_M`, report

$$
\epsilon_\iota=\|\iota_MU_M-U_{M+1}\iota_M\|
$$

on the active carried subspace.

### 5.8 Quasi-free closure and covariance purity

Every generator in the model is `dΓ` of a one-particle map, so quasi-free
states remain quasi-free — including under the mean-field self-consistency
`h=h(Γ(t),g(t))`. For the covariance layer report

$$
\epsilon_\Gamma=\|\Gamma^2-\Gamma\|
$$

for pure Slater states across evolution and across every mean-field
iteration, and require Wick-evaluated certificates to match dense Fock
references on crossover fixtures.

## 6. Data model

All new C++ public types live on classes in an existing Tessera namespace. No new
free-function API is introduced.

### 6.1 Edge mode data

Each edge indexes one two-level occupation mode. The edge record stores geometry
and orientation, not a normalized local pure state:

```cpp
struct EdgeQuantumData {
  std::complex<double> squaredLength;
  std::int8_t orientationSign;
  std::uint64_t modeId;
};
```

`modeId` identifies the factor `span{|0>,|1>}` inside the global exterior Fock
space. Reversing the edge applies one documented conjugation/permutation convention
and flips `orientationSign`. A per-edge occupation or Bloch vector is derived from
the global density operator; a stored product preparation is an optional boundary
fixture and must be labeled as such.

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
  int positiveSignature;
  int negativeSignature;
  double frequencyLower;
  double frequencyUpper;
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

Fibers of adjacent components may overlap on shared interface cells. The
next-level one-particle space is the abstract labeled sum `⊞_v E_v`,
carried with its embedding `J` and Gram matrix `G=J†WJ`; a run declares
exactly one of: carry `G` in every subsequent formula, certify `‖G−I‖≤ε`
through the composable amplitude budget, or quotient `ker G`. An internal
direct sum in `C(K)` is never asserted.

### 6.5 Derived transport

```cpp
struct FiberTransportRead {
  ComponentId from;
  ComponentId to;
  int rank;
  Eigen::MatrixXcd rawMap;
  Eigen::MatrixXcd unitaryMap;
  std::complex<double> determinantPhase;
  int centerSector;
  double leakage;
  double polarResidual;
  double determinantResidual;
  double frameConditionNumber;
  bool projectiveOnly;
  bool accepted;
};
```

### 6.6 Particle reads

```cpp
struct QuarkRead {
  ComponentId component;
  int exteriorParity;
  int colorRank;
  double triangleAnchorScore;
  double anchorPhaseCoherence;
  std::optional<int> determinantWinding;
  std::string windingClosure;
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
  std::optional<double> varJ2;
  std::optional<std::complex<double>> rotationCharacter;
  double persistence;
  std::vector<std::string> failedCertificates;
};
```

Unknown or uncertified values are `null`, not zero. `windingClosure`
records the declared closure of an open world-tube segment
(matched-reference transport or boundary-register trivialization); a
winding without a closure specification is not certified. `varJ2` is the
Wick-evaluated `Var(J²)` on quasi-free candidates; the sharp spin
certificate requires both `totalJ2=3/4` and `varJ2≈0`.

### 6.7 Covariance-layer state

```cpp
class CovarianceState {
 public:
  Eigen::MatrixXcd gamma() const;        // Γ_ij = <a_j† a_i>
  double purityDefect() const;           // ‖Γ² − Γ‖
  double occupation(std::size_t mode) const;
  std::complex<double> wickExpectation(const NormalPolynomial& poly) const;
};
```

The covariance layer is the primary exact representation of the quasi-free
sector. It is initialized from accepted band projectors (`Γ=P`) or from
boundary registers, evolved by the one-particle transport, and queried by
Wick contraction. The graded Fock DAG of Algorithm F remains the dense
oracle and the carrier for explicitly non-Gaussian boundary data.

## 7. Proposed source layout

| Concern | Public header | Implementation | Focused tests |
|---|---|---|---|
| Recursive quotient | `include/cobordism/RecursiveQuotient.h` | `src/cobordism/RecursiveQuotient.cpp` | `tests/cobordism/test_recursive_quotient_python.py` |
| Intrinsic components | `include/observables/PersistentModularity.h` | `src/observables/PersistentModularity.cpp` | `tests/observables/test_persistent_modularity_python.py` |
| Spectral fibers | `include/observables/SpectralFiber.h` | `src/observables/SpectralFiber.cpp` | `tests/observables/test_spectral_fiber_python.py` |
| Color algebra | `include/observables/ColorFiber.h` | `src/observables/ColorFiber.cpp` | `tests/observables/test_color_fiber_python.py` |
| Derived connection | `include/observables/FiberConnection.h` | `src/observables/FiberConnection.cpp` | `tests/observables/test_fiber_connection_python.py` |
| Exterior/Fock state | `include/quantum/GradedFock.h` | `src/quantum/GradedFock.cpp` | `tests/quantum/test_graded_fock_python.py` |
| Covariance layer | `include/quantum/CovarianceState.h` | `src/quantum/CovarianceState.cpp` | `tests/quantum/test_covariance_state_python.py` |
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

The current `ModularityOptimizer` evaluates Newman-Girvan modularity on a
combinatorial/nonnegative one-skeleton and therefore does not encode signed or
complex Hodge weights. Treat it only as a deterministic proposal generator and
record the resolution parameter. The downstream gap, localization, persistence,
transport, and anchor certificates are weight-aware and decisive. Include explicit
Fortunato-Barthélemy resolution-limit fixtures so modularity cannot imprint a fake
preferred scale on the recursive hierarchy. Modularity proposes candidate
supports; it may not veto an otherwise certified fiber.

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

## 10. Algorithm C — recursive response reduction

For each accepted partition and Hodge degree:

1. classify cells as component interior or interface;
2. compute an exact integer nullspace basis for topological interior zero modes;
3. project the numerical operator onto the supported interior complement;
4. factor `L_II` once with a sparse rank-revealing factorization;
5. solve `L_II X=L_IB` without forming an inverse and form the exact static
   response `L_eff=L_BB-L_BI X`;
6. for every accepted nonzero band window, evaluate shifted solves
   `(L_II-λI)X(λ)=L_IB` and the exact Feshbach response `F_B(λ)`;
7. when a reusable linear eigenproblem is needed, retain interface constraint modes
   and selected fixed-interface modes in a Craig-Bampton/AMLS basis;
8. retain harmonic, resonant, and selected interior coordinates as vertex stalks;
9. assemble the next-level one-particle space as the abstract labeled sum
   `⊞_v E_v` with embedding `J` and Gram matrix `G=J†WJ`, declaring exactly
   one of carry-`G`, certify-`‖G−I‖≤ε`, or quotient-`ker G`; never assert an
   internal direct sum;
10. emit an operator-valued quotient graph; emit a cellular-sheaf realization only
    if explicit restriction maps reproduce the response blocks and satisfy their
    composition rules; and
11. verify static quadratic response and band-window resolvent/eigen residuals on
    deterministic probe vectors.

Nested reductions reuse child factorizations and shifted factorizations by band
window. A local topology or metric move invalidates only the affected component
ancestry. Plain static Schur reduction is never used as a claim of nonzero spectral
preservation.

## 11. Algorithm D — exact color kernel

`ColorFiber` operates on three oriented edge modes and supplies:

- the `N=0,1,2,3` exterior-sector projectors;
- creation and annihilation matrices;
- the eight normalized Gell-Mann generators on `N=1`;
- the exact Fourier color frame `F_3` built from `ω`;
- the `3`, `3̄`, singlet, and adjoint-octet projectors;
- `det(C)` and `det(C†C)` singlet certificates; and
- the complex-squared-length color vector `c=z/||z||_2`;
- perimeter and Hilbert normalizers as distinct methods; and
- weighted triangle-anchor matrices `A_τ=|W_τ|^{1/2}R_τΦ`, calibrated atlas
  scores and anchor profiles, and phase-coherence certificates for abstract
  rank-three bands.

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

An accepted color fiber need not concentrate on one literal triangle. It must
have a stable calibrated atlas score `a²=Σ_τ w_τ|det A_τ|²` with
`A_τ=|W_τ|^{1/2}R_τΦ` and the convex weighting `{w_τ}` declared before the
data are examined, plus coherent determinant-line phases on overlapping
oriented triangles. In the positive regime each `|det A_τ|²≤1`, so
`a²∈[0,1]` with value one exactly at full concentration on the weighted edge
span; the reported datum is the profile — maximal term, participation ratio,
and determinant-phase dispersion — alongside the score. Signed sectors
restrict with `|W_τ|^{1/2}` and report the restricted block's Krein
signature separately. A single-triangle fixture is the exact oracle; an
extended anchored fiber is the production case.

## 12. Algorithm E — spectral transport and Wilson observables

Given two accepted equal-rank fibers:

1. construct the existing chain transfer `T_AB` from their connecting simplices;
2. in the positive self-adjoint regime compute `M_AB=Φ_A†W_AT_ABΦ_B`; in the
   non-normal regime compute `M_AB=Ψ_A†W_AT_ABΦ_B`;
3. calculate rank, singular values, leakage, endpoint gaps, metric signatures, and
   frame condition numbers before normalization;
4. reject if any applicable threshold fails;
5. in the positive regime take the `r×r` polar factor `V_AB∈U(r)`;
6. at rank three store `V_AB`, `det V_AB`, and its projective
   `PU(3)≅SU(3)/Z3` class;
7. if a fundamental `SU(3)` lift is requested, continue a cube-root branch from a
   fixed base frame and record the accumulated `Z3` center sector;
8. multiply accepted maps around a loop and report full `U(r)`, determinant-line,
   projective/adjoint, and explicitly lifted fundamental observables as applicable;
9. for a closed full-rank world-tube family, compute the unwrapped determinant
   winding and invalidate it if the gap or rank closes; for an open cobordism
   segment compute only the relative winding, closing the composite with the
   inverse matched reference transport or with fixed boundary-register
   endpoint trivializations and recording the closure specification; and
10. in the non-normal regime retain the certified `GL(r,C)` transport unless a
    separate pseudo-unitary reduction is justified by matching Krein signatures.

Gauge tests apply independent random `U(r)` frame changes at every component and
require closed holonomies to transform by base-point conjugation. Rank-three tests
also exercise all three cube-root branches: center-blind observables must agree,
while a lifted fundamental observable must report the branch/center sector. A
deliberately leaking fixture must be rejected even though polar normalization can
produce a unitary matrix.

## 13. Algorithm F — quasi-free covariance layer and graded Fock engine

The one-particle edge space is `h=span{|e>}` and the global carrier is
`F_-(h)=Λ•h`. The covariance layer below is the primary exact representation
of the quasi-free sector; the lazy Fock DAG is the dense oracle and the
carrier for explicitly non-Gaussian boundary data. Per-edge occupations are
marginals, not a product-state ontology.

For a block one-particle operator `L=[[L_A,C],[C†,L_B]]`, construct
`dΓ(L)=Σ_ij L_ij a_i†a_j`. Verify exactly that direct sums become graded tensor
products and that coupling blocks become hopping terms. Free many-body eigenvalues
are occupation subset sums of one-particle eigenvalues.

### 13.1 Covariance layer (primary quasi-free path)

Represent the number-conserving quasi-free state by `Γ_ij=<a_j†a_i>` with
`iΓ̇=[h,Γ]` (equivalently, conjugation by the one-particle transport of a
cobordism step) and `Γ²=Γ` for pure Slater states; a pairing sector extends
`Γ` to the full Nambu covariance without changing the closure statement.
Initialize from accepted band projectors (`Γ=P`) or boundary registers.
Evaluate every polynomial certificate by Wick contraction: occupations,
parities, Pauli/Gram determinants, the color wedge `|S_ABC|²`, and both
`<J²>` and `Var(J²)=<(J²)²>−<J²>²`. Implement the mean-field loop `h=h(Γ,g)`
used by the certificates-blind backreaction sub-mode, with a purity
certificate at every iteration. Never materialize a Fock vector on the
quasi-free path; cross-validate against the lazy DAG of §13.3 on quasi-free
fixtures.

### 13.2 Exact occupation representation

For up to the machine-word threshold, an exterior basis state is a bitset. Creating
mode `i` multiplies by

$$
(-1)^{\operatorname{popcount}(b\ \&\ ((1\ll i)-1))}.
$$

Above that threshold, use a chunked bitset with the same prefix-popcount rule. Mode
order comes from oriented component lineage. Relabeling rebuilds the canonical order
and applies the corresponding permutation parity. This order is a compilation
choice, not extra physical data; no Kasteleyn orientation is required for the
abstract exterior algebra.

### 13.3 Lazy exact state

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

### 13.4 Gluon candidates

Even traceless quark-antiquark bilinears supply the `8` sector. A gluon candidate is
a persistent transported octet excitation with zero baryon flux and even parity.
Unbounded occupation is approached by adding more microscopic edge modes, not by
changing a qubit into an oscillator.

## 14. Algorithm G — exchange and spin holonomy

Track an isolated odd component subspace through a closed motion in configuration
space. Consecutive frames use certified overlap transport. The determinant of the
raw loop holonomy contains an ordinary path-dependent Berry phase and is not itself
the exchange sign. Construct a non-exchanging reference loop with the same geometric
footprint and report

$$
\widehat\chi_F=\det U_{\mathrm{exchange}}/\det U_{\mathrm{reference}}.
$$

Independently track the permutation of localized odd component blocks and report its
parity plus the residual in-block motion after reference cancellation.

Required fixtures:

- the normalized single exchange ratio and structural permutation parity give `-1`;
- the normalized double exchange ratio gives `+1`;
- exchanging an odd cluster with an even composite gives `+1`;
- duplicate one-particle modes produce zero wedge norm;
- a vertex relabeling or in-band frame rotation changes no character;
- closing the spectral gap invalidates the read instead of emitting a sign.

The physical rotation path is separate from a label permutation, and it is
never left abstract: the geometric `2π` loop is the documented total-space
spin holonomy cycle of the cluster frame, executed as a closed loop with a
matched co-moving reference loop; require their determinant ratio to be
`-1` before reporting spin `1/2`. If the emergent geometry is
manifold-like and the claim is continuum spin, also construct or reject a lift of
the frame holonomy from `SO(d)` to `Spin(d)` and report any `w2` obstruction. A
Kasteleyn orientation is only a possible surface-dimer implementation, not the
general spin certificate. Require the doubly cancelled spin-statistics ratio
`χ̂(exchange)χ̂(2π)^{-1}` to be `+1`. This extends the total-space readout demanded by
[`joint_proton_spin_findings.md`](joint_proton_spin_findings.md); products of
per-hole Bloch vectors are insufficient.

## 15. Algorithm H — quark and baryon discovery

### 15.1 Quark classifier

Run on all persistent components, without feeding back into the optimizer. A quark
candidate requires:

- odd exterior parity;
- an accepted rank-three color band;
- a stable oriented-triangle anchor score and determinant-phase coherence;
- bounded transport leakage;
- sufficient persistence and localization;
- a certified relative determinant-line winding `ν=+1` or `-1` in the
  closed-composite sense of Algorithm E (matched-reference or
  boundary-register closure recorded), provisionally interpreted as baryon
  flux `B=ν/3`; and
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
| triangle anchor | accepted for all 3 quark fibers |
| baryon flux | `+1` |
| flavor occupation | `uud` |
| electric Gauss flux | `+1` |
| total `J²` | `3/4` |
| `Var(J²)` | `≈0`, Wick-certified on quasi-free candidates |
| normalized `2π` character | `-1` |
| spin lift | accepted when a continuum spin claim is made |
| composite parity | odd |

A partial match is a “baryon candidate” with an explicit list of missing or failed
certificates, never a proton.

The variance row separates a proton from an accidental expectation value: a
Gaussian state can be an exact `J²` eigenstate, but a generic Slater
determinant with `<J²>=3/4` need not be spin-1/2. If every certificate
except the sharp spin passes across the entire accepted covariance-only
class under refinement, the verdict is `quasi-free sharp-spin obstruction`
— a branch point mandating exactly one of the named non-Gaussian mechanisms
as an explicit scope decision, never a silent failure and never a
refutation of the geometry.

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

Emergence runs in one of the two labeled, Gaussian-closed sub-modes of
§4.1, recorded in provenance: `strict` (no state backreaction) and
`meanfield` (certificates-blind backreaction, in which only the carried
state's energy density enters `F_base` through the covariance-layer loop
`h=h(Γ,g)`). The firewall below applies to both sub-modes.

After an accepted geometry move or a configurable analysis cadence:

1. update affected Hodge/Regge caches;
2. update the local component hierarchy;
3. update affected spectral projectors and transports;
4. update the covariance-layer state and, where a non-Gaussian boundary
   sector is active, the lazy Fock expression for the interaction;
5. evaluate particle reads; and
6. checkpoint the raw state plus certificates.

In emergence mode, geometry-changing refinement is driven only by
particle-independent geometric/numerical indicators already in the base problem:
Regge/Hodge stationarity residuals, curvature concentration, mesh quality, and
solver discretization error. Coarse-response residual, band-gap loss, transport
leakage, modularity, Wilson/center reads, exchange reads, anchor score, and amplitude
Gram defect may trigger post-hoc recomputation or mark a certificate unknown, but
they cannot accept, reject, or prioritize a geometry move. Any carrier refinement
driven by those quantities is synthesis and must be stamped accordingly.
Pachner/refinement moves reuse the existing implementations and
manifold/orientation gates.

## 17. Analytic-first performance contract

| Kernel | Preferred exact/structured path | Scaling target | Prohibited default |
|---|---|---:|---|
| topology | Smith normal form / integer boundary maps | sparse, component-local | Betti number from eigenvalue threshold alone |
| component score | cached exact `ΔQ` | near `O(|E|)` per sweep | recompute all communities per move |
| coarse response | sparse static/shifted Schur solves; AMLS band surrogate | affected component/window factorization | explicit dense inverse or DC spectrum claim |
| product/Fock spectrum | Künneth sums only for product complexes; `dΓ` subset sums for Fock sectors | output-sensitive | diagonalize full Kronecker/Fock matrix |
| quasi-free state | covariance evolution `iΓ̇=[h,Γ]` plus Wick contraction | `O(M²)`–`O(M³)` per step | eager `2^M` vector for a quasi-free sector |
| local topology update | Woodbury/secular low-rank update | affected rank and star | rebuild every global operator |
| color algebra | fixed `3×3`/`8×8` formulas | `O(1)` | generic symbolic solver at runtime |
| singlet | `3×3` determinant/Gram determinant | `O(1)` | sampling color permutations |
| exchange sign | bit parity plus reference-cancelled determinant holonomy | `O(1)` algebraic sign; `O(r³)` per overlap | raw determinant phase as sign |
| Regge/Hodge derivatives | analytic complex/Wirtinger gradients | affected stars | finite differences |
| fiber transport | `r×r` overlap; `U(r)` polar; determinant/projective split at `r=3` | `O(nr²+r³)` | hard-coded `SU(3)` at every scale |
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
  "submode": "strict",
  "geometry_revision": 0,
  "raw_complex": {},
  "edge_quantum_data": {},
  "objective": {},
  "hierarchy": [],
  "fibers": [],
  "transports": [],
  "covariance": {
    "active": true,
    "purity_defect": 0.0
  },
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
2. A hand-solvable path and triangle match their analytic static Kron reductions.
3. A hand-solvable block pencil matches the shifted Feshbach response across its
   declared frequency window; the same fixture demonstrates that static Schur does
   not preserve its nonzero eigenvalues.
4. Product-complex one-particle spectra equal pairwise sums, while `dΓ(L)` spectra
   equal occupation subset sums.
5. `F_3` is unitary and its determinant has unit modulus.
6. Gell-Mann commutators and trace normalization are exact within representation
   rounding.
7. CAR and Pauli determinant identities hold for every three-mode basis state and
   every induced mode reordering.
8. The `Λ^3 C^3` state is invariant under random `SU(3)` matrices.
9. A literal triangle and an extended oriented triangle atlas pass anchor tests;
   an abstract unanchored rank-three band fails.
10. Closed `U(r)` holonomies are gauge covariant; at `r=3`, all cube-root branches
    agree on projective/adjoint observables and expose their distinct center lifts.
11. A leaking or ill-conditioned transfer is rejected before polar normalization.
12. Raw exchange loops may contain arbitrary common Berry phase, while the matched
    single/double exchange ratios are `-1/+1`.
13. Vacuum embedding preserves all existing amplitudes.
14. Cached low-rank updates equal cold recomputation.
15. Wick-evaluated certificates on the covariance layer match dense Fock
    references: occupations, parity, Pauli/Gram, `|S_ABC|²`, `<J²>`, and
    `Var(J²)`.
16. A single-mode spin-1/2 Slater fixture is an exact `J²` eigenstate with
    zero variance; a generic Slater fixture with `<J²>=3/4` reports nonzero
    variance and fails the sharp certificate.
17. Purity `‖Γ²−Γ‖` is preserved across evolution and across the mean-field
    loop.
18. An open-segment winding fixture yields the same integer under
    matched-reference and boundary-register closure, and returns unknown
    without a declared closure.
19. A defective (non-semisimple) pencil fixture reports distinct algebraic
    and geometric multiplicities.
20. An overlapping-interface two-component fixture is exact under its
    declared labeled-sum option and demonstrates that a naive internal
    direct sum miscounts.

### 20.2 Property tests

- random vertex relabeling changes no hierarchy, particle verdict, closed
  holonomy/center read, or normalized exchange character;
- random orientation-preserving retriangulation changes amplitudes only by the
  measured Gram/coarse-response defect;
- all accepted bands remain accepted under in-band basis rotations;
- thresholds near a closing gap return “uncertified,” not a discontinuous particle
  label;
- adding a disconnected vacuum component does not change prior observables;
- matched positive/negative Krein sectors retain their inertia through accepted
  transport, while non-normal fixtures report biorthogonal conditioning; and
- a certified conjugate quark-antiquark creation homotopy preserves total parity
  and determinant winding; if rank/gap closes, baryon flux becomes unknown.

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
2. run unforced joint stationarity with geometry-only refinement;
3. build and persist the recursive component hierarchy;
4. maintain the exact covariance-layer state for the quasi-free sector,
   with certified Fock sectors only for oracle tests or explicitly
   non-Gaussian boundary data;
5. report all quark, gauge, exchange, and baryon certificates;
6. distinguish “no baryon,” “baryon candidate,” “certified proton,” and
   “quasi-free sharp-spin obstruction” without a target-dependent code
   path;
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
- static/shifted response reduction and AMLS fixtures;
- exterior algebra, second quantization, and graded tensor primitives.

### Wave 1 — intrinsic fibers

- label-free persistent modularity;
- localized spectral-band/projector tracking;
- exact triangle color algebra and rank-three triangle anchoring.

### Wave 2 — transport and statistics

- derived `U(r)` transport, determinant/projective rank-three sectors, and Wilson
  loops;
- Berry-cancelled exchange holonomy, structural permutation parity, and normalized
  physical `2π` loop/spin-lift certificate;
- lazy inductive-limit Fock engine;
- quasi-free covariance layer with Wick certificate evaluation,
  cross-validated against the Fock engine.

### Wave 3 — particles and interactions

- quark/antiquark classification, baryon/charge/flavor flux reads;
- even color-octet/gluon sector;
- three-quark singlet and proton total-space readout.

### Wave 4 — unforced complete simulation

- optimizer/refinement integration;
- multiscale continuum and spectral-dimension validation, including the
  covariance-only dichotomy experiment and the stationarity-defect
  correlation measurement;
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
