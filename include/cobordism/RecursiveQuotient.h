// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_RECURSIVEQUOTIENT_H
#define TESSERA_COBORDISM_RECURSIVEQUOTIENT_H

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <complex>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "cobordism/Certificate.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime { class Spacetime; }
namespace tessera::cobordism {
using namespace ::tessera::spacetime;

class AnalyticCache;

/// The declared treatment of the labeled-sum embedding Gram matrix
/// (design spec section 5.8 / 6.4). Every run proceeds by EXACTLY ONE
/// declared option; the implementation never assumes the geometric images of
/// the retained fibers are independent inside the chain space.
enum class FiberEmbeddingPolicy {
  /// Carry \f$ G = J^\dagger W J \f$ exactly in every subsequent formula.
  CarryGramExactly,
  /// Certify \f$ \|G - I\| \le \varepsilon \f$ and propagate
  /// \f$ \varepsilon \f$ through the composable amplitude budget.
  CertifiedNearIsometry,
  /// Quotient \f$ \ker G \f$ and restate the retained ranks.
  QuotientKernel,
};

/// Why a retained coordinate was kept instead of eliminated (design spec
/// section 10 step 8: harmonic, resonant, and selected interior coordinates
/// become explicit stalk/fiber coordinates — never silently deleted).
enum class RetainedCoordinateKind {
  /// An interface cell (always retained; the \f$ B \f$ block).
  Interface,
  /// An interior kernel mode of \f$ L_{II} \f$ (a topological/harmonic zero
  /// mode at \f$ \lambda = 0 \f$).
  Harmonic,
  /// An interior kernel mode of \f$ L_{II} - \lambda I \f$ at a declared
  /// resonance \f$ \lambda \neq 0 \f$.
  Resonant,
  /// A caller-selected interior cell coordinate.
  Selected,
};

/// # RecursiveQuotient
///
/// Recursive static and shifted response reduction of a (Hodge) operator
/// over a declared cell partition (epic #763, ticket #768; design spec
/// sections 5.3, 6.4, 10; whitepaper "A component is an exact static
/// response vertex" and "The master recursive construction").
///
/// ## Exact identities and their domains
///
/// Cells split into interface cells \f$ B \f$ and per-component interior
/// cells \f$ I = \sqcup_v I_v \f$, blocking the operator as
/// \f$ L = \begin{pmatrix} L_{BB} & L_{BI} \\ L_{IB} & L_{II} \end{pmatrix} \f$
/// with \f$ L_{II} \f$ block-diagonal over components by construction (an
/// interior cell couples only within its own component).
///
///  - **Static (\f$ \lambda = 0 \f$).** The exact supported static response
///    \f[ L_{\text{eff}} = L_{BB} - L_{BI} L_{II}^{+} L_{IB}, \f]
///    evaluated by sparse/rank-revealing FACTOR SOLVES of
///    \f$ L_{II} X = L_{IB} \f$ — the inverse/pseudoinverse is never formed.
///    In the **positive self-adjoint** regime this is the exact interior
///    minimization: for every compatible interface probe \f$ b \f$,
///    \f$ \min_{x_I} [b;x_I]^\dagger L [b;x_I] = b^\dagger L_{\text{eff}} b \f$
///    with minimizer \f$ x_I^* = -L_{II}^{+} L_{IB} b \f$. In the
///    **Hermitian-indefinite** regime the same equation is a STATIONARITY
///    condition, not a minimum. In the **non-normal** regime it is certified
///    block elimination, and solvability requires the compatibility
///    condition \f$ L_{IB} b \perp \ker L_{II}^{\dagger} \f$ (the left
///    kernel). Interior kernels are never regularized away: kernel modes are
///    RETAINED as explicit stalk coordinates and only the supported
///    complement is eliminated.
///  - **Shifted / Feshbach--Schur (band window).** For a spectral parameter
///    \f$ \lambda \f$ with \f$ L_{II} - \lambda I \f$ invertible,
///    \f[ F_B(\lambda) = L_{BB} - \lambda I -
///        L_{BI} (L_{II} - \lambda I)^{-1} L_{IB}, \f]
///    with the exact determinant factorization
///    \f$ \det(L - \lambda I) = \det(L_{II} - \lambda I)\det F_B(\lambda) \f$.
///    Hence \f$ \lambda \in \operatorname{spec} L \iff 0 \in
///    \operatorname{spec} F_B(\lambda) \f$ AWAY from the interior spectrum.
///    The order of the zero of \f$ \det F_B(\cdot) \f$ at \f$ \lambda \f$ is
///    the ALGEBRAIC multiplicity of \f$ \lambda \f$ in \f$ L \f$ (plus the
///    interior contribution when interior eigenvalues fall inside the
///    counting contour — reported separately, never conflated), while
///    \f$ \dim\ker F_B(\lambda) \f$ is the GEOMETRIC multiplicity; the two
///    agree only in the self-adjoint / semisimple setting (`multiplicity`
///    reports both, honestly). At an interior resonance the solve is
///    replaced only after the compatibility check
///    \f$ L_{IB} b \perp \ker (L_{II} - \lambda I)^{\dagger} \f$ and the
///    resonant interior modes are retained explicitly. The PLAIN static
///    Schur complement does NOT preserve the nonzero spectrum, and no
///    nonzero-spectrum claim is ever attached to a static reduction
///    (`Certificate::domain()` distinguishes `Static` from `BandWindow`).
///  - **Craig--Bampton / AMLS surrogate.** When a reusable LINEAR reduced
///    eigenproblem is needed over a declared frequency window, the basis of
///    interface constraint modes \f$ \Psi = -L_{II}^{+} L_{IB} \f$ plus
///    per-component fixed-interface modes below a declared cutoff gives the
///    reduced Hermitian pencil \f$ (V^\dagger L V,\ V^\dagger V) \f$. This
///    is a CERTIFIED APPROXIMATION — its certificate reports the declared
///    window, the discarded-mode gap, and the fine-space eigenresiduals of
///    the reduced pairs; it is refused outright in the non-normal regime (a
///    self-adjoint solver is never applied to a non-self-adjoint operator).
///  - **Labeled retained-fiber sum.** The next-level one-particle space is
///    the ABSTRACT labeled sum \f$ \boxplus_v E_v \f$ of the retained fibers
///    (per component: its interface cells plus its retained interior
///    modes), with the explicit embedding \f$ J \f$ into the chain space and
///    Gram matrix \f$ G = J^\dagger W J \f$. Adjacent fibers may overlap on
///    shared interface cells, so an internal direct sum is NEVER asserted;
///    each run proceeds by exactly one declared `FiberEmbeddingPolicy`.
///  - **Response network / sheaf realization.** The next level is an
///    operator-valued response network: vertices carry the retained fibers,
///    links carry the effective blocks of the reduced operator. A cellular
///    sheaf (or, when every stalk is one-dimensional, simplicial) realization
///    is emitted ONLY when explicit restriction maps REPRODUCE the blocks to
///    the declared tolerance; otherwise the general network is retained and
///    the realization certificate reports `holds() == false` — restriction
///    maps are never invented.
///
/// ## Metric regimes
///
/// The operator travels with the diagonal chain-space metric \f$ W \f$ it
/// is self-adjoint against (identity unless stated). The regime on every
/// certificate is detected against that metric:
///
///  - `PositiveSemidefinite` — \f$ WL \f$ Hermitian, \f$ W > 0 \f$, and
///    \f$ WL \succeq 0 \f$ (structural for the \f$ k = 0 \f$ graph
///    Laplacian; verified below the dense crossover otherwise). Energy
///    \f$ x^\dagger W L x \f$ is minimized.
///  - `HermitianIndefinite` — \f$ WL \f$ Hermitian but \f$ W \f$ signed or
///    \f$ WL \f$ indefinite (the real signed-weight d'Alembertian on real
///    \f$ \ell^2 \f$). The interior equation is a stationarity condition.
///  - `NonNormal` — everything else (complex weights / complex
///    \f$ \ell^2 \f$). Certified block elimination with the left-kernel
///    compatibility check; no variational claim.
///
/// The spacetime path takes `HodgeLaplacian::laplacian(degree)` exactly as
/// built — the Hermitian \f$ k = 0 \f$ graph Laplacian (metric = identity)
/// or the signed-weight d'Alembertian at \f$ k \ge 1 \f$ (metric = the
/// signed `HodgeLaplacian::weights(k)`); there is no Euclidean switch.
///
/// ## Partitions
///
/// Components may come from the discovered `PersistentModularity` partition
/// (vertex supports over the one-skeleton; a \f$ k \f$-cell belongs to a
/// component when ALL its vertices lie in the support) or from an explicit
/// caller-supplied cell partition. Component supports may OVERLAP: a cell
/// claimed by more than one component is automatically an interface cell. A
/// cell is interior to component \f$ v \f$ exactly when it is claimed only
/// by \f$ v \f$ and every nonzero coupling row/column of the operator stays
/// inside \f$ v \f$'s cells; every other cell is interface. Cell membership
/// is matched by vertex SET — no vertex order is ever imposed, and a global
/// relabeling yields an isomorphic reduction.
///
/// ## Interior nullspaces
///
/// On a spacetime-backed instance the TOPOLOGICAL interior zero modes are
/// computed exactly over the integers from the boundary maps: the kernel of
/// the stacked integer matrix
/// \f$ [\partial_k[:,I_v];\ \partial_{k+1}[I_v,:]^{\top}] \f$ (fraction-free
/// elimination; overflow fails loudly rather than approximating). This is
/// the metric-independent (combinatorial) statement; the NUMERICAL kernel of
/// the weighted block — which gates solvability and the pseudoinverse — is
/// computed by rank-revealing factorization and cross-checked against the
/// integer count where both apply. Solvability of an interface load requires
/// orthogonality to the appropriate kernel: \f$ \ker L_{II} \f$ itself in
/// the (semi)definite Hermitian regimes, the LEFT kernel
/// \f$ \ker L_{II}^\dagger \f$ in the non-normal regime.
///
/// ## Caching and nesting
///
/// Per-component static contributions are cached in the shared #764
/// `AnalyticCache` keyed by the component's cell vertex-id set, so an
/// accepted local move (published as a `TouchedStar`) invalidates ONLY the
/// touched component and its ancestry — disjoint siblings are served from
/// cache and cached results equal cold recomputation. `nextLevel` reduces
/// the reduced operator again (parent/child lineage is carried per
/// coordinate); nested reduction equals one-shot reduction whenever the
/// elimination order is valid (the Schur quotient property). Shifted
/// factorizations are memoized per spectral parameter within an instance.
///
/// Nothing in this class enters the emergence objective: it is a read-only
/// reduction of an already-relaxed operator.
class RecursiveQuotient {
  public:
    /// Reduction options. All tolerances are RELATIVE (scale-free).
    struct Options {
      Options();  // out-of-line so Options() can be an in-class default arg

      /// Certificate tolerance for `holds()` on the produced certificates.
      double tolerance{1e-10};
      /// Relative rank-revealing threshold for kernel/rank decisions.
      double rankTolerance{1e-9};
      /// Dimension at and above which dense kernels refuse (the
      /// `DenseReference` convention). Per-component interior blocks below
      /// it may use dense rank-revealing (complete orthogonal) solves; at or
      /// above it only the sparse paths run.
      int denseCrossover{512};
      /// The declared labeled-sum Gram treatment for this run.
      FiberEmbeddingPolicy embeddingPolicy{FiberEmbeddingPolicy::CarryGramExactly};
      /// \f$ \varepsilon \f$ for `CertifiedNearIsometry`.
      double nearIsometryEpsilon{1e-10};
      /// Caller-selected interior cells to RETAIN as explicit stalk
      /// coordinates instead of eliminating (matrix path: fine indices).
      std::vector<int> selectedInteriorIndices{};
      /// Caller-selected interior cells for the spacetime path, as vertex-id
      /// tuples (matched by vertex set).
      std::vector<std::vector<std::uint64_t>> selectedInteriorCells{};
    };

    /// One retained stalk/fiber coordinate of the reduced space, with its
    /// provenance (never silently deleted; design spec section 10 step 8).
    struct RetainedCoordinate {
      RetainedCoordinateKind kind{RetainedCoordinateKind::Interface};
      /// Owning component (every retained interior mode has one; an
      /// interface cell may be shared — this is the FIRST claiming
      /// component; all claimants are in `LabeledFiberSumRead`).
      int component{0};
      /// Fine-space index for `Interface`/`Selected` coordinates; -1 for
      /// mode coordinates (`Harmonic`/`Resonant`).
      int fineIndex{-1};
      /// The fine-space column vector this coordinate embeds to (length =
      /// fine dimension; an indicator for cell coordinates, the kernel-mode
      /// vector for mode coordinates).
      std::vector<std::complex<double>> embedding{};
      /// Human-readable provenance, e.g. "cell(3,7)", "harmonic[c1#0]",
      /// "resonant[c0#1@(2.5,0)]"; nested levels prefix "L<level>:".
      std::string provenance{};
    };

    /// Interior nullspace of one component (topological + numerical).
    struct InteriorNullspaceRead {
      int component{0};
      /// dim ker of the weighted interior block (numerical, at
      /// `rankTolerance`).
      std::size_t nullity{0};
      /// Exact integer topological zero-mode count (spacetime path;
      /// combinatorial kernel of the stacked boundary blocks). Equals
      /// `integerBasis.size()`. 0 on the matrix path.
      std::size_t integerNullity{0};
      /// Exact integer basis vectors over the component's interior cells
      /// (spacetime path; each of length `interiorCells(component).size()`).
      std::vector<std::vector<long>> integerBasis{};
      /// Numerical right-kernel basis, flat row-major (|I_v| x nullity).
      std::vector<std::complex<double>> kernelBasis{};
      /// Numerical LEFT-kernel basis of \f$ L_{II}^\dagger \f$, flat
      /// row-major (|I_v| x leftNullity). Equals the right kernel in the
      /// Hermitian regimes.
      std::vector<std::complex<double>> leftKernelBasis{};
      /// Measured \f$ \|L_{II} Z\| / \|L_{II}\| \f$ over the returned basis.
      Certificate certificate{};
    };

    /// The static reduction read: the effective operator over
    /// interface + retained coordinates, with per-coordinate provenance.
    struct StaticReductionRead {
      /// Fine indices of the kept cells (interface + selected), ascending
      /// (the canonical reduced-coordinate order is: kept cells ascending,
      /// then retained mode coordinates in component order).
      std::vector<int> interfaceIndices{};
      /// All reduced coordinates in order (size = reduced dimension).
      std::vector<RetainedCoordinate> coordinates{};
      /// The reduced operator, flat row-major (reducedDim x reducedDim). Its
      /// leading interface block is \f$ L_{BB} - L_{BI} L_{II}^{+} L_{IB} \f$.
      std::vector<std::complex<double>> effectiveOperator{};
      /// Max relative interior solve residual
      /// \f$ \|L_{II}X - L_{IB}\| / \|L_{IB}\| \f$ across components.
      double solveResidual{0.0};
      /// Max compatibility violation \f$ \|Y^\dagger L_{IB}\| / \|L_{IB}\| \f$
      /// over interior (left-)kernels — 0 when every load is compatible.
      double compatibilityResidual{0.0};
      /// Static-domain certificate in the detected regime.
      Certificate certificate{};
    };

    /// One evaluation of the exact Feshbach--Schur response pencil.
    struct FeshbachRead {
      std::complex<double> lambda{};
      /// Declared band window (caller-supplied; plain frequencies).
      double windowLower{0.0};
      double windowUpper{0.0};
      /// \f$ F_B(\lambda) \f$ over the kept cells (interface + selected)
      /// plus any resonant-retained modes, flat row-major.
      std::vector<std::complex<double>> response{};
      /// The coordinates of `response` (kept cells first, then any
      /// retained resonant modes).
      std::vector<RetainedCoordinate> coordinates{};
      /// Whether \f$ \lambda \f$ resonates with the interior spectrum (a
      /// rank-deficient shifted block was met and its kernel retained).
      bool resonant{false};
      /// Max relative shifted solve residual across components.
      double solveResidual{0.0};
      /// Max resonant compatibility violation (left-kernel test), 0 when
      /// not resonant or compatible.
      double compatibilityResidual{0.0};
      /// Relative determinant-factorization residual
      /// \f$ |\det(L-\lambda) - \det(L_{II}-\lambda)\det F_B(\lambda)| \f$
      /// (scale-normalized), measured below the dense crossover; NaN above.
      double determinantResidual{0.0};
      Certificate certificate{};
    };

    /// Honest multiplicity report at a candidate eigenvalue (band domain).
    struct MultiplicityRead {
      std::complex<double> lambda{};
      double contourRadius{0.0};
      int nodes{0};
      /// Winding of \f$ \det F_B \f$ around the contour (zeros minus poles
      /// of the pencil determinant inside).
      int responseWinding{0};
      /// Winding of \f$ \det(L_{II} - z) \f$ around the contour (the
      /// interior-spectrum contribution inside the contour — reported
      /// separately, never conflated with the response winding).
      int interiorWinding{0};
      /// Algebraic multiplicity of the spectrum of \f$ L \f$ inside the
      /// contour: `responseWinding + interiorWinding` (exact determinant
      /// factorization).
      int algebraic{0};
      /// \f$ \dim\ker F_B(\lambda) \f$ at `rankTolerance`.
      int geometric{0};
      /// Whether algebraic == geometric (guaranteed only in the
      /// self-adjoint / semisimple setting).
      bool semisimple{false};
      /// Max per-step phase advance / pi over both unwrapped determinant
      /// phases (must stay well below 1 for an alias-free winding).
      double phaseStepMargin{0.0};
      Certificate certificate{};
    };

    /// Craig--Bampton / AMLS retained-mode surrogate over a declared window.
    struct CraigBamptonRead {
      double windowLower{0.0};
      double windowUpper{0.0};
      /// Fixed-interface eigenvalue cutoff used for mode retention.
      double modeCutoff{0.0};
      /// Retained fixed-interface mode count per component.
      std::vector<int> retainedModes{};
      /// Reduction basis V, flat row-major (fineDim x reducedDim): interface
      /// unit block + constraint modes, then fixed-interface modes.
      std::vector<std::complex<double>> basis{};
      /// Reduced stiffness \f$ V^\dagger W L V \f$, flat row-major
      /// (\f$ V^\dagger L V \f$ under the identity metric).
      std::vector<std::complex<double>> reducedStiffness{};
      /// Reduced mass \f$ V^\dagger W V \f$, flat row-major (Hermitian
      /// positive definite — the reusable LINEAR eigenproblem is
      /// \f$ K y = \lambda M y \f$).
      std::vector<std::complex<double>> reducedMass{};
      /// Smallest DISCARDED fixed-interface eigenvalue minus `windowUpper`
      /// (the discarded-mode gap; +inf when nothing was discarded).
      double discardedModeGap{0.0};
      /// Reduced eigenvalues inside the window, ascending.
      std::vector<double> windowEigenvalues{};
      /// Fine-space relative eigenresiduals
      /// \f$ \|L V y - \lambda V y\| / (\|L\|\,\|V y\|) \f$, one per
      /// window eigenvalue.
      std::vector<double> eigenResiduals{};
      Certificate certificate{};
    };

    /// The abstract labeled sum \f$ \boxplus_v E_v \f$ with embedding and
    /// Gram data (design spec section 6.4).
    struct LabeledFiberSumRead {
      /// Component index of each summand block, in embedding column order.
      std::vector<int> summandComponents{};
      /// Nominal rank \f$ r_v \f$ of each summand.
      std::vector<int> summandRanks{};
      /// The embedding \f$ J \f$ into the fine chain space, flat row-major
      /// (fineDim x totalRank), columns |W|-unit-normalized.
      std::vector<std::complex<double>> embedding{};
      /// \f$ G = J^\dagger W J \f$, flat row-major (totalRank x totalRank).
      std::vector<std::complex<double>> gram{};
      /// The declared policy this run proceeds by.
      FiberEmbeddingPolicy policy{FiberEmbeddingPolicy::CarryGramExactly};
      /// \f$ \|G - I\|_2 \f$.
      double gramDefect{0.0};
      /// \f$ \dim\ker G \f$ at `rankTolerance` (the labeled-sum
      /// overcounting; 0 exactly when the internal sum happens to be
      /// direct).
      std::size_t quotientNullity{0};
      /// Total rank of the labeled sum: \f$ \sum_v r_v \f$ nominal.
      std::size_t nominalRank{0};
      /// Effective rank after the declared treatment:
      /// nominal for `CarryGramExactly`/`CertifiedNearIsometry`,
      /// \f$ \operatorname{rank} G \f$ for `QuotientKernel`.
      std::size_t effectiveRank{0};
      /// Orthonormal basis of \f$ (\ker G)^\perp \f$, flat row-major
      /// (totalRank x effectiveRank), populated under `QuotientKernel`.
      std::vector<std::complex<double>> quotientBasis{};
      Certificate certificate{};
    };

    /// One operator-valued link of the next-level response network.
    struct ResponseEdge {
      int from{0};
      int to{0};
      /// The effective block between the two stalks, flat row-major
      /// (stalkDim(from) x stalkDim(to)).
      std::vector<std::complex<double>> block{};
    };

    /// The next-level operator-valued response network.
    struct ResponseNetworkRead {
      /// Stalk dimension per component (interface cells claimed + retained
      /// interior modes owned).
      std::vector<int> stalkDimensions{};
      /// Reduced-coordinate indices of each stalk (shared interface cells
      /// appear in EVERY claiming stalk — the network never asserts an
      /// internal direct sum; `LabeledFiberSumRead` carries the Gram data).
      std::vector<std::vector<int>> stalkCoordinates{};
      /// Diagonal blocks (one per component), flat row-major.
      std::vector<std::vector<std::complex<double>>> vertexBlocks{};
      /// Off-diagonal links (only nonzero or stalk-sharing pairs).
      std::vector<ResponseEdge> edges{};
      /// Largest |entry| of the reduced operator not covered by any
      /// vertex/edge block (0 = the network reproduces the operator).
      double coverageResidual{0.0};
      Certificate certificate{};
    };

    /// A cellular-sheaf (or simplicial) realization attempt of the response
    /// network. Emitted ONLY when the restriction maps REPRODUCE the blocks.
    struct SheafRealizationRead {
      /// Whether a certified realization was emitted (false = the general
      /// response network is retained; maps below are empty).
      bool emitted{false};
      /// Whether every stalk is one-dimensional (a weighted simplicial
      /// 1-complex realization).
      bool simplicial{false};
      /// Edge stalk dimension per network edge.
      std::vector<int> edgeStalkDimensions{};
      /// Restriction maps per network edge: for edge e = (u, v), the pair
      /// \f$ \rho_{u\to e} \f$ (edgeDim x stalkDim(u)) then
      /// \f$ \rho_{v\to e} \f$ (edgeDim x stalkDim(v)), flat row-major.
      std::vector<std::vector<std::complex<double>>> restrictionMaps{};
      /// Max relative block-reconstruction residual of the sheaf Laplacian
      /// against the response network blocks.
      double reconstructionResidual{0.0};
      Certificate certificate{};
    };

    /// Build over an explicit operator (fixtures and next-level recursion —
    /// after one elimination the coarse object is generally a response
    /// network, not a simplicial complex). `op` is flat row-major
    /// (`dim` x `dim`); `weights` is the diagonal chain-space metric
    /// \f$ W \f$ (empty = identity); `components` are 0-based fine index
    /// sets, possibly overlapping, whose union must cover every index.
    /// @throws std::invalid_argument on malformed sizes/partition.
    [[nodiscard]] static RecursiveQuotient overMatrix(
        const std::vector<std::complex<double>> &op, int dim,
        const std::vector<std::complex<double>> &weights,
        const std::vector<std::vector<int>> &components,
        const Options &options = Options());

    /// Build over a spacetime's Hodge operator at `degree`, with components
    /// given as explicit k-cell sets (each cell a vertex-id tuple, matched
    /// by vertex SET). The operator is `HodgeLaplacian::laplacian(degree)`
    /// as built (see "Metric regimes" above). An `AnalyticCache` bound to
    /// the same spacetime enables per-component reuse across accepted moves.
    /// @throws std::invalid_argument on an unknown cell or uncovered cells.
    [[nodiscard]] static RecursiveQuotient overCells(
        std::shared_ptr<Spacetime> st, int degree,
        const std::vector<std::vector<std::vector<std::uint64_t>>> &componentCells,
        const Options &options = Options(),
        std::shared_ptr<AnalyticCache> cache = nullptr);

    /// Build over a spacetime's Hodge operator at `degree`, with components
    /// given as vertex supports (the `PersistentModularity` component
    /// support convention): a k-cell belongs to a component when ALL its
    /// vertices lie in the support; cells claimed by no support are
    /// gathered into one residual component appended after the supplied
    /// ones.
    [[nodiscard]] static RecursiveQuotient overVertexSupports(
        std::shared_ptr<Spacetime> st, int degree,
        const std::vector<std::vector<std::uint64_t>> &componentVertexSupports,
        const Options &options = Options(),
        std::shared_ptr<AnalyticCache> cache = nullptr);

    /// Fine dimension (number of k-cells / coordinates at this level).
    [[nodiscard]] int dimension() const noexcept { return dim_; }
    /// Number of components.
    [[nodiscard]] int componentCount() const noexcept {
      return static_cast<int>(components_.size());
    }
    /// Hodge degree (spacetime paths; -1 on the matrix path).
    [[nodiscard]] int degree() const noexcept { return degree_; }
    /// Nesting level: 0 for a base instance, parent level + 1 under
    /// `nextLevel`.
    [[nodiscard]] int level() const noexcept { return level_; }
    /// The certificate regime detected for the operator (see
    /// `CertificateRegime`).
    [[nodiscard]] CertificateRegime regime() const noexcept { return regime_; }

    /// Ascending fine indices of the KEPT cell coordinates: the interface
    /// cells \f$ B \f$ plus any caller-selected retained interior cells
    /// (which are never eliminated; `StaticReductionRead::coordinates`
    /// distinguishes the kinds).
    [[nodiscard]] const std::vector<int> &interfaceIndices() const noexcept {
      return interfaceIndices_;
    }
    /// Fine indices of component `component`'s interior cells, ascending.
    /// @throws std::out_of_range on a bad component index.
    [[nodiscard]] const std::vector<int> &interiorIndices(int component) const;

    /// Provenance of each fine coordinate at this level (cell vertex tuples
    /// on the spacetime path, inherited reduced-coordinate provenance under
    /// `nextLevel`).
    [[nodiscard]] const std::vector<std::string> &coordinateProvenance()
        const noexcept {
      return provenance_;
    }

    /// The interior nullspace of one component (integer topological basis
    /// on the spacetime path, numerical kernel and left kernel always).
    /// @throws std::out_of_range on a bad component index.
    [[nodiscard]] InteriorNullspaceRead interiorNullspace(int component) const;

    /// The exact supported static reduction (computed once and memoized;
    /// per-component contributions are served from the bound
    /// `AnalyticCache` when fresh).
    [[nodiscard]] const StaticReductionRead &staticReduction() const;

    /// Verify the regime-appropriate static certificate on one kept-cell
    /// probe `b` (length = `interfaceIndices().size()`): minimized fine
    /// energy \f$ x^\dagger WLx \f$ vs \f$ b^\dagger (WL_{\text{eff}}) b \f$
    /// in the positive self-adjoint regime, interior stationarity in the
    /// Hermitian-indefinite regime, certified block elimination + the
    /// left-kernel compatibility check in the non-normal regime. Retained
    /// mode coordinates are held at zero.
    /// @throws std::invalid_argument on size mismatch.
    [[nodiscard]] Certificate staticProbeCertificate(
        const std::vector<std::complex<double>> &probe) const;

    /// Run `staticProbeCertificate` over the deterministic probe set (every
    /// interface basis vector and the all-ones vector) and return the worst
    /// certificate.
    [[nodiscard]] Certificate verifyStatic() const;

    /// Evaluate the exact Feshbach--Schur response \f$ F_B(\lambda) \f$ over
    /// a caller-supplied window (plain lower/upper frequencies — band
    /// SELECTION is out of scope here). Shifted factorizations are memoized
    /// per \f$ \lambda \f$. @throws std::invalid_argument when
    /// `windowLower > windowUpper`.
    [[nodiscard]] FeshbachRead feshbach(std::complex<double> lambda,
                                        double windowLower,
                                        double windowUpper) const;

    /// Honest multiplicity report at `lambda`: algebraic multiplicity from
    /// the winding of the unwrapped determinant phases of
    /// \f$ \det F_B(\cdot) \f$ and \f$ \det(L_{II} - \cdot) \f$ around the
    /// circle of `radius`, geometric multiplicity from
    /// \f$ \dim\ker F_B(\lambda) \f$. The winding is validated by doubling
    /// the node count until stable. @throws std::invalid_argument on a
    /// non-positive radius or node count < 8.
    [[nodiscard]] MultiplicityRead multiplicity(std::complex<double> lambda,
                                                double radius,
                                                int nodes = 64) const;

    /// Craig--Bampton retained-mode basis over the declared window: retain
    /// per-component fixed-interface modes with eigenvalue <= `modeCutoff`
    /// (must be >= `windowUpper`). Hermitian regimes with a positive chain
    /// metric only. This is a CERTIFIED APPROXIMATION: `residualTolerance`
    /// is the caller-declared acceptance residual its certificate holds
    /// against (a negative value selects the strict `Options::tolerance`,
    /// under which a genuinely truncated surrogate honestly reports
    /// `holds() == false` while still carrying its window, gap, and
    /// residuals).
    /// @throws std::invalid_argument in the non-normal regime, on an
    ///   indefinite metric, a bad window, or `modeCutoff < windowUpper`;
    ///   std::length_error when a component's interior block is at/above
    ///   the dense crossover (the dense fixed-interface eigensolve refuses
    ///   at scale).
    [[nodiscard]] CraigBamptonRead craigBampton(
        double windowLower, double windowUpper, double modeCutoff,
        double residualTolerance = -1.0) const;

    /// The abstract labeled retained-fiber sum with embedding and Gram data,
    /// treated by the run's declared `FiberEmbeddingPolicy`.
    [[nodiscard]] LabeledFiberSumRead labeledFiberSum() const;

    /// The composable amplitude budget of the `CertifiedNearIsometry`
    /// policy: two embeddings with Gram defects $ arepsilon_A,
    /// arepsilon_B $ compose (tensor) to at most
    /// [ arepsilon_{AB} \le arepsilon_A + arepsilon_B +
    ///     arepsilon_Aarepsilon_B , ]
    /// and the amplitude error obeys
    /// $ |a^\dagger G b - a^\dagger b| \le arepsilon\|a\|\|b\| $
    /// (whitepaper, "Interactions and the expanding Hilbert space"). This is
    /// how a certified $ arepsilon $ PROPAGATES to composite reads.
    [[nodiscard]] static double composeNearIsometryBudget(
        double epsilonA, double epsilonB) noexcept {
      return epsilonA + epsilonB + epsilonA * epsilonB;
    }

    /// The next-level operator-valued response network (component stalks +
    /// effective blocks of the static reduction).
    [[nodiscard]] ResponseNetworkRead responseNetwork() const;

    /// Attempt the cellular-sheaf / simplicial realization of the response
    /// network. `emitted == false` (with the failing residual on the
    /// certificate) when the blocks are NOT reproduced — the general network
    /// is then retained; nothing is invented. Hermitian regimes only (a
    /// sheaf Laplacian is self-adjoint); the non-normal regime always
    /// refuses.
    [[nodiscard]] SheafRealizationRead sheafRealization() const;

    /// Reduce again: a child quotient over this level's reduced operator
    /// (matrix path), with `components` indexing the REDUCED coordinates.
    /// The child inherits provenance ("L<level>:" prefixes), level + 1, and
    /// this level's chain metric restricted through the reduced coordinates.
    [[nodiscard]] RecursiveQuotient nextLevel(
        const std::vector<std::vector<int>> &components,
        const Options &options) const;

    /// `nextLevel` with this instance's options.
    [[nodiscard]] RecursiveQuotient nextLevel(
        const std::vector<std::vector<int>> &components) const;

    /// Drop memoized reductions/factorizations and, on the spacetime path,
    /// re-read the operator values for the SAME cell complex (metric moves;
    /// a structural move needs a fresh instance). The bound `AnalyticCache`
    /// still gates per-component reuse: after an accepted move is published
    /// there, the next `staticReduction` recomputes ONLY the invalidated
    /// components.
    void invalidate();

    /// The options this instance runs with.
    [[nodiscard]] const Options &options() const noexcept { return options_; }

  private:
    struct ComponentSolve;  // per-component factorization + kernel payload

    RecursiveQuotient() = default;

    void initMatrix(const std::vector<std::complex<double>> &op, int dim,
                    const std::vector<std::complex<double>> &weights,
                    const std::vector<std::vector<int>> &components,
                    const Options &options);
    void classify();
    void detectRegime(bool structuralPsd);
    [[nodiscard]] std::shared_ptr<ComponentSolve> componentSolve(
        int component) const;
    [[nodiscard]] std::shared_ptr<ComponentSolve> computeSolve(
        int component, std::complex<double> lambda) const;
    [[nodiscard]] const std::vector<std::shared_ptr<ComponentSolve>> &
    shiftedSolves(std::complex<double> lambda) const;
    [[nodiscard]] std::vector<std::complex<double>> contourDeterminants(
        std::complex<double> lambda, double radius, int nodes,
        std::vector<std::complex<double>> &interiorDets) const;
    [[nodiscard]] static int windingFromPhases(
        const std::vector<std::complex<double>> &values, double *maxStep);
    [[nodiscard]] std::vector<std::uint64_t> componentVertexIds(
        int component) const;
    [[nodiscard]] std::vector<long> integerKernelStack(int component,
                                                       int *rows) const;

    // --- problem data (op_/weights_ refresh under invalidate()) ------------
    Eigen::SparseMatrix<std::complex<double>> op_{};
    Eigen::VectorXcd weights_{};        // diagonal chain metric W
    double opNorm_{0.0};                // scale for relative residuals
    int dim_{0};
    int degree_{-1};
    int level_{0};
    CertificateRegime regime_{CertificateRegime::NonNormal};
    Options options_{};
    std::vector<std::vector<int>> components_{};       // claimed cells
    std::vector<std::vector<int>> interior_{};         // per component
    std::vector<int> interfaceIndices_{};              // kept cells, ascending
    std::vector<RetainedCoordinateKind> keptKinds_{};  // Interface/Selected
    std::vector<int> keptOwner_{};                     // first claimant
    std::vector<int> interfacePosition_{};             // fine -> kept position
    std::vector<std::vector<int>> claimants_{};        // fine -> components
    std::vector<std::string> provenance_{};            // fine coordinates
    std::uint64_t partitionFingerprint_{0};            // cache-kind qualifier
    std::shared_ptr<Spacetime> st_{};
    std::shared_ptr<AnalyticCache> cache_{};
    // spacetime path extras: per-cell vertex tuples + integer boundary maps
    std::vector<std::vector<std::uint64_t>> cellVertices_{};
    bool hasBoundary_{false};
    std::vector<long> boundaryK_{};                    // ∂_degree, flat
    int boundaryKRows_{0};
    std::vector<long> boundaryK1_{};                   // ∂_{degree+1}, flat
    int boundaryK1Cols_{0};

    // --- memoized results ---------------------------------------------------
    mutable std::optional<StaticReductionRead> static_{};
    mutable std::vector<std::shared_ptr<ComponentSolve>> solves_{};
    mutable std::map<std::pair<double, double>,
                     std::vector<std::shared_ptr<ComponentSolve>>>
        shifted_{};
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_RECURSIVEQUOTIENT_H
