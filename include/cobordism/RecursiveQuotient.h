// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_RECURSIVEQUOTIENT_H
#define TESSERA_COBORDISM_RECURSIVEQUOTIENT_H

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <limits>
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

/// The declared treatment of the labeled-sum embedding Gram matrix.
/// Every run proceeds by EXACTLY ONE
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

/// How a level's operator was produced from its parent — the response step
/// \f$ \RN_{\ell+1}(\lambda) = \mathrm{Feshbach}_{P_\ell}(\RN_\ell(\lambda)) \f$
/// of the master recursion, which is a PENCIL recursion: the static
/// \f$ \lambda = 0 \f$ complement is one point of it, not the whole of it.
enum class LevelOrigin {
  /// A base instance built directly over a complex or an explicit matrix.
  Base,
  /// The exact supported static Schur complement at \f$ \lambda = 0 \f$.
  StaticResponse,
  /// The exact energy-dependent Feshbach--Schur pencil evaluated at a
  /// declared \f$ \lambda \f$ over a declared band window.
  BandPencil,
  /// A cached LINEAR Craig--Bampton/AMLS surrogate over a declared frequency
  /// window — a certified approximation, never an exact spectral identity.
  Surrogate,
};

/// Why a retained coordinate was kept instead of eliminated: harmonic,
/// resonant, and selected interior coordinates become explicit stalk/fiber
/// coordinates — never silently deleted.
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
/// over a declared cell partition (epic #763, ticket #768; whitepaper
/// "A component is an exact static response vertex" and "The master
/// recursive construction").
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
///  - **Labeled fiber sum.** The next-level one-particle space is the
///    ABSTRACT labeled sum \f$ \boxplus_v E_v \f$, with the explicit
///    embedding \f$ J \f$ into the chain space and Gram matrix
///    \f$ G = J^\dagger W J \f$. Adjacent fibers may overlap on shared
///    interface cells, so an internal direct sum is NEVER asserted; each run
///    proceeds by exactly one declared `FiberEmbeddingPolicy`. Two summand
///    readings are available and are never conflated: `labeledFiberSum` sums
///    the reduction's own RETAINED COORDINATES (interface cells plus owned
///    interior modes), which carry no band certificate, while
///    `certifiedFiberSum` sums the boxed display's \f$ E_v \f$ — CERTIFIED
///    ISOLATED BANDS supplied by the fiber layer, with each band's isolation
///    gap and certificate carried onto its summand.
///  - **Fock stage.** `fockStage` closes the boxed display's final line,
///    \f$ \HK_{\ell+1} = \Fock(\hh_{\ell+1}) \f$, at the SPECTRUM level: the
///    one-particle compression onto the labeled sum, its spectrum, and the
///    exact free many-body spectrum as occupation subset sums. The
///    \f$ 2^M \f$ space is never materialized, and the read refuses past a
///    declared term budget rather than allocating.
///  - **Pencil recursion.** The response step is
///    \f$ \RN_{\ell+1}(\lambda) = \mathrm{Feshbach}_{P_\ell}(\RN_\ell(\lambda)) \f$.
///    `nextLevel` takes the static \f$ \lambda = 0 \f$ point of it;
///    `nextLevelAtLambda` takes the exact energy-dependent pencil at a
///    declared \f$ \lambda \f$; `nextLevelFromSurrogate` takes a certified
///    linear AMLS surrogate. Every child carries its origin, declared window,
///    residuals, and producing certificate on `levelProvenance()`.
///    `childPersistentPartition` supplies \f$ P_\ell \f$ at every scale, so
///    the recursion discovers its own components rather than partitioning
///    only at level zero.
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
///    \f$ WL \succeq 0 \f$, VERIFIED by a pivoted LDLT below the dense
///    crossover. There is no structural shortcut at any degree (#805): degree
///    zero is measured like the rest, and a Lorentzian \f$ L_0 \f$ is
///    routinely indefinite. Energy \f$ x^\dagger W L x \f$ is minimized.
///  - `HermitianIndefinite` — \f$ WL \f$ Hermitian but \f$ W \f$ signed or
///    \f$ WL \f$ indefinite (the real signed-weight d'Alembertian on real
///    \f$ \ell^2 \f$). The interior equation is a stationarity condition.
///  - `NonNormal` — everything else (complex weights / complex
///    \f$ \ell^2 \f$). Certified block elimination with the left-kernel
///    compatibility check; no variational claim.
///
/// The spacetime path takes `HodgeLaplacian::laplacian(degree)` exactly as
/// built — the signed-weight d'Alembertian at EVERY degree, with metric =
/// `HodgeLaplacian::weights(degree)` (the identity at degree zero, where
/// \f$ L_0 = \partial_1 W_1^{-1}\partial_1^{\dagger} \f$); there is no
/// Euclidean switch and no degree-zero special case. The regime is MEASURED
/// from that operator at every degree — degree zero is not declared
/// `PositiveSemidefinite` from a convention, and on a Lorentzian complex it
/// routinely is not (#805).
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

    /// How THIS level was produced from its parent, with the declared window
    /// and the residuals of the producing response step carried on the child
    /// (whitepaper "The master recursive construction": a cached linear
    /// \f$ \RN_{\ell+1} \f$ is an AMLS/component-mode surrogate WITH a
    /// declared frequency window and residual — the window and the residual
    /// travel with the level, they are not left behind at the parent).
    struct LevelProvenanceRead {
      /// The response step that produced this level.
      LevelOrigin origin{LevelOrigin::Base};
      /// The spectral parameter the parent pencil was evaluated at
      /// (`BandPencil` only; NaN otherwise — never 0, which would claim a
      /// static reduction that never happened).
      std::complex<double> lambda{std::numeric_limits<double>::quiet_NaN(),
                                  std::numeric_limits<double>::quiet_NaN()};
      /// The declared band/frequency window (`BandPencil`, `Surrogate`;
      /// NaN on `Base`/`StaticResponse`, which carry no window).
      double windowLower{std::numeric_limits<double>::quiet_NaN()};
      double windowUpper{std::numeric_limits<double>::quiet_NaN()};
      /// Max relative interior solve residual of the producing step.
      double solveResidual{std::numeric_limits<double>::quiet_NaN()};
      /// Max compatibility (left-kernel) violation of the producing step.
      double compatibilityResidual{std::numeric_limits<double>::quiet_NaN()};
      /// Worst fine-space eigenresidual of the retained window pairs
      /// (`Surrogate` only; NaN otherwise).
      double surrogateResidual{std::numeric_limits<double>::quiet_NaN()};
      /// Smallest discarded fixed-interface eigenvalue minus the window
      /// upper edge (`Surrogate` only; NaN otherwise).
      double discardedModeGap{std::numeric_limits<double>::quiet_NaN()};
      /// Whether the parent \f$ \lambda \f$ resonated with the interior
      /// spectrum (the shifted block was rank-deficient and its kernel was
      /// retained explicitly).
      bool resonant{false};
      /// The producing step's own certificate, carried verbatim. A
      /// `Surrogate` level therefore travels with a CERTIFIED-APPROXIMATION
      /// certificate and can never be mistaken for an exact reduction.
      Certificate certificate{};
    };

    /// One certified isolated band handed to `certifiedFiberSum` as the
    /// summand \f$ E_v \f$ of the boxed display. This is PLAIN DATA: the
    /// producing band lives in the fiber layer, and this class never reaches
    /// into it — the caller maps a certified band onto its component's fine
    /// coordinates (cells matched by vertex SET, never by index) and hands
    /// the frame and the certificate across.
    struct CertifiedBand {
      /// The component this band belongs to.
      int component{0};
      /// The band's right frame over THIS level's fine coordinates, flat
      /// row-major (`dimension()` x `rank`). Columns spanning the band.
      std::vector<std::complex<double>> frame{};
      /// Band rank \f$ r_v \f$ (number of eigenvalues in the band).
      std::size_t rank{0};
      /// Distance to the nearest eigenvalue below / above the band — the
      /// ISOLATION the boxed display's "certified isolated subspace"
      /// requires. NaN when the side is unknown.
      double lowerGap{std::numeric_limits<double>::quiet_NaN()};
      double upperGap{std::numeric_limits<double>::quiet_NaN()};
      /// The band's frequency window [min Re, max Re].
      double frequencyLower{std::numeric_limits<double>::quiet_NaN()};
      double frequencyUpper{std::numeric_limits<double>::quiet_NaN()};
      /// Whether the band met every certification threshold of its producing
      /// configuration. An UNCERTIFIED band is still summed and reported —
      /// it is never silently dropped — but it makes the labeled sum's own
      /// certificate fail to hold.
      bool accepted{false};
      /// The band's certificate, carried verbatim onto the summand.
      Certificate certificate{};
    };

    /// The certificate data of one summand of a certified labeled sum.
    struct CertifiedFiberSummand {
      /// The component this summand came from.
      int component{0};
      /// Nominal rank of the summand.
      std::size_t rank{0};
      /// The band's isolation gaps, carried from the producing band.
      double lowerGap{std::numeric_limits<double>::quiet_NaN()};
      double upperGap{std::numeric_limits<double>::quiet_NaN()};
      /// The band's frequency window.
      double frequencyLower{std::numeric_limits<double>::quiet_NaN()};
      double frequencyUpper{std::numeric_limits<double>::quiet_NaN()};
      /// Whether the producing band was accepted.
      bool accepted{false};
      /// The producing band's certificate.
      Certificate certificate{};
    };

    /// One retained stalk/fiber coordinate of the reduced space, with its
    /// provenance (never silently deleted).
    struct RetainedCoordinate {
      /// Why this coordinate was retained.
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
      /// The component this read describes.
      int component{0};
      /// dim ker of the weighted interior block (numerical, at
      /// `rankTolerance`).
      std::size_t nullity{0};
      /// Exact integer topological zero-mode count (spacetime path;
      /// combinatorial kernel of the stacked boundary blocks). Equals
      /// `integerBasis.size()`. 0 on the matrix path — check
      /// `integerNullityMeasured` before comparing.
      std::size_t integerNullity{0};
      /// Whether the exact integer nullity was computed at all. False on the
      /// matrix path (no boundary maps) and when the integer kernel overflowed;
      /// `integerNullity == 0` then means "not measured", not "measured zero".
      bool integerNullityMeasured{false};
      /// `nullity - integerNullity` — the discrepancy between the numerical
      /// kernel of the weighted interior block and the exact integer
      /// topological nullity, RECORDED rather than silently dropped (#805).
      /// Zero means the two agree; a nonzero value means the operator's
      /// numerical kernel is not the combinatorial one (a signed/complex metric
      /// can open or close a zero mode the topology does not have, and the
      /// weighted kernel differs from the unit-weight one in general). NaN when
      /// `integerNullityMeasured` is false — never 0, which would claim an
      /// agreement that was never measured.
      double nullityDiscrepancy{std::numeric_limits<double>::quiet_NaN()};
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
      /// The spectral parameter the pencil was evaluated at.
      std::complex<double> lambda{};
      /// Declared band window (caller-supplied), lower edge.
      double windowLower{0.0};
      /// Declared band window (caller-supplied), upper edge.
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
      /// Band-window certificate in the detected regime.
      Certificate certificate{};
    };

    /// Honest multiplicity report at a candidate eigenvalue (band domain).
    struct MultiplicityRead {
      /// The candidate eigenvalue the contour is centred on.
      std::complex<double> lambda{};
      /// Radius of the counting contour.
      double contourRadius{0.0};
      /// Node count of the stabilized (doubled) evaluation.
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
      /// Certified-numerical winding certificate (stability + margin).
      Certificate certificate{};
    };

    /// Craig--Bampton / AMLS retained-mode surrogate over a declared window.
    struct CraigBamptonRead {
      /// Declared frequency window, lower edge.
      double windowLower{0.0};
      /// Declared frequency window, upper edge.
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
      /// Certified-approximation certificate against the declared residual
      /// tolerance.
      Certificate certificate{};
    };

    /// The abstract labeled sum \f$ \boxplus_v E_v \f$ with embedding and
    /// Gram data.
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
      /// Whether the summands are CERTIFIED ISOLATED BANDS (the boxed
      /// display's \f$ E_v \f$) rather than the retained-coordinate reading.
      /// False for `labeledFiberSum()`, true for `certifiedFiberSum()`.
      bool fromCertifiedBands{false};
      /// Per-summand band certificates — populated only when
      /// `fromCertifiedBands`. Empty otherwise: a retained-coordinate
      /// summand carries no band certificate, and none is invented for it.
      std::vector<CertifiedFiberSummand> summandCertificates{};
      /// The smallest isolation gap over the summed bands (the weakest link
      /// of the "certified ISOLATED subspace" claim). NaN when not summed
      /// from certified bands, or when every gap is unknown.
      double worstIsolationGap{std::numeric_limits<double>::quiet_NaN()};
      /// Whether EVERY summed band was accepted by its producing
      /// configuration. False (with the certificate failing to hold) when any
      /// summand is uncertified — the sum is still returned, honestly.
      bool allBandsAccepted{false};
      /// Certificate of the declared policy's claim.
      Certificate certificate{};
    };

    /// The Fock stage \f$ \HK_{\ell+1} = \Fock(\hh_{\ell+1}) \f$ over the
    /// labeled sum — the boxed display's final line, the expanding state
    /// space of the recursion.
    ///
    /// The many-body space is carried at the SPECTRUM level, per the
    /// exactness contract ("occupation subset sums for \f$ d\Gamma(L) \f$,
    /// not diagonalization of an eager Fock matrix"; "keep tensor products
    /// lazy"). The \f$ 2^M \f$ vector is never allocated: the free many-body
    /// spectrum is the exact set of occupation subset sums, and it REFUSES
    /// rather than allocating past the declared term budget.
    struct FockStageRead {
      /// \f$ M = \dim\hh_{\ell+1} \f$: the labeled sum's effective rank under
      /// its declared policy.
      std::size_t modes{0};
      /// The policy the labeled sum was treated by.
      FiberEmbeddingPolicy policy{FiberEmbeddingPolicy::CarryGramExactly};
      /// \f$ \|G - I\| \f$ of the underlying labeled sum, carried through.
      double gramDefect{std::numeric_limits<double>::quiet_NaN()};
      /// The one-particle operator on the labeled-sum basis,
      /// \f$ h = J^\dagger W L J \f$ (restricted to \f$ (\ker G)^\perp \f$
      /// under `QuotientKernel`), flat row-major (modes x modes).
      std::vector<std::complex<double>> oneParticle{};
      /// The Gram \f$ G \f$ on the same basis, carried so that a
      /// `CarryGramExactly` run can use \f$ h \f$ against it rather than
      /// pretending the basis is orthonormal.
      std::vector<std::complex<double>> gram{};
      /// Eigenvalues of \f$ h \f$, ascending by (Re, Im).
      std::vector<std::complex<double>> oneParticleSpectrum{};
      /// \f$ \dim\Fock(\hh) = 2^M \f$ as a double (exact through 2^53;
      /// +inf beyond). The space itself is never materialized.
      double fockDimension{std::numeric_limits<double>::quiet_NaN()};
      /// Whether the free many-body spectrum below was materialized.
      bool spectrumMaterialized{false};
      /// The exact free many-body spectrum of \f$ d\Gamma(h) \f$: all
      /// \f$ 2^M \f$ occupation subset sums, ascending. Empty (with
      /// `spectrumMaterialized == false`) when the budget refused it.
      std::vector<std::complex<double>> fockSpectrum{};
      /// Certificate of the one-particle compression.
      Certificate certificate{};
    };

    /// One operator-valued link of the next-level response network.
    struct ResponseEdge {
      /// Source component of the link.
      int from{0};
      /// Target component of the link.
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
      double coverageResidual{std::numeric_limits<double>::quiet_NaN()};
      /// Exact-tiling certificate (residual = uncovered magnitude).
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
      double reconstructionResidual{std::numeric_limits<double>::quiet_NaN()};
      /// Realization certificate; `holds()` gates `emitted`.
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
    /// Build over a symmetric PENCIL \f$ (\tilde A, M) \f$ on geometric images
    /// (the chain-level Whitney Hodge pencil, specification §7): `A` and `M`
    /// are flat row-major `dim` x `dim`, `M` the sparse complex-symmetric
    /// inverse chain metric (base level) or the carried Gram
    /// \f$ \mathcal G \f$ (child level). Every shifted elimination is taken on
    /// \f$ \mathcal P(\lambda) = \tilde A - \lambda M \f$ (interior, coupling, and
    /// interface blocks alike), the static reduction at \f$ \lambda = 0 \f$
    /// coincides with the operator path, and a child level carries
    /// \f$ \mathcal G_{\ell+1} = T^T M T \f$ with \f$ T \f$ the constraint modes
    /// (interface cells extended by \f$ -\mathcal P_{II}^{-1}\mathcal P_{IB} \f$,
    /// resonant kernel modes as retained) — the Craig–Bampton congruence,
    /// `FiberEmbeddingPolicy::CarryGramExactly`. Labeled-sum Grams on a pencil
    /// level are \f$ J^T M J \f$ (the transpose pairing). The Hermitian
    /// surrogate's \f$ M^{-1/2} \f$ orthonormalization is never applied to a
    /// pencil level: `nextLevelFromSurrogate` carries the congruence instead.
    [[nodiscard]] static RecursiveQuotient overPencil(
        const std::vector<std::complex<double>> &A,
        const std::vector<std::complex<double>> &M, int dim,
        const std::vector<std::vector<int>> &components,
        const Options &options = Options());
    /// Whether this level is a pencil level (see `overPencil`).
    [[nodiscard]] bool isPencil() const noexcept { return pencil_; }
    /// The pencil's metric \f$ M \f$ (base) or carried Gram \f$ \mathcal G \f$
    /// (child), flat row-major `dim` x `dim`; empty on an operator level.
    [[nodiscard]] std::vector<std::complex<double>> pencilMetric() const;

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

    /// The k-cell vertex tuples of this level's fine coordinates, in
    /// coordinate order. Spacetime paths only: EMPTY on the matrix path and
    /// on child levels, whose coordinates are reduced coordinates rather than
    /// cells. This is what a caller matches a band's cells against — by
    /// vertex SET, never by index — to place a `CertifiedBand`'s frame on
    /// this level's coordinates.
    [[nodiscard]] const std::vector<std::vector<std::uint64_t>> &cellVertices()
        const noexcept {
      return cellVertices_;
    }

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
    ///
    /// The summands here are the RETAINED COORDINATES of the reduction (a
    /// component's claimed interface cells plus the interior modes it owns).
    /// That is the reduction's own stalk structure, and it carries no band
    /// certificate. For the boxed display's \f$ E_v \f$ — the certified
    /// isolated subspace — use `certifiedFiberSum`.
    [[nodiscard]] LabeledFiberSumRead labeledFiberSum() const;

    /// The labeled sum \f$ \boxplus_v E_v \f$ over CERTIFIED ISOLATED BANDS:
    /// the boxed display's \f$ E_v = \f$ "certified isolated subspace of
    /// \f$ C_v \f$", with each band's isolation gap and certificate carried
    /// onto its summand.
    ///
    /// Bands are summed in the order given; an uncertified band is summed and
    /// reported rather than dropped, and makes the sum's certificate fail to
    /// hold. The declared `FiberEmbeddingPolicy` treats the Gram exactly as
    /// for `labeledFiberSum` — adjacent bands may overlap on shared cells, so
    /// an internal direct sum is never asserted.
    /// @throws std::invalid_argument on a frame whose size is not
    ///   `dimension() * rank`, or a band naming an unknown component.
    [[nodiscard]] LabeledFiberSumRead certifiedFiberSum(
        const std::vector<CertifiedBand> &bands) const;

    /// The Fock stage \f$ \Fock(\boxplus_v E_v) \f$ over a labeled sum: the
    /// one-particle compression \f$ h = J^\dagger W L J \f$ onto the sum's
    /// basis, its spectrum, and the exact free many-body spectrum of
    /// \f$ d\Gamma(h) \f$ as occupation subset sums.
    ///
    /// `maxTerms` bounds the materialized many-body spectrum; beyond it the
    /// read REFUSES (`spectrumMaterialized == false`) instead of allocating
    /// \f$ 2^M \f$ entries. Nothing here materializes a Fock vector.
    /// @throws std::invalid_argument when the sum's embedding does not match
    ///   this level's dimension.
    [[nodiscard]] FockStageRead fockStage(
        const LabeledFiberSumRead &sum,
        std::size_t maxTerms = std::size_t{1} << 22) const;

    /// \f$ P = \mathrm{PersistentPartition}(\RN) \f$: partition the
    /// coordinates of an operator-valued response network by persistent
    /// modularity over its off-diagonal magnitude graph
    /// \f$ w_{ij} = |R_{ij}| + |R_{ji}| \f$ (a symmetric nonnegative
    /// similarity; the diagonal never enters).
    ///
    /// This is the DISCOVERY step of the boxed display, available at every
    /// scale rather than at level zero only. Modularity is a heuristic
    /// PROPOSAL generator: it proposes candidate supports and never vetoes an
    /// otherwise certified fiber. Coordinates isolated by the operator come
    /// back as singleton components, so the returned partition always covers
    /// every index exactly once.
    /// @throws std::invalid_argument on a malformed operator size or a
    ///   non-positive restart count.
    [[nodiscard]] static std::vector<std::vector<int>> persistentPartition(
        const std::vector<std::complex<double>> &op, int dim,
        double gamma = 1.0, int restarts = 4, std::uint64_t baseSeed = 0);

    /// `persistentPartition` of THIS level's reduced operator — the partition
    /// \f$ P_\ell \f$ to hand straight to `nextLevel`, so that the recursion
    /// discovers its own components at every scale:
    /// `child = parent.nextLevel(parent.childPersistentPartition())`.
    [[nodiscard]] std::vector<std::vector<int>> childPersistentPartition(
        double gamma = 1.0, int restarts = 4, std::uint64_t baseSeed = 0) const;

    /// The composable amplitude budget of the `CertifiedNearIsometry`
    /// policy: two embeddings with Gram defects \f$ \varepsilon_A,
    /// \varepsilon_B \f$ compose (tensor) to at most
    /// \f[ \varepsilon_{AB} \le \varepsilon_A + \varepsilon_B +
    ///     \varepsilon_A\varepsilon_B , \f]
    /// and the amplitude error obeys
    /// \f$ |a^\dagger G b - a^\dagger b| \le \varepsilon\|a\|\|b\| \f$
    /// (whitepaper, "Interactions and the expanding Hilbert space"). This
    /// is how a certified \f$ \varepsilon \f$ PROPAGATES to composite
    /// reads.
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

    /// Reduce again at \f$ \lambda = 0 \f$: a child quotient over this
    /// level's STATIC reduced operator, with `components` indexing the
    /// REDUCED coordinates. The child inherits provenance ("L<level>:"
    /// prefixes), level + 1, and this level's chain metric restricted
    /// through the reduced coordinates.
    [[nodiscard]] RecursiveQuotient nextLevel(
        const std::vector<std::vector<int>> &components,
        const Options &options) const;

    /// `nextLevel` with this instance's options.
    [[nodiscard]] RecursiveQuotient nextLevel(
        const std::vector<std::vector<int>> &components) const;

    /// Reduce again ON THE PENCIL:
    /// \f$ \RN_{\ell+1}(\lambda) = \mathrm{Feshbach}_{P_\ell}(\RN_\ell(\lambda)) \f$
    /// at a declared \f$ \lambda \f$ over a declared band window. The child's
    /// operator is the exact energy-dependent response \f$ F_B(\lambda) \f$
    /// — NOT the static complement — and it carries the window, the solve and
    /// compatibility residuals, the resonance flag, and the producing
    /// certificate on `levelProvenance()`.
    ///
    /// `components` index the pencil's reduced coordinates, which include any
    /// RESONANT modes retained at \f$ \lambda \f$ and therefore need not
    /// match the static reduction's coordinates. Use
    /// `persistentPartition(feshbach(...).response, ...)` to discover them.
    /// @throws std::invalid_argument when `windowLower > windowUpper` or the
    ///   partition does not cover the pencil's coordinates.
    [[nodiscard]] RecursiveQuotient nextLevelAtLambda(
        const std::vector<std::vector<int>> &components,
        std::complex<double> lambda, double windowLower, double windowUpper,
        const Options &options) const;

    /// `nextLevelAtLambda` with this instance's options.
    [[nodiscard]] RecursiveQuotient nextLevelAtLambda(
        const std::vector<std::vector<int>> &components,
        std::complex<double> lambda, double windowLower,
        double windowUpper) const;

    /// Reduce again through a CERTIFIED LINEAR SURROGATE: the cached
    /// Craig--Bampton/AMLS reduction over a declared frequency window, made
    /// into a child level.
    ///
    /// The surrogate's reduced pencil \f$ (K, M) = (V^\dagger W L V,
    /// V^\dagger W V) \f$ is a GENERALIZED problem, while a level carries a
    /// diagonal chain metric. The child is therefore built on the
    /// \f$ M \f$-orthonormalized basis \f$ V M^{-1/2} \f$: its operator is
    /// \f$ M^{-1/2} K M^{-1/2} \f$ and its metric is the identity. That
    /// congruence preserves the generalized eigenvalues of \f$ (K, M) \f$
    /// EXACTLY, so no spectral content is traded for the convenience — the
    /// approximation is entirely in the truncation, which the carried
    /// certificate and `discardedModeGap` report.
    ///
    /// The child's `levelProvenance().certificate` is the surrogate's own
    /// CERTIFIED-APPROXIMATION certificate, so a surrogate level can never be
    /// mistaken downstream for an exact reduction.
    /// @throws as `craigBampton`, plus std::invalid_argument when the
    ///   partition does not cover the surrogate's coordinates.
    [[nodiscard]] RecursiveQuotient nextLevelFromSurrogate(
        const std::vector<std::vector<int>> &components, double windowLower,
        double windowUpper, double modeCutoff, double residualTolerance,
        const Options &options) const;

    /// `nextLevelFromSurrogate` with this instance's options.
    [[nodiscard]] RecursiveQuotient nextLevelFromSurrogate(
        const std::vector<std::vector<int>> &components, double windowLower,
        double windowUpper, double modeCutoff,
        double residualTolerance = -1.0) const;

    /// How this level was produced from its parent, with the declared window
    /// and the producing step's residuals and certificate.
    [[nodiscard]] const LevelProvenanceRead &levelProvenance() const noexcept {
      return levelProvenance_;
    }

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
    // Measures the regime from the operator and its carried metric. There is
    // no "assert PSD from a convention" path (#805).
    void detectRegime();
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
    // Shared child assembly: build a level over `op` (reduced x reduced) with
    // the chain metric induced by `coordinates`' embeddings through W.
    [[nodiscard]] RecursiveQuotient childOver(
        const std::vector<std::complex<double>> &op,
        const std::vector<RetainedCoordinate> &coordinates,
        const std::vector<std::vector<int>> &components,
        const Options &options) const;
    // Pencil child: the same reduced operator, with the carried Gram
    // T^T M T over the constraint modes of `solves` (and the retained
    // resonant embeddings), as a pencil level.
    [[nodiscard]] RecursiveQuotient pencilChildOver(
        const std::vector<std::complex<double>> &op,
        const std::vector<RetainedCoordinate> &coordinates,
        const std::vector<std::vector<int>> &components,
        const Options &options,
        const std::vector<std::shared_ptr<ComponentSolve>> &solves) const;
    [[nodiscard]] Eigen::MatrixXcd pencilConstraintModes(
        const std::vector<RetainedCoordinate> &coordinates,
        const std::vector<std::shared_ptr<ComponentSolve>> &solves) const;
    // The Gram/policy treatment shared by both labeled-sum entry points.
    [[nodiscard]] LabeledFiberSumRead summarizeFiberSum(
        const std::vector<Eigen::VectorXcd> &columns) const;

    // --- problem data (op_/weights_ refresh under invalidate()) ------------
    Eigen::SparseMatrix<std::complex<double>> op_{};
    Eigen::VectorXcd weights_{};        // diagonal chain metric W
    bool pencil_{false};                // pencil level: shifts by lambda*M, Gram carried
    Eigen::SparseMatrix<std::complex<double>> pencilMetric_{};  // M (base) or G (child)
    double opNorm_{0.0};                // scale for relative residuals
    int dim_{0};
    int degree_{-1};
    int level_{0};
    LevelProvenanceRead levelProvenance_{};
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
