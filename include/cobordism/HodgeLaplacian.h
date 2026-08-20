// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_HODGELAPLACIAN_H
#define TESSERA_COBORDISM_HODGELAPLACIAN_H

#include <complex>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "cobordism/Cochain.h"
#include "cobordism/Spectrum.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime { class Spacetime; }
namespace tessera::cobordism {
using namespace ::tessera::spacetime;

/// # HodgeLaplacian
///
/// The Hodge Laplacian on a `Spacetime`, degree-parameterized by `int k`.
///
/// At \f$ k = 0 \f$ it is the **Hermitian U(1)-weighted graph Laplacian**
/// \f$ L = D - A \f$ on the 1-skeleton, assembled directly from the complex edge
/// weights \f$ \text{squaredLength}\cdot e^{i\,\text{phase}} \f$ carried by each
/// `Edge` (the magnitude convention on the degree keeps \f$ L \f$ Hermitian and
/// \f$ e^{-iLt} \f$ unitary). This path is unchanged.
///
/// At \f$ k \geq 1 \f$ it is the **metric Hodge Laplacian**, assembled from the
/// integer boundary maps \f$ \partial_k,\partial_{k+1} \f$ (`ChainComplex`) and
/// the diagonal inner-product weights \f$ W_k \f$ — the per-\f$ k \f$-simplex
/// Euclidean volumes (`Simplex::volume`), with \f$ W_0 = I \f$. With the metric
/// adjoint \f$ \partial_k^* = W_k^{-1}\partial_k^{\top} W_{k-1} \f$ the Laplacian
/// is \f$ L_k = \partial_k^*\partial_k + \partial_{k+1}\partial_{k+1}^* \f$,
/// returned in its **symmetric** (\f$ W_k \f$-orthonormal) representation
/// \f$ W_k^{1/2} L_k W_k^{-1/2} = B_k^{\top}B_k + B_{k+1}B_{k+1}^{\top} \f$ with
/// \f$ B_k = W_{k-1}^{1/2}\partial_k W_k^{-1/2} \f$ — symmetric positive
/// semidefinite, so `SelfAdjointEigenSolver` applies and the spectrum/kernel are
/// those of \f$ L_k \f$ (a similarity). By the discrete Hodge theorem
/// \f$ \ker L_k \cong H_k \f$, so \f$ \dim\ker L_k = b_k \f$ for **any** positive
/// weights; passing `metric = false` uses unit weights — the combinatorial
/// \f$ \partial_k^{\top}\partial_k + \partial_{k+1}\partial_{k+1}^{\top} \f$ — as
/// a same-kernel cross-check. A negative \f$ k \f$ throws; a \f$ k \f$ beyond the
/// top dimension has no \f$ k \f$-cells and yields empty results.
///
/// ## Lorentzian d'Alembertian (§5.6)
///
/// The `lorentzian = true` path keeps the same metric-Hodge construction but
/// weights \f$ W_k \f$ with the **signed** `Simplex::volume()` — the honest,
/// signature-respecting content in which a timelike edge (\f$ l^2 < 0 \f$) carries
/// a *negative* volume — instead of \f$ |\text{volume}| \f$. The inner product
/// then goes **indefinite**: the symmetric \f$ W_k^{1/2} \f$ similarity breaks
/// (the square root of a negative weight is imaginary), so the operator
/// \f$ L_k = \partial_k^*\partial_k + \partial_{k+1}\partial_{k+1}^* \f$ with the
/// *signed* metric adjoint \f$ \partial_k^* = W_k^{-1}\partial_k^{\top}W_{k-1} \f$
/// is assembled **directly** — \f$ L_k = W_k^{-1}\partial_k^{\top}W_{k-1}\partial_k
/// + \partial_{k+1}W_{k+1}^{-1}\partial_{k+1}^{\top}W_k \f$ — and is generally
/// **non-self-adjoint** (a discrete d'Alembertian). It is diagonalized with a
/// general `Eigen::EigenSolver`, so eigenvalues may be negative or complex. The
/// clean \f$ \ker L_k \cong H_k \f$ degrades to a pseudo-Hodge decomposition:
/// "harmonic" becomes the small-\f$ |\lambda| \f$ near-kernel, and a near-kernel
/// representative \f$ h \f$ may be **null** in the indefinite metric
/// (\f$ \langle h,h\rangle_W = \sum_i W_{k,i}|h_i|^2 \approx 0 \f$). When every
/// \f$ l^2 > 0 \f$ the signed content equals \f$ |\text{volume}| \f$ and this path
/// reproduces the Euclidean spectrum/kernel above. The Euclidean
/// (`lorentzian = false`) path is untouched.
///
/// For \f$ k = 0 \f$ vertices are indexed by a stable order (sorted vertex id
/// \f$ \to 0..N-1 \f$); for \f$ k \geq 1 \f$ the \f$ k \f$-cells follow the
/// canonical `ChainComplex` column order (sorted vertex-id tuples), so the
/// returned flat row-major matrices are reproducible and align with
/// `boundaryMatrix(k)` and `weights(k)`.
///
/// ## Assembly (k = 0)
///
/// - Adjacency \f$ A_{ij} = \sum_{(i,j)} \text{squaredLength}\cdot e^{i\,\text{phase}} \f$,
///   summed over edges between \f$ i \f$ and \f$ j \f$; the stored source→target
///   orientation carries \f$ +\text{phase} \f$, the reverse carries
///   \f$ -\text{phase} \f$, so \f$ A = A^\dagger \f$ (Hermitian).
/// - Degree \f$ D_{ii} = \sum |\text{squaredLength}| \f$ over incident edges
///   (the magnitude convention — keeps \f$ L \f$ Hermitian and \f$ e^{-iLt} \f$
///   unitary for complex weights).
/// - Laplacian \f$ L = D - A \f$.
///
/// This class is the *operator* only: it does not compute fluxes, cycle bases,
/// or Betti numbers (those are `WilsonLoop` / `ChainComplex`'s job), and it does
/// not gauge-transform the mesh (gauge invariance is exercised by rephasing the
/// edges and rebuilding). The Hermitian eigendecomposition is lazily computed
/// (Eigen `SelfAdjointEigenSolver<MatrixXcd>`) and cached.
class HodgeLaplacian {
  public:
    /// Which quantity the diagonal inner-product weight \f$ W_k \f$ is built from.
    ///
    /// BOTH are fully Lorentzian and complex-valued; this is a choice of inner
    /// product, not of signature, and neither reintroduces a Euclidean path.
    ///
    /// * `Content` — the \f$ k \f$-content itself, \f$ W_k = V \f$. This is the
    ///   textbook diagonal DEC star: the weight of a \f$ k \f$-simplex is its
    ///   \f$ k \f$-volume. For an edge that is \f$ \ell = \sqrt{\ell^2} \f$, so a
    ///   TIMELIKE edge's weight is imaginary. Consequence: spacelike and timelike
    ///   contributions to \f$ \langle h,h\rangle_W \f$ are 90° apart in the complex
    ///   plane and can never cancel, so no null (lightlike) kernel direction
    ///   exists at any boost.
    ///
    /// * `SquaredContent` — the squared \f$ k \f$-content, \f$ W_k = V^2 \f$. For an
    ///   edge that is exactly \f$ \ell^2 \f$. Being \f$ \det G/(d!)^2 \f$ it is a
    ///   POLYNOMIAL in the squared edge lengths, so on real signed \f$ \ell^2 \f$ it
    ///   is real and signed — timelike cells carry a negative weight rather than
    ///   an imaginary one, and genuine null kernel directions survive.
    ///
    /// The two give measurably different spectra. On the 3-cycle with one
    /// timelike edge (\f$ \ell^2 = -\alpha^2 \f$) `Content` gives
    /// \f$ \mathrm{spec}(L_1) = \{0, 3, 1 - 2i/\alpha\} \f$ and
    /// \f$ \langle h,h\rangle_W = 2/3 + i\alpha/3 \f$, with no null crossing;
    /// `SquaredContent` keeps the weights real-signed and restores one.
    enum class WeightConvention { Content, SquaredContent };

    /// Whether spectral-entropy diagnostics retain the complex entries of
    /// \f$L_k\f$ or perform a phase-blind ablation first.
    ///
    /// `IncludeComplexPhase` uses \f$M=L_k\f$. `IgnoreComplexPhase` uses the
    /// entrywise magnitude \f$M_{ij}=|(L_k)_{ij}|\f$. The latter does NOT replace
    /// an edge length or squared edge length by its magnitude; it changes only
    /// the operator used by this entropy observable.
    enum class EntropyPhaseMode { IncludeComplexPhase, IgnoreComplexPhase };

    /// Construct the operator over a triangulation. Edge weights/phases are read
    /// lazily (at the first matrix/spectrum query), so the spacetime must
    /// outlive the operator; the held `shared_ptr` keeps it alive.
    ///
    /// @param weights Which quantity \f$ W_k \f$ is built from (see
    ///   `WeightConvention`). Defaults to `SquaredContent`: it is real and signed
    ///   on real signed \f$ \ell^2 \f$, polynomial in the squared edge lengths so
    ///   it carries no branch, and it preserves genuine null kernel directions.
    explicit HodgeLaplacian(std::shared_ptr<Spacetime> st,
                            WeightConvention weights = defaultWeightConvention());

    /// The process-wide default `WeightConvention`, read by the constructor's
    /// default argument AT THE CALL SITE — so every internally-constructed
    /// operator (MultiCobordism's r_U terms, the near-kernel residual, the
    /// register readout, the observables) follows it unless a caller passes an
    /// explicit convention. Ships as `SquaredContent`; an experiment (e.g. the
    /// animation's --hodge-weights flag) may flip it ONCE at startup. Not a
    /// per-call knob: flipping it mid-run mixes conventions across cached
    /// spectra.
    [[nodiscard]] static WeightConvention defaultWeightConvention() noexcept {
      return defaultWeightConvention_;
    }
    static void setDefaultWeightConvention(WeightConvention convention) noexcept {
      defaultWeightConvention_ = convention;
    }

    /// Weighted adjacency \f$ A \f$ as a flat row-major \f$ N\times N \f$ array
    /// of complex entries. Hermitian by construction.
    [[nodiscard]] std::vector<std::complex<double>> adjacency() const;

    /// Degree vector \f$ (D_{00},\dots,D_{N-1,N-1}) \f$, real, length \f$ N \f$
    /// (magnitude convention \f$ D_{ii} = \sum |\text{squaredLength}| \f$).
    [[nodiscard]] std::vector<double> degree() const;

    /// Laplacian \f$ L_k \f$ as a flat row-major matrix of complex entries:
    /// \f$ N\times N \f$ for \f$ k = 0 \f$ (\f$ L = D - A \f$), else
    /// \f$ |C_k|\times|C_k| \f$ (the symmetric metric Laplacian above; imaginary
    /// parts are zero). For \f$ k \geq 1 \f$, `metric = false` selects unit
    /// weights (the combinatorial Laplacian); `metric` is ignored at \f$ k = 0 \f$.
    /// @throws std::runtime_error for \f$ k < 0 \f$. Empty for \f$ k \f$ above the
    ///   top dimension.
    /// With `lorentzian = true` (and `metric`, \f$ k \geq 1 \f$) it is instead the
    /// **signed-weight d'Alembertian** assembled directly (generally
    /// non-symmetric; the imaginary parts of the returned entries are still zero,
    /// the operator being real).
    [[nodiscard]] std::vector<std::complex<double>> laplacian(int k = 0,
                                                             bool metric = true) const;

    /// Diagonal inner-product weights \f$ W_k \f$ (length \f$ |C_k| \f$) in the
    /// canonical `ChainComplex` column order: the per-\f$ k \f$-simplex Euclidean
    /// volume (`Simplex::volume`, magnitude; degenerate cells fall back to 1 to
    /// keep \f$ W_k \f$ positive). \f$ W_0 = I \f$ (all ones). Empty for \f$ k < 0 \f$
    /// or \f$ k \f$ above the top dimension. With `lorentzian = true` the entries
    /// are the **signed** `Simplex::volume()` (timelike cells negative; degenerate
    /// cells still fall back to \f$ +1 \f$ so \f$ W_k \f$ stays invertible).
    [[nodiscard]] std::vector<std::complex<double>> weights(int k) const;

    /// Exact analytic gradient \f$ \partial L_k^{\text{sym}} / \partial \ell^2_e \f$
    /// of the symmetric metric Hodge Laplacian (\f$ k \ge 1 \f$) with respect to one
    /// edge's squared length, as a flat \f$ |C_k|\times|C_k| \f$ row-major matrix in
    /// the canonical column order. With \f$ L_k = B_k^\top B_k + B_{k+1}B_{k+1}^\top \f$,
    /// \f$ B_k=\mathrm{diag}(\sqrt{W_{k-1}})\,\partial_k\,\mathrm{diag}(1/\sqrt{W_k}) \f$,
    /// only the inner-product weights \f$ W_j=|\!\operatorname{vol}| \f$ depend on
    /// \f$ \ell^2 \f$, so \f$ \partial B_k=\mathrm{diag}(a_{k-1})B_k+B_k\,\mathrm{diag}(b_k) \f$
    /// with \f$ a_j=\tfrac{\partial W_j}{2W_j} \f$ — and \f$ \partial W_j \f$ is the
    /// per-simplex `Simplex::volumeGradient` (signed for the `|vol|` weight). The
    /// degree-generic keystone for the arbitrary-\f$ k \f$ \f$ r_U \f$ gradient.
    /// Empty for \f$ k < 1 \f$ or an absent edge.
    [[nodiscard]] std::vector<std::complex<double>> laplacianGradient(
        int k, std::uint64_t edgeA, std::uint64_t edgeB) const;

    /// Von Neumann entropy of the normalized positive Hodge operator
    /// \f[ A_k=M_k^\dagger M_k,\qquad
    ///     \rho_k=A_k/\operatorname{Tr}A_k,\qquad
    ///     S_k=-\operatorname{Tr}(\rho_k\log\rho_k). \f]
    ///
    /// `phaseMode` selects \f$M_k=L_k\f$ or the phase-blind entrywise
    /// \f$|L_k|\f$ ablation. Empty and identically-zero operators have entropy
    /// zero. Exact zero modes are omitted from \f$0\log 0\f$ and from the
    /// derivative, which is the derivative on the fixed-rank stratum selected
    /// by the current topology.
    [[nodiscard]] double spectralEntropy(
        int k, EntropyPhaseMode phaseMode =
                   EntropyPhaseMode::IncludeComplexPhase) const;

    /// Gradient of `spectralEntropy` with respect to the COMPLEX squared edge
    /// coordinates \f$z_e=\ell_e^2\f$, in `EdgeList` order. The returned
    /// convention is
    /// \f$h_e=\partial S/\partial\operatorname{Re}z_e
    ///       -i\,\partial S/\partial\operatorname{Im}z_e\f$, so
    /// \f$\overline h\f$ is the steepest-ascent displacement in the complex
    /// \f$z\f$ plane. Available for the metric operators \f$k\ge1\f$; degree
    /// zero throws because its magnitude-weighted diagonal is non-holomorphic
    /// and has no `laplacianGradient` implementation.
    [[nodiscard]] std::vector<std::complex<double>> spectralEntropyGradient(
        int k, EntropyPhaseMode phaseMode =
                   EntropyPhaseMode::IncludeComplexPhase) const;

    /// \f$\sum_e |\partial S/\partial z_e|^2\f$, the entropy-stationarity
    /// residual used by the joint Regge-Hodge objective.
    [[nodiscard]] double spectralEntropyGradientNorm(
        int k, EntropyPhaseMode phaseMode =
                   EntropyPhaseMode::IncludeComplexPhase) const;

    /// Whether \f$ \| L - L^\dagger \| \le \text{tol} \f$ (Frobenius norm) for
    /// the \f$ k = 0 \f$ Laplacian. True by construction.
    [[nodiscard]] bool isHermitian(double tol = 1e-12) const;

    /// Unitarity residual of the time-evolution operator
    /// \f$ U = e^{-iLt} = V\,\mathrm{diag}(e^{-i\lambda t})\,V^\dagger \f$ formed
    /// from the eigendecomposition: returns \f$ \| U U^\dagger - I \| \f$
    /// (Frobenius). ~0 for the Hermitian \f$ L \f$.
    [[nodiscard]] double unitarityResidual(double t = 1.0) const;

    /// The eigendecomposition of \f$ L_k \f$ as a `Spectrum` (real ascending
    /// eigenvalues + eigenvectors as `Cochain`s; `Spectrum::isHermitian()` is
    /// true). The self-adjoint \f$ k = 0 \f$ graph Laplacian and the symmetric
    /// metric Hodge Laplacian (\f$ k \geq 1 \f$). For \f$ k \geq 1 \f$, `metric`
    /// selects volume vs. unit weights (ignored at \f$ k = 0 \f$). The eigenvectors
    /// are indexed over the sorted-id vertex order (\f$ k = 0 \f$) or the canonical
    /// `ChainComplex` \f$ k \f$-simplex column order (\f$ k \geq 1 \f$).
    /// @throws std::runtime_error for \f$ k < 0 \f$. Empty above the top dimension.
    [[nodiscard]] Spectrum spectrum(int k = 0, bool metric = true) const;

    /// Eigenvalues of \f$ L_k \f$ (real, ascending), a flat view consistent with
    /// `spectrum(k, metric)`. For \f$ k \geq 1 \f$, `metric` selects volume vs.
    /// unit weights (ignored at \f$ k = 0 \f$).
    /// @throws std::runtime_error for \f$ k < 0 \f$. Empty above the top dimension.
    [[nodiscard]] std::vector<std::complex<double>> eigenvalues(int k = 0, bool metric = true) const;

    /// Eigenvectors of \f$ L_k \f$ as a flat row-major \f$ M\times M \f$ array
    /// (\f$ M = N \f$ for \f$ k = 0 \f$, else \f$ |C_k| \f$); column \f$ j \f$
    /// (entries at indices \f$ iM + j \f$) is the eigenvector for the
    /// \f$ j \f$-th ascending eigenvalue — a flat view consistent with the
    /// `Cochain`s of `spectrum(k, metric)`. For \f$ k \geq 1 \f$, `metric` selects
    /// volume vs. unit weights (ignored at \f$ k = 0 \f$).
    /// @throws std::runtime_error for \f$ k < 0 \f$. Empty above the top dimension.
    [[nodiscard]] std::vector<std::complex<double>> eigenvectors(int k = 0,
                                                               bool metric = true) const;

    /// Harmonic representatives: the eigenvectors with \f$ |\lambda| < \text{tol} \f$
    /// (a basis for \f$ \ker L_k \cong H_k \f$, so the count is the harmonic
    /// dimension \f$ = b_k \f$), as `Cochain`s over the \f$ k \f$-simplex ordering.
    /// For \f$ k \geq 1 \f$, `metric` selects volume vs. unit weights (ignored at
    /// \f$ k = 0 \f$). @throws std::runtime_error for \f$ k < 0 \f$. Empty above the
    /// top dimension.
    [[nodiscard]] std::vector<Cochain> harmonics(int k = 0, double tol = 1e-9,
                                                 bool metric = true) const;

    /// The harmonic amplitude matrix: the same representatives as
    /// `harmonics(k, tol, metric)` — the eigenvectors with
    /// \f$ |\lambda| < \text{tol} \f$, in ascending-eigenvalue order — stacked
    /// as the **rows** of a flat row-major \f$ \dim\ker L_k \times M \f$
    /// complex array (\f$ M = N \f$ at \f$ k = 0 \f$, else \f$ |C_k| \f$),
    /// columns in the same sorted-vertex / canonical `ChainComplex`
    /// \f$ k \f$-cell order the `Cochain`s index. One call replaces the
    /// per-cell `amplitudeFor` round-trips a register layer makes to read its
    /// harmonics; entry \f$ [\,r\,M + c\,] \f$ equals
    /// `harmonics(k, tol, metric)[r].amplitude(c)` exactly. Empty when the
    /// kernel is empty (\f$ b_k = 0 \f$) or \f$ k \f$ is above the top
    /// dimension. For \f$ k \geq 1 \f$, `metric` selects volume vs. unit
    /// weights (ignored at \f$ k = 0 \f$).
    /// @throws std::runtime_error for \f$ k < 0 \f$.
    [[nodiscard]] std::vector<std::complex<double>> harmonicMatrix(
        int k = 0, double tol = 1e-9, bool metric = true) const;

    /// The indefinite norms \f$ \langle h,h\rangle_W = \sum_i W_{k,i}|h_i|^2 \f$
    /// (signed \f$ W_k \f$) of the near-kernel representatives, one per column of
    /// `harmonics(k, tol, metric)` and in the same order. A value
    /// \f$ \approx 0 \f$ flags a **null** harmonic (a lightlike kernel direction);
    /// all entries are positive on an all-spacelike complex.
    /// @throws std::runtime_error for \f$ k < 0 \f$.
    [[nodiscard]] std::vector<std::complex<double>> nullNorms(
        int k, double tol = 1e-9, bool metric = true) const;

  private:
    std::shared_ptr<Spacetime> st_;
    WeightConvention weightConvention_{WeightConvention::SquaredContent};
    static WeightConvention defaultWeightConvention_;

    // Stable vertex order: ids_[idx] = vertex id, idToIndex_[id] = idx. Built
    // once in the constructor (the vertex set is fixed for the operator's life;
    // only the edge weights/phases are read lazily).
    std::vector<std::uint64_t> ids_{};
    std::unordered_map<std::uint64_t, std::size_t> idToIndex_{};
    std::size_t order_{0};  // N = |V|

    // Lazy, cached Hermitian eigendecomposition of the k=0 Laplacian.
    mutable bool decomposed_{false};
    mutable std::vector<double> evals_{};               // ascending, length N
    mutable std::vector<std::complex<double>> evecs_{};  // flat N*N, columns

    // General (non-symmetric) eigendecomposition of the signed-weight d'Alembertian
    // L_k, cached per (k, metric). Eigenvalues/eigenvectors are complex and sorted
    // ascending by (Re, Im); `wk` is the signed weight diagonal kept for the
    // indefinite null-norm <h,h>_W = sum_i wk[i] |h_i|^2.
    struct SpectrumCache {
      int dim{0};
      std::vector<std::complex<double>> evals{};         // sorted, length |C_k|
      std::vector<std::complex<double>> evecs{};         // flat |C_k|*|C_k|, columns
      std::vector<std::complex<double>> wk{};                          // signed W_k, length |C_k|
    };
    /// The spectrum map, SHARED across every HodgeLaplacian built on the same
    /// spacetime at the same geometry (#688): the residual/readout path
    /// constructs a fresh operator per call (~275x per objective evaluation,
    /// measured on #683), and a per-instance map re-ran the dense
    /// ComplexEigenSolver each time. The constructor adopts the spacetime's
    /// revision-stamped spectral slot when current, else creates a fresh map
    /// and stores it; entries are keyed by (k, metric, weight convention), so
    /// instances with different conventions share the map without collisions.
    /// The deliberately-uncached ``apply``/``residual`` honesty paths and the
    /// k=0 Hermitian decomposition below are untouched.
    struct SharedSpectrumMap {
      std::unordered_map<long long, SpectrumCache> map{};
    };
    std::shared_ptr<SharedSpectrumMap> sharedSpectra_{};

    // Throw for k < 0 (no negative-degree chains).
    static void requireNonNegativeDegree(int k);

    // The sorted vertex-id tuples a degree-k Cochain is indexed over.
    // `useVertexSet` returns the full sorted-id vertex order (the Hermitian k=0
    // basis, length N); otherwise the canonical ChainComplex k-simplex column
    // order (the metric / Lorentzian basis, length |C_k|).
    [[nodiscard]] std::vector<std::vector<std::uint64_t>> cochainOrdering(
        int k, bool useVertexSet) const;

    // Assemble a Spectrum from flat eigenvalues/eigenvectors. `evecsFlat` is
    // row-major dim*dim with entry [i*dim + j] = component i of eigenvector j
    // (column j); `ordering` indexes the components; `hermitian` flags the
    // real-ascending regime.
    static Spectrum makeSpectrum(
        int degree, std::vector<std::vector<std::uint64_t>> ordering,
        const std::vector<std::complex<double>> &evals,
        const std::vector<std::complex<double>> &evecsFlat, int dim,
        bool hermitian);

    // Build/fetch the cached symmetric spectrum of L_k^sym (k >= 1). Key folds in
    // `metric` so the metric and combinatorial spectra are cached separately.

    // Build/fetch the cached general spectrum of the signed-weight d'Alembertian.
    const SpectrumCache &ensureSpectrum(int k, bool metric) const;

    // Assemble the adjacency (flat row-major N*N) and degree (length N) from the
    // current edge weights/phases, using the stable vertex order. Kept Eigen-free
    // in its signature so the public header carries no Eigen dependency.
    void assemble(std::vector<std::complex<double>> &A, std::vector<double> &D) const;

    // Build and cache the eigendecomposition of L (k=0) if not already done.
    void ensureDecomposition() const;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_HODGELAPLACIAN_H
