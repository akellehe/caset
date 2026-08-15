// Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
#pragma once

#include "mesh/ForwardDeclarations.h"
#include "matter/MatterConfiguration.h"
#include <complex>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <Eigen/SparseCore>

#ifdef TESSERA_CUDA
#include "cuda/regge_cuda.h"
#endif

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::spacetime {}
// === cross-subsystem fwd-decls ===
namespace tessera::spacetime {
  class Spacetime;
}
namespace tessera::simulations {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::quantum;


/// Regge equation solver.
///
/// Given a triangulated spacetime and a matter configuration, adjusts the
/// edge lengths so that the deficit angles at every hinge satisfy the
/// discretized Einstein equations:
///
/// \f[
///   \frac{\partial S}{\partial \ell^2_e} = 0
/// \f]
///
/// The total action is:
/// \f[
///   S = \underbrace{\sum_h A_h\,\varepsilon_h}_{S_{\text{grav}}}
///     \underbrace{- M \sum_{e \in W} \sqrt{-\ell^2_e}}_{S_{\text{matter}}}
/// \f]
/// (Timelike edges have \f$\ell^2 < 0\f$; spacelike edges have \f$\ell^2 > 0\f$.)
///
/// ## Algorithm
///
/// 1. Compute the Gram matrix \f$G\f$ of each top-simplex from its edge
///    lengths.
/// 2. Derive dihedral angles from the cofactors of \f$G\f$.
/// 3. Sum dihedral angles at each hinge → actual deficit angles.
/// 4. Minimize \f$F = \|\nabla S\|^2 = \sum_e (\partial S/\partial \ell^2_e)^2\f$
///    by gradient descent.  \f$F = 0\f$ exactly at a stationary point of
///    the action, i.e. when the Regge equations are satisfied.
///    (Minimizing \f$S\f$ directly diverges because it is unbounded below.)
///
class ReggeSolver {
  public:
    ReggeSolver(std::shared_ptr<Spacetime> spacetime,
                MatterConfiguration matter);

    // ==================== Geometry queries ====================

    /// Dihedral angle at hinge \a h within top-simplex \a sigma.
    [[nodiscard]] std::complex<double> dihedralAngle(SimplexPtr sigma,
                                        SimplexPtr hinge) const;

    /// Deficit angle at a hinge: \f$\varepsilon_h = 2\pi - \sum_\sigma \theta_h^{(\sigma)}\f$.
    [[nodiscard]] std::complex<double> deficitAngle(SimplexPtr hinge) const;

    /// Area of a triangular hinge (for the Regge action weighting).
    [[nodiscard]] static std::complex<double> hingeArea(SimplexPtr hinge);

    /// Gravitational Regge action: \f$S_{\text{grav}} = \sum_h A_h\,\varepsilon_h\f$, with \f$A_h\f$ the honest signed Lorentzian hinge area and \f$\varepsilon_h\f$ the complex Lorentzian deficit (#641).
    [[nodiscard]] std::complex<double> reggeAction() const;

    /// **Dual** Lorentzian Regge action on \f$W^*\f$:
    /// \f$S_{\text{Regge}}(W^*) = \sum_h |\!\star\! h|\,\varepsilon_h\f$, the
    /// circumcentric **dual** content of each (d-2)-hinge
    /// (``Simplex::dualVolume``) weighted by its **complex Lorentzian** deficit
    /// (``Simplex::lorentzianDeficitAngle``). Returns ``std::complex``: the real
    /// part is the angle-defect curvature, the imaginary part the boost
    /// (rapidity / light-cone) content from timelike-normal-plane hinges. Pure
    /// gravity (matter-independent); the gravitational prior for the
    /// Regge-mediated synthesis objective. Refs: Regge (1961);
    /// Ambjorn-Jurkiewicz-Loll (Lorentzian CDT); Sorkin (Lorentzian angles);
    /// Asante-Dittrich arXiv:2104.00485.
    [[nodiscard]] std::complex<double> dualReggeAction() const;

    /// The \f$(d\!-\!2)\f$ **hinge** vertex-tuples that are faces of the given top
    /// \f$d\f$-cells — the set of hinges whose dual-action contribution a move over
    /// those cells can change. Each input cell is a vertex-id tuple; the result is
    /// the dedup'd set of \f$(d\!-\!1)\f$-vertex sub-tuples (sorted ids). Pure
    /// topology (no geometry): the **affected-hinge index** for the incremental
    /// \f$\Delta S_{\text{Regge}}\f$. Build it from a move's touched cells
    /// (created ∪ removed ∪ the top cofaces of a perturbed edge) once, then use the
    /// SAME set for the before/after legs of ``dualReggeActionOverHinges``.
    [[nodiscard]] std::vector<std::vector<std::uint64_t>> hingeFacesOfCells(
        const std::vector<std::vector<std::uint64_t>> &cells) const;

    /// The **localized** dual (Sorkin) Regge action
    /// \f$\sum_{h} |\!\star\! h|\,\varepsilon_h\f$ over only the given
    /// \f$(d\!-\!2)\f$ hinge tuples that are *genuine* in the live complex (a
    /// registered simplex with a top coface — orphans contribute 0, exactly as in
    /// ``dualReggeAction``). Each tuple is resolved by vertex id (order-independent,
    /// ``findSimplexByVerts``) and the per-term measure is the **same** circumcentric
    /// ``dualVolume`` as ``dualReggeAction``, so it inherits the dual action by
    /// construction. This is the before/after leg of the incremental
    /// \f$\Delta S_{\text{Regge}}\f$: evaluated over a FIXED affected-hinge set
    /// across a move, \f$\Delta S = S_{\text{after}} - S_{\text{before}}\f$ is exact
    /// (every hinge outside the set is unchanged and cancels).
    [[nodiscard]] std::complex<double> dualReggeActionOverHinges(
        const std::vector<std::vector<std::uint64_t>> &hinges) const;

    /// The edges whose per-edge action gradient `∂S/∂ℓ²_e` a move over the given top
    /// cells can change — the **affected-edge index** for the incremental
    /// `Δ‖∇S_Regge‖²`. A move changes `∂S/∂ℓ²_e` only when an affected hinge
    /// (`hingeFacesOfCells(cells)`) contributes to `e`, i.e. only for edges that share
    /// a top cell with an affected hinge. Returns those edges as sorted `(a,b)` id
    /// pairs (deduped). Build it from a move's touched cells; because cells are
    /// added/removed, take the **union** of this set evaluated before and after the
    /// move, then use that fixed set for both legs of `gradientNorm2OverEdges`.
    [[nodiscard]] std::vector<std::pair<std::uint64_t, std::uint64_t>>
    affectedEdgesOfCells(
        const std::vector<std::vector<std::uint64_t>> &cells) const;

    /// The **localized** squared gradient norm `Σ_{e∈edges} |∂S/∂ℓ²_e|²` of the dual
    /// (Sorkin) Regge action — the geometry term of the optimizer objective
    /// `F = ‖∇S_Regge‖² + Γ·r_U` (extremize the action, δS=0). Each `∂S/∂ℓ²_e` is the
    /// **full** per-edge gradient `Σ_{h∋e}[∂|★h|·ε_h + |★h|·∂ε_h]` summed over *all*
    /// hinges incident to `e` (`e`'s star — local), built from the same per-hinge
    /// analytic gradients as `actionGradientExact`; **complex** modulus (Re and Im
    /// together). Over a FIXED affected-edge set across a move,
    /// `Δ‖∇S_Regge‖² = after − before` is exact (every edge outside the set keeps its
    /// gradient and cancels). `gradientNorm2OverEdges` over *all* edges equals
    /// `Σ_e |actionGradientExact()_e|²`.
    [[nodiscard]] double gradientNorm2OverEdges(
        const std::vector<std::pair<std::uint64_t, std::uint64_t>> &edges) const;

    /// Point-particle matter action:
    /// \f$S_{\text{matter}} = -M \sum_{e \in W,\ e\ \text{timelike}} \sqrt{-\mathrm{Re}\,\ell^2_e}\f$.
    /// Causal character is the canonical ``Edge::isTimelike()`` (null edges
    /// contribute 0); under the ordinary-Lorentzian convention (resident
    /// \f$\ell^2\f$ real and signed — see ``Edge::setSquaredLength``, #581)
    /// the Re basis is exact: \f$\sqrt{-\mathrm{Re}\,\ell^2} = \sqrt{-\ell^2}\f$.
    /// Real return by construction.
    [[nodiscard]] double matterAction() const;

    /// Total action: \f$S = S_{\text{grav}} + S_{\text{matter}}\f$.
    /// The Regge equations are \f$\partial S/\partial \ell^2_e = 0\f$.
    [[nodiscard]] std::complex<double> totalAction() const;

    /// Squared gradient norm: \f$F = \sum_e (\partial S/\partial \ell^2_e)^2\f$.
    /// Non-negative; zero exactly when the Regge equations are satisfied.
    [[nodiscard]] double actionGradientNorm() const;

    /// **Exact analytic** gradient of the complex dual (Sorkin) Regge action
    /// ``dualReggeAction`` = Σ_h |★h|·ε_h: ∂S/∂ℓ²_e for each edge, in the
    /// ``getEdgeList()`` order (matching ``actionGradient``). Assembled by the
    /// product rule Σ_h [∂|★h|·ε_h + |★h|·∂ε_h] from the per-hinge analytic
    /// gradients ``Simplex::dualVolumeGradient`` and
    /// ``lorentzianDeficitAngleGradient`` — no finite differences. Complex (Re S
    /// and Im S together). Matches a central-difference of ``dualReggeAction`` to
    /// machine precision but at one pass, not 2·|E| action evaluations.
    [[nodiscard]] std::vector<std::complex<double>> actionGradientExact() const;

    /// Exact analytic Hessian ∂²S/∂ℓ²_e∂ℓ²_f of the dual Lorentzian Regge action,
    /// as a dense |E|×|E| complex matrix in the ``getEdgeList`` order. The next
    /// product-rule layer beyond ``actionGradientExact``:
    /// Σ_h [∂²|★h|·ε_h + ∂|★h|_e·∂ε_h_f + ∂|★h|_f·∂ε_h_e + |★h|·∂²ε_h], assembled
    /// from the per-hinge ``dualVolumeHessian`` / ``lorentzianDeficitAngleHessian``
    /// (and their gradients) — no finite differences. Removes the FD-Hessian floor
    /// in the stationary-action relaxation (exact Newton / Gauss-Newton).
    [[nodiscard]] std::vector<std::vector<std::complex<double>>>
    actionHessianExact() const;

    /// Sparse assembly of the exact analytic Hessian ``actionHessianExact``.
    /// ∂²S/∂ℓ²_e∂ℓ²_f is nonzero only when edges e,f share a hinge (local
    /// coupling), so the Hessian is sparse: assembled directly as an Eigen
    /// ``SparseMatrix`` (never densified) at O(nnz) memory instead of O(|E|²).
    /// Same per-hinge product-rule terms as the dense ``actionHessianExact``
    /// (``hingeHessianEntries``); equal to it to machine precision on the
    /// nonzero pattern. ``getEdgeList`` order; column-major.
    [[nodiscard]] Eigen::SparseMatrix<std::complex<double>>
    actionHessianExactSparse() const;

    // ==================== Solver ====================

    /// One gradient-descent step minimizing \f$F = \|\nabla S\|^2\f$.
    ///
    /// @return F = \f$\|\nabla S\|^2\f$ before the update
    double step(double learningRate = 0.001);

    /// Iterate step() until convergence or max iterations.
    ///
    /// @param progress Optional callback invoked after each iteration with
    ///   (iteration, F).  Useful for progress bars.
    /// @return (converged, final_F, iterations)
    using ProgressCallback = std::function<void(int iter, double F)>;
    std::tuple<bool, double, int> solve(double tol = 1e-8,
                                         int maxIters = 5000,
                                         double learningRate = 0.001,
                                         ProgressCallback progress = nullptr);

    // ==================== Accessors ====================

    [[nodiscard]] const std::shared_ptr<Spacetime> &getSpacetime() const noexcept {
        return spacetime_;
    }

    [[nodiscard]] const MatterConfiguration& getMatter() const noexcept {
        return matter_;
    }

  private:
    std::shared_ptr<Spacetime> spacetime_;
    MatterConfiguration matter_;

    /// Collect all (d-2)-simplices (hinges) in the complex.
    [[nodiscard]] std::vector<SimplexPtr> collectHinges() const;

    /// All (edgeI, edgeJ, ∂²S term) contributions of a single hinge to the
    /// action Hessian, with edge indices resolved via @p eidx. The shared
    /// per-hinge product-rule kernel behind both the dense ``actionHessianExact``
    /// and the sparse ``actionHessianExactSparse`` assemblies.
    [[nodiscard]] std::vector<
        std::tuple<std::size_t, std::size_t, std::complex<double>>>
    hingeHessianEntries(
        const SimplexPtr &hinge,
        const std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t>
            &eidx) const;

    /// Compute the gradient of the total action: ∂S/∂ℓ²_e for each edge.
    [[nodiscard]] std::vector<double> actionGradient() const;

#ifdef TESSERA_CUDA
    /// Flatten mesh topology into GPU-friendly arrays.
    [[nodiscard]] cuda::GpuMeshData flattenMeshForGpu() const;
#endif

};

} // namespace tessera::simulations
