// Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
#pragma once

#include "mesh/ForwardDeclarations.h"
#include "matter/MatterConfiguration.h"
#include <complex>
#include <cstdint>
#include <functional>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

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
    [[nodiscard]] double dihedralAngle(SimplexPtr sigma,
                                        SimplexPtr hinge) const;

    /// Deficit angle at a hinge: \f$\varepsilon_h = 2\pi - \sum_\sigma \theta_h^{(\sigma)}\f$.
    [[nodiscard]] double deficitAngle(SimplexPtr hinge) const;

    /// Area of a triangular hinge (for the Regge action weighting).
    [[nodiscard]] static double hingeArea(SimplexPtr hinge);

    /// Gravitational Regge action: \f$S_{\text{grav}} = \sum_h A_h\,\varepsilon_h\f$.
    [[nodiscard]] double reggeAction() const;

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

    /// Point-particle matter action: \f$S_{\text{matter}} = -M \sum_{e \in W} \sqrt{-\ell^2_e}\f$.
    [[nodiscard]] double matterAction() const;

    /// Total action: \f$S = S_{\text{grav}} + S_{\text{matter}}\f$.
    /// The Regge equations are \f$\partial S/\partial \ell^2_e = 0\f$.
    [[nodiscard]] double totalAction() const;

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

    /// Collect all (d-2)-simplices (hinges) in the complex. Cached; see
    /// ``ensureTopologyCache``. The returned reference is valid until the next
    /// topology change (a rebuild may reallocate the backing vector).
    [[nodiscard]] const std::vector<SimplexPtr>& collectHinges() const;

    /// Compute the gradient of the total action: ∂S/∂ℓ²_e for each edge.
    [[nodiscard]] std::vector<double> actionGradient() const;

    // ---- Topology cache (edge index + hinge list) -----------------------
    // ``eidx`` and the hinge list depend only on the triangulation, not on the
    // edge lengths, so in a fixed-topology relaxation (Phase-2: only ℓ²
    // changes) they are constant across every iteration.  Cache them and
    // rebuild only on a topology change, detected by an O(1) signature
    // ``(edge count, simplex count)`` that a metric-only ``setSquaredLength``
    // leaves untouched but any Pachner add/remove perturbs.

    /// Sorted vertex-id edge key, matching the per-hinge gradient/Hessian maps
    /// (``Simplex::lorentzianDeficitAngleGradient`` etc.).
    using EdgeKey = std::pair<std::uint64_t, std::uint64_t>;

    /// Hash for ``EdgeKey`` (``std::unordered_map`` needs an explicit one for
    /// ``std::pair``). Defined out-of-line over ``Fingerprint::mix64``.
    struct EdgeKeyHash {
        [[nodiscard]] std::size_t operator()(const EdgeKey &key) const noexcept;
    };

    /// Edge key → position in ``getEdgeList()->toVector()`` (the gradient/Hessian
    /// row/column order, matching ``actionGradient``).
    using EdgeIndex = std::unordered_map<EdgeKey, std::size_t, EdgeKeyHash>;

    mutable std::vector<SimplexPtr> cachedHinges_;
    mutable EdgeIndex cachedEidx_;
    mutable std::size_t cachedEdgeCount_ = 0;
    mutable std::pair<std::size_t, std::size_t> cachedTopologySignature_{0, 0};
    mutable bool topologyCached_ = false;

    /// (Re)build ``cachedHinges_`` and ``cachedEidx_`` if the triangulation has
    /// changed since the last build (or it has never been built); a no-op on a
    /// pure metric change. Called by ``collectHinges``, ``actionGradientExact``
    /// and ``actionHessianExact``.
    void ensureTopologyCache() const;

#ifdef TESSERA_CUDA
    /// Flatten mesh topology into GPU-friendly arrays.
    [[nodiscard]] cuda::GpuMeshData flattenMeshForGpu() const;
#endif

};

} // namespace tessera::simulations
