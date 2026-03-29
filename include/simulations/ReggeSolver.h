// MIT License -- Copyright (c) 2025 Andrew Kelleher
#pragma once

#include "mesh/ForwardDeclarations.h"
#include "matter/MatterConfiguration.h"
#include <functional>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

#ifdef CASET_CUDA
#include "cuda/regge_cuda.h"
#endif

namespace caset {

class Spacetime;

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

    /// Point-particle matter action: \f$S_{\text{matter}} = -M \sum_{e \in W} \sqrt{-\ell^2_e}\f$.
    [[nodiscard]] double matterAction() const;

    /// Total action: \f$S = S_{\text{grav}} + S_{\text{matter}}\f$.
    /// The Regge equations are \f$\partial S/\partial \ell^2_e = 0\f$.
    [[nodiscard]] double totalAction() const;

    /// Squared gradient norm: \f$F = \sum_e (\partial S/\partial \ell^2_e)^2\f$.
    /// Non-negative; zero exactly when the Regge equations are satisfied.
    [[nodiscard]] double actionGradientNorm() const;

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

    /// Compute the gradient of the total action: ∂S/∂ℓ²_e for each edge.
    [[nodiscard]] std::vector<double> actionGradient() const;

#ifdef CASET_CUDA
    /// Flatten mesh topology into GPU-friendly arrays.
    [[nodiscard]] cuda::GpuMeshData flattenMeshForGpu() const;
#endif

};

} // namespace caset
