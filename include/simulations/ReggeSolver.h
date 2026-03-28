// MIT License -- Copyright (c) 2025 Andrew Kelleher
#pragma once

#include "mesh/ForwardDeclarations.h"
#include "matter/MatterConfiguration.h"
#include <functional>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace caset {

class Spacetime;

/// Regge equation solver.
///
/// Given a triangulated spacetime and a matter configuration, adjusts the
/// edge lengths so that the deficit angles at every hinge satisfy the
/// discretized Einstein equations:
///
/// \f[
///   \varepsilon_h = \varepsilon^{\text{target}}_h
/// \f]
///
/// where the target deficit is determined by the local stress-energy.
///
/// ## Algorithm
///
/// 1. Compute the Gram matrix \f$G\f$ of each top-simplex from its edge
///    lengths.
/// 2. Derive dihedral angles from the cofactors of \f$G\f$.
/// 3. Sum dihedral angles at each hinge → actual deficit angles.
/// 4. Gradient-descend on the squared residual
///    \f$L = \sum_h (\varepsilon_h - \varepsilon^*_h)^2\f$
///    with respect to squared edge lengths.
///
class ReggeSolver {
  public:
    ReggeSolver(std::shared_ptr<Spacetime> spacetime,
                MatterConfiguration matter);

    // ==================== Geometry queries ====================

    /// Dihedral angle at hinge \a h within top-simplex \a sigma.
    ///
    /// The hinge is a \f$(d{-}2)\f$-simplex (triangle in 4-D).  The dihedral
    /// angle is the angle between the two \f$(d{-}1)\f$-faces of \a sigma
    /// that meet at \a h.
    ///
    /// Uses the Gram-matrix cofactor formula (Simplex.h §295–306):
    /// \f[
    ///   \cos\theta_{ij} = -\frac{C_{ij}}{\sqrt{C_{ii}\,C_{jj}}}
    /// \f]
    /// where \a i,j are the two vertices of \a sigma opposite to \a h.
    [[nodiscard]] double dihedralAngle(SimplexPtr sigma,
                                        SimplexPtr hinge) const;

    /// Deficit angle at a hinge: \f$\varepsilon_h = 2\pi - \sum_\sigma \theta_h^{(\sigma)}\f$.
    [[nodiscard]] double deficitAngle(SimplexPtr hinge) const;

    /// Area of a triangular hinge (for the Regge action weighting).
    [[nodiscard]] static double hingeArea(SimplexPtr hinge);

    /// Full Regge action \f$S = \sum_h A_h\,\varepsilon_h\f$.
    [[nodiscard]] double reggeAction() const;

    // ==================== Solver ====================

    /// Squared residual \f$L = \sum_h (\varepsilon_h - \varepsilon^*_h)^2\f$.
    [[nodiscard]] double residual() const;

    /// One gradient-descent step on the squared residual.
    /// Adjusts every edge's squared length by
    /// \f$\ell^2_e \mathrel{-}= \eta\,\partial L / \partial \ell^2_e\f$.
    ///
    /// @return the new residual after the step
    double step(double learningRate = 0.001);

    /// Iterate step() until convergence or max iterations.
    ///
    /// @param progress Optional callback invoked after each iteration with
    ///   (iteration, residual).  Useful for progress bars.
    /// @return (converged, final_residual, iterations)
    using ProgressCallback = std::function<void(int iter, double residual)>;
    std::tuple<bool, double, int> solve(double tol = 1e-8,
                                         int maxIters = 5000,
                                         double learningRate = 0.001,
                                         ProgressCallback progress = nullptr);

    // ==================== Accessors ====================

    [[nodiscard]] std::shared_ptr<Spacetime> getSpacetime() const noexcept {
        return spacetime_;
    }

    [[nodiscard]] const MatterConfiguration& getMatter() const noexcept {
        return matter_;
    }

  private:
    std::shared_ptr<Spacetime> spacetime_;
    MatterConfiguration matter_;
    std::unordered_map<std::uint64_t, double> targetDeficits_;

    /// Collect all (d-2)-simplices (hinges) in the complex.
    [[nodiscard]] std::vector<SimplexPtr> collectHinges() const;

    /// Build the Gram matrix for a top-simplex from its edge lengths.
    /// Returns a flat (d×d) row-major matrix (vertex 0 is the origin).
    [[nodiscard]] static std::vector<double> gramMatrix(SimplexPtr sigma);

    /// Cofactor matrix of a square matrix (flat row-major, size n×n).
    [[nodiscard]] static std::vector<double> cofactorMatrix(
        const std::vector<double> &M, int n);

    /// Determinant of a square matrix (flat row-major, size n×n).
    [[nodiscard]] static double determinant(
        const std::vector<double> &M, int n);
};

} // namespace caset
