// MIT License
// Copyright (c) 2025 Andrew Kelleher
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef TESSERA_COBORDISM_LEVENBERGMARQUARDT_H
#define TESSERA_COBORDISM_LEVENBERGMARQUARDT_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>

#include <Eigen/Dense>

namespace tessera::cobordism {

/// # LevenbergMarquardt
///
/// A focused **bounded least-squares Levenberg–Marquardt** solver with random
/// multi-restart, for the non-convex residual landscapes the §4b synthesis loop
/// produces. It minimizes \f$ r(x) = \|f(x)\|^2 \f$ over a box-constrained
/// parameter vector \f$ x \f$, where \f$ f \f$ is an arbitrary residual the
/// caller supplies.
///
/// The solver is **decoupled** from any particular model: it is driven entirely
/// by three caller-supplied functors, so it knows nothing of eigenstates,
/// complexes, or cobordism —
///   * **`Residual`** — \f$ x \mapsto f(x) \f$, the least-squares residual
///     vector (the cost is \f$ \|f(x)\|^2 \f$);
///   * **`Clamp`** — project a parameter vector back into the feasible box;
///   * **`Sample`** — draw a fresh random restart point from a given engine.
///
/// ## Algorithm
///
/// Each `minimize` is one bounded descent: a numerical (central-difference)
/// Jacobian \f$ J \f$, damped normal equations
/// \f$ (J^\top J + \mu\,\mathrm{diag}(J^\top J))\,\delta = -J^\top f \f$, and a
/// \f$ \mu \f$ line search (shrink \f$ \mu \f$ on a cost decrease, grow it
/// otherwise) — stopping at `maxIterations`, when no step improves, or once the
/// cost falls below the configured `epsilon`. `multiRestart` runs `minimize`
/// from independently sampled starts and keeps the best, stopping early as soon
/// as a restart drives the cost below the supplied acceptance threshold (the
/// landscape is non-convex, so a single descent is not enough).
///
/// After `minimize` / `multiRestart` returns, the `Residual` functor has been
/// evaluated last at the returned parameters — so any state it writes as a side
/// effect (e.g. model parameters) is left realized at the best point.
class LevenbergMarquardt {
  public:
    /// Maps a parameter vector to the least-squares residual \f$ f(x) \f$; the
    /// cost minimized is \f$ \|f(x)\|^2 \f$.
    using Residual = std::function<Eigen::VectorXd(const Eigen::VectorXd &)>;
    /// Projects a parameter vector into the feasible box (returns the projection).
    using Clamp = std::function<Eigen::VectorXd(const Eigen::VectorXd &)>;
    /// Draws a fresh random restart point from the given engine.
    using Sample = std::function<Eigen::VectorXd(std::mt19937_64 &)>;

    /// The outcome of a minimization: the best parameter vector found and its
    /// cost \f$ r = \|f(x)\|^2 \f$.
    struct Result {
      Eigen::VectorXd parameters{};
      double cost{std::numeric_limits<double>::infinity()};
    };

    /// @param maxIterations cap on the LM iterations of a single `minimize`.
    /// @param epsilon       convergence tolerance: a descent stops once its cost
    ///                      falls to or below this (0 ⇒ run the full budget).
    explicit LevenbergMarquardt(int maxIterations = 200, double epsilon = 0.0);

    /// One bounded LM descent from `x0` (clamped first). Returns the best point
    /// reached and its cost; leaves `residual` evaluated at that point.
    [[nodiscard]] Result minimize(const Residual &residual, const Clamp &clamp,
                                  Eigen::VectorXd x0) const;

    /// Multi-restart minimization: draw `restarts` (at least one) starts via
    /// `sample`, `minimize` each, and return the best. Stops early once a restart
    /// reaches a cost below `epsilon`. `numParams` is the parameter-vector length;
    /// when it is zero there is nothing to optimize and the cost is the single
    /// residual evaluation at the empty vector.
    /// @param seed    seeds the engine handed to `sample` (reproducible draws).
    /// @param epsilon across-restart acceptance threshold for the early stop.
    [[nodiscard]] Result multiRestart(const Residual &residual, const Clamp &clamp,
                                      const Sample &sample, std::size_t numParams,
                                      int restarts, std::uint64_t seed,
                                      double epsilon) const;

  private:
    int maxIterations_;
    double epsilon_;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_LEVENBERGMARQUARDT_H
