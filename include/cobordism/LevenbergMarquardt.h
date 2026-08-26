// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_LEVENBERGMARQUARDT_H
#define TESSERA_COBORDISM_LEVENBERGMARQUARDT_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>

#include <Eigen/Dense>

namespace tessera::cobordism {

/// A bounded least-squares Levenberg-Marquardt solver with reproducible random
/// restarts. Restored from the fixed-boundary realizability implementation used
/// by the 2026-06-24 cobordism report.
class LevenbergMarquardt {
  public:
    using Residual = std::function<Eigen::VectorXd(const Eigen::VectorXd &)>;
    using Clamp = std::function<Eigen::VectorXd(const Eigen::VectorXd &)>;
    using Sample = std::function<Eigen::VectorXd(std::mt19937_64 &)>;

    struct Result {
      Eigen::VectorXd parameters{};
      double cost{std::numeric_limits<double>::infinity()};
    };

    explicit LevenbergMarquardt(int maxIterations = 200,
                                double epsilon = 0.0);

    [[nodiscard]] Result minimize(const Residual &residual, const Clamp &clamp,
                                  Eigen::VectorXd x0) const;

    [[nodiscard]] Result multiRestart(
        const Residual &residual, const Clamp &clamp, const Sample &sample,
        std::size_t numParams, int restarts, std::uint64_t seed,
        double epsilon) const;

  private:
    int maxIterations_;
    double epsilon_;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_LEVENBERGMARQUARDT_H
