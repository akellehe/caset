// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/LevenbergMarquardt.h"

#include <algorithm>
#include <utility>

namespace tessera::cobordism {

LevenbergMarquardt::LevenbergMarquardt(int maxIterations, double epsilon)
    : maxIterations_(maxIterations), epsilon_(epsilon) {}

LevenbergMarquardt::Result
LevenbergMarquardt::minimize(const Residual &residual, const Clamp &clamp,
                             Eigen::VectorXd x0) const {
  constexpr double kStep = 1e-6;
  constexpr double kMuInit = 1e-3;
  constexpr double kMuFloor = 1e-12;
  constexpr double kMuCeil = 1e12;
  constexpr double kDiagFloor = 1e-12;
  constexpr int kLineSearchTries = 12;

  Eigen::VectorXd x = clamp(std::move(x0));
  Eigen::VectorXd f = residual(x);
  double cost = f.squaredNorm();
  const auto nParams = static_cast<Eigen::Index>(x.size());
  const auto nRows = static_cast<Eigen::Index>(f.size());
  double mu = kMuInit;
  for (int iteration = 0; iteration < maxIterations_ && cost > epsilon_;
       ++iteration) {
    Eigen::MatrixXd jacobian(nRows, nParams);
    for (Eigen::Index column = 0; column < nParams; ++column) {
      Eigen::VectorXd plus = x;
      Eigen::VectorXd minus = x;
      plus[column] += kStep;
      minus[column] -= kStep;
      plus = clamp(std::move(plus));
      minus = clamp(std::move(minus));
      const double denominator = plus[column] - minus[column];
      const Eigen::VectorXd fPlus = residual(plus);
      const Eigen::VectorXd fMinus = residual(minus);
      if (denominator != 0.0)
        jacobian.col(column) = (fPlus - fMinus) / denominator;
      else
        jacobian.col(column).setZero();
    }
    const Eigen::MatrixXd normal = jacobian.transpose() * jacobian;
    const Eigen::VectorXd gradient = jacobian.transpose() * f;
    bool improved = false;
    for (int trial = 0; trial < kLineSearchTries; ++trial) {
      Eigen::MatrixXd damped = normal;
      for (Eigen::Index diagonal = 0; diagonal < nParams; ++diagonal)
        damped(diagonal, diagonal) +=
            mu * (normal(diagonal, diagonal) + kDiagFloor);
      const Eigen::VectorXd delta = damped.ldlt().solve(-gradient);
      const Eigen::VectorXd candidate = clamp(Eigen::VectorXd(x + delta));
      const Eigen::VectorXd fCandidate = residual(candidate);
      const double candidateCost = fCandidate.squaredNorm();
      if (candidateCost < cost) {
        x = candidate;
        f = fCandidate;
        cost = candidateCost;
        mu = std::max(mu * 0.5, kMuFloor);
        improved = true;
        break;
      }
      mu *= 4.0;
      if (mu > kMuCeil)
        break;
    }
    if (!improved)
      break;
  }
  (void)residual(x);
  return {std::move(x), cost};
}

LevenbergMarquardt::Result
LevenbergMarquardt::multiRestart(const Residual &residual, const Clamp &clamp,
                                 const Sample &sample, std::size_t numParams,
                                 int restarts, std::uint64_t seed,
                                 double epsilon) const {
  if (numParams == 0) {
    const Eigen::VectorXd empty(0);
    return {empty, residual(empty).squaredNorm()};
  }

  std::mt19937_64 randomEngine(seed);
  Result best;
  for (int restart = 0; restart < std::max(restarts, 1); ++restart) {
    Result trial = minimize(residual, clamp, sample(randomEngine));
    if (trial.cost < best.cost)
      best = std::move(trial);
    if (best.cost < epsilon)
      break;
  }
  (void)residual(best.parameters);
  return best;
}

}  // namespace tessera::cobordism
