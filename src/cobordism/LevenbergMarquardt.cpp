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

#include "cobordism/LevenbergMarquardt.h"

#include <algorithm>
#include <utility>

namespace tessera::cobordism {

LevenbergMarquardt::LevenbergMarquardt(int maxIterations, double epsilon)
    : maxIterations_(maxIterations), epsilon_(epsilon) {}

LevenbergMarquardt::Result LevenbergMarquardt::minimize(const Residual &residual,
                                                        const Clamp &clamp,
                                                        Eigen::VectorXd x0) const {
  constexpr double kStep = 1e-6;        // central-difference step
  constexpr double kMuInit = 1e-3;      // initial LM damping
  constexpr double kMuFloor = 1e-12;    // smallest damping after a shrink
  constexpr double kMuCeil = 1e12;      // give up the line search past this
  constexpr double kDiagFloor = 1e-12;  // guards a zero J^T J diagonal
  constexpr int kLineSearchTries = 12;  // damping grows up to this many times

  Eigen::VectorXd x = clamp(std::move(x0));
  Eigen::VectorXd f = residual(x);
  double cost = f.squaredNorm();
  const auto nParams = static_cast<Eigen::Index>(x.size());
  const auto nRows = static_cast<Eigen::Index>(f.size());
  double mu = kMuInit;
  for (int iter = 0; iter < maxIterations_ && cost > epsilon_; ++iter) {
    Eigen::MatrixXd J(nRows, nParams);
    for (Eigen::Index j = 0; j < nParams; ++j) {
      Eigen::VectorXd xp = x;
      Eigen::VectorXd xm = x;
      xp[j] += kStep;
      xm[j] -= kStep;
      xp = clamp(std::move(xp));
      xm = clamp(std::move(xm));
      const double denom = xp[j] - xm[j];
      const Eigen::VectorXd fp = residual(xp);
      const Eigen::VectorXd fm = residual(xm);
      if (denom != 0.0)
        J.col(j) = (fp - fm) / denom;
      else
        J.col(j).setZero();
    }
    const Eigen::MatrixXd A = J.transpose() * J;
    const Eigen::VectorXd grad = J.transpose() * f;
    bool improved = false;
    for (int tries = 0; tries < kLineSearchTries; ++tries) {
      Eigen::MatrixXd H = A;
      for (Eigen::Index d = 0; d < nParams; ++d) H(d, d) += mu * (A(d, d) + kDiagFloor);
      const Eigen::VectorXd delta = H.ldlt().solve(-grad);
      const Eigen::VectorXd xNew = clamp(Eigen::VectorXd(x + delta));
      const Eigen::VectorXd fNew = residual(xNew);
      const double costNew = fNew.squaredNorm();
      if (costNew < cost) {
        x = xNew;
        f = fNew;
        cost = costNew;
        mu = std::max(mu * 0.5, kMuFloor);
        improved = true;
        break;
      }
      mu *= 4.0;
      if (mu > kMuCeil) break;
    }
    if (!improved) break;
  }
  (void)residual(x);  // leave the residual functor realized at the returned x
  return {std::move(x), cost};
}

LevenbergMarquardt::Result LevenbergMarquardt::multiRestart(
    const Residual &residual, const Clamp &clamp, const Sample &sample,
    std::size_t numParams, int restarts, std::uint64_t seed,
    double epsilon) const {
  if (numParams == 0) {
    const Eigen::VectorXd empty(0);
    return {empty, residual(empty).squaredNorm()};
  }

  std::mt19937_64 rng(seed);
  Result best;  // cost = +inf
  for (int r = 0; r < std::max(restarts, 1); ++r) {
    Result trial = minimize(residual, clamp, sample(rng));
    if (trial.cost < best.cost) best = std::move(trial);
    if (best.cost < epsilon) break;
  }
  (void)residual(best.parameters);  // realize the residual functor at the best x
  return best;
}

}  // namespace tessera::cobordism
