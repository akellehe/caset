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

#include "cobordism/RealizabilityOracle.h"

#include <Eigen/Dense>

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "cobordism/EigenstateSynthesis.h"
#include "cobordism/LevenbergMarquardt.h"
#include "quantum/ChoiJamiolkowski.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

using cd = std::complex<double>;

RealizabilityOracle::RealizabilityOracle(std::shared_ptr<Spacetime> bulk)
    : bulk_(std::move(bulk)) {
  if (!bulk_) throw std::invalid_argument("RealizabilityOracle: null bulk");
}

std::vector<cd> RealizabilityOracle::bend(const std::vector<cd> &U, int dA,
                                          int dB) {
  // Choi–Jamiołkowski bend: vec(U) = Σ_{ij} U_{ij} |i⟩_A ⊗ |j⟩_B, the row-major
  // flatten (#vectorize validates the dims and U.size() == dA·dB).
  std::vector<cd> psi = ::tessera::quantum::ChoiJamiolkowski::vectorize(U, dA, dB);
  double nrm = 0.0;
  for (const cd &z : psi) nrm += std::norm(z);
  if (nrm > 0.0) {
    const double inv = 1.0 / std::sqrt(nrm);
    for (cd &z : psi) z *= inv;
  }
  return psi;
}

double RealizabilityOracle::fillInterior(EigenstateSynthesis &es,
                                         const std::vector<cd> &target,
                                         double epsilon, int restarts,
                                         int maxCones, std::uint64_t seed,
                                         std::vector<cd> &witnessOut,
                                         int &conesApplied) const {
  const std::size_t L = target.size();  // the pinned boundary-support length
  double bestR = std::numeric_limits<double>::infinity();
  conesApplied = 0;

  for (int cone = 0;; ++cone) {
    const std::size_t order = es.order();       // total vertices this pass
    const std::size_t m = es.numInteriorEdges();  // free interior edges
    const std::size_t nAux = order - L;           // free auxiliary amplitudes
    const std::size_t nParams = 2 * m + 2 * nAux;

    // Assemble ψ from a parameter vector: the boundary support is the fixed
    // target (first L sorted-id vertices); the auxiliary amplitudes (interior /
    // coned-in apices) come from the tail of x (real, imag per vertex).
    const auto buildPsi = [&target, L, m, nAux,
                           order](const Eigen::VectorXd &x) -> std::vector<cd> {
      std::vector<cd> psi = target;
      psi.resize(order, cd(0.0, 0.0));
      for (std::size_t k = 0; k < nAux; ++k) {
        const auto re = static_cast<Eigen::Index>(2 * m + 2 * k);
        psi[L + k] = cd(x[re], x[re + 1]);
      }
      return psi;
    };

    // Residual: write the interior weights/phases onto ∂W's complement (∂W
    // itself is never touched), normalize the assembled ψ, and return the
    // stacked g = Lψ - λψ ([Re; Im], length 2·order) whose ‖g‖² is r(ψ) — the
    // §4b residual the Levenberg–Marquardt loop drives to zero.
    const auto residual =
        [&es, &buildPsi, m, order](const Eigen::VectorXd &x) -> Eigen::VectorXd {
      if (m > 0) {
        std::vector<double> w(m), th(m);
        for (std::size_t i = 0; i < m; ++i) {
          w[i] = x[static_cast<Eigen::Index>(i)];
          th[i] = x[static_cast<Eigen::Index>(m + i)];
        }
        es.setInteriorWeights(w);
        es.setInteriorPhases(th);
      }
      std::vector<cd> psi = buildPsi(x);
      double nrm = 0.0;
      for (const cd &z : psi) nrm += std::norm(z);
      if (nrm > 0.0) {
        const double inv = 1.0 / std::sqrt(nrm);
        for (cd &z : psi) z *= inv;
      }
      const std::vector<cd> Lp = es.apply(psi);
      cd lam(0.0, 0.0);
      for (std::size_t i = 0; i < order; ++i) lam += std::conj(psi[i]) * Lp[i];
      const double lambda = lam.real();
      Eigen::VectorXd f(static_cast<Eigen::Index>(2 * order));
      for (std::size_t i = 0; i < order; ++i) {
        const cd g = Lp[i] - lambda * psi[i];
        f[static_cast<Eigen::Index>(i)] = g.real();
        f[static_cast<Eigen::Index>(order + i)] = g.imag();
      }
      return f;
    };

    // Clamp interior weights/phases into the §4b box and the auxiliary
    // amplitudes into [-kAuxBound, kAuxBound].
    const auto clamp = [m, nAux](const Eigen::VectorXd &x) -> Eigen::VectorXd {
      Eigen::VectorXd c = x;
      for (std::size_t i = 0; i < m; ++i) {
        const auto wi = static_cast<Eigen::Index>(i);
        const auto ti = static_cast<Eigen::Index>(m + i);
        c[wi] = std::min(std::max(c[wi], kWMin), kWMax);
        c[ti] = std::min(std::max(c[ti], -kThetaBound), kThetaBound);
      }
      for (std::size_t k = 0; k < 2 * nAux; ++k) {
        const auto ai = static_cast<Eigen::Index>(2 * m + k);
        c[ai] = std::min(std::max(c[ai], -kAuxBound), kAuxBound);
      }
      return c;
    };

    // Sample a restart: w ∈ [kWMin, kWMax], θ ∈ [-π, π], aux ∈ [-1, 1]. Built
    // once and captured (the single-rng draw convention GeometrySynthesizer uses).
    std::uniform_real_distribution<double> wDist(kWMin, kWMax);
    std::uniform_real_distribution<double> tDist(-kPi, kPi);
    std::uniform_real_distribution<double> aDist(-1.0, 1.0);
    const auto sample = [m, nAux, wDist, tDist,
                         aDist](std::mt19937_64 &rng) mutable -> Eigen::VectorXd {
      Eigen::VectorXd x0(static_cast<Eigen::Index>(2 * m + 2 * nAux));
      for (std::size_t i = 0; i < m; ++i) {
        x0[static_cast<Eigen::Index>(i)] = wDist(rng);
        x0[static_cast<Eigen::Index>(m + i)] = tDist(rng);
      }
      for (std::size_t k = 0; k < 2 * nAux; ++k)
        x0[static_cast<Eigen::Index>(2 * m + k)] = aDist(rng);
      return x0;
    };

    const LevenbergMarquardt lm(kMaxIterations, epsilon);
    const LevenbergMarquardt::Result best =
        lm.multiRestart(residual, clamp, sample, nParams, restarts,
                        seed + static_cast<std::uint64_t>(cone), epsilon);

    // Leave the complex realized at the best parameters and read off the unit
    // witness state there (multiRestart leaves `residual` evaluated last at
    // best.parameters; re-evaluate so the witness ψ matches the live complex).
    residual(best.parameters);
    witnessOut = buildPsi(best.parameters);
    double wn = 0.0;
    for (const cd &z : witnessOut) wn += std::norm(z);
    if (wn > 0.0) {
      const double inv = 1.0 / std::sqrt(wn);
      for (cd &z : witnessOut) z *= inv;
    }
    bestR = best.cost;

    // Stop on convergence, an exhausted budget, or a complex that cannot grow
    // (growInterior is the capacity/structure gate; it leaves ∂W byte-fixed).
    if (bestR < epsilon) break;
    if (cone >= maxCones) break;
    if (!es.growInterior(seed + 1000 + static_cast<std::uint64_t>(cone))) break;
    ++conesApplied;
  }
  return bestR;
}

RealizabilityOracle::Verdict RealizabilityOracle::decide(
    const std::vector<cd> &U, int dA, int dB, double epsilon, int restarts,
    int maxCones, std::uint64_t seed) {
  const std::vector<cd> target = bend(U, dA, dB);  // ψ_U, length dA·dB

  EigenstateSynthesis es(bulk_);
  if (es.order() < target.size())
    throw std::invalid_argument(
        "RealizabilityOracle: the bulk has fewer vertices than the bent target "
        "needs on its output-boundary support (dA*dB).");

  Verdict v;
  v.target = target;
  const double r =
      fillInterior(es, target, epsilon, restarts, maxCones, seed, v.state,
                   v.conesApplied);
  v.residual = r;
  v.realizable = (r < epsilon);
  v.floor = v.realizable ? 0.0 : r;
  v.eigenvalue = es.rayleigh(v.state);
  v.interiorVertexCount = es.interiorVertexCount();
  v.witness = bulk_;
  return v;
}

}  // namespace tessera::cobordism
