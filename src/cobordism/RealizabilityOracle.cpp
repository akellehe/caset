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
  // Degree-aware fill. At k = 0 the target is matched as an EIGENVECTOR of
  // L = D - A on the first L sorted-id vertices (§4b), tuning interior weights +
  // phases. At k >= 1 the target is matched as the boundary block of a HARMONIC
  // form (ker L_k) on the ∂W cells, tuning interior weights only — the real
  // metric L_k ignores the U(1) phases (§5.2 / the DW bridge, #176).
  const int degree = es.degree();
  const bool harmonic = degree >= 1;
  const bool tunePhases = degree == 0;
  const std::size_t L = target.size();  // the pinned boundary-support length
  double bestR = std::numeric_limits<double>::infinity();
  conesApplied = 0;

  for (int cone = 0;; ++cone) {
    const std::size_t dim = es.dimension();       // psi length (|V| or |C_k|)
    const std::size_t m = es.numInteriorEdges();  // free interior edges (weights)

    // The psi-component support: targetIdx carries the fixed target, auxIdx the
    // free auxiliary amplitudes. k=0 uses the first-L convention; k>=1 the ∂W
    // cells (boundaryStateIndices) so the target lands on the geometric boundary.
    std::vector<std::size_t> targetIdx, auxIdx;
    if (degree == 0) {
      targetIdx.reserve(L);
      for (std::size_t i = 0; i < L; ++i) targetIdx.push_back(i);
      for (std::size_t i = L; i < dim; ++i) auxIdx.push_back(i);
    } else {
      targetIdx = es.boundaryStateIndices();
      auxIdx = es.interiorStateIndices();
    }
    if (targetIdx.size() != L)
      throw std::invalid_argument(
          "RealizabilityOracle: target length does not match the bulk's boundary "
          "support at this degree");
    const std::size_t nAux = auxIdx.size();
    const std::size_t nPhase = tunePhases ? m : 0;
    const std::size_t auxBase = m + nPhase;   // x offset of the aux re/im block
    const std::size_t nParams = auxBase + 2 * nAux;

    // Assemble ψ from a parameter vector: the boundary support is the fixed
    // target; the auxiliary amplitudes (interior / coned-in cells) come from the
    // tail of x (real, imag per free component).
    const auto buildPsi = [&target, &targetIdx, &auxIdx, L, nAux, auxBase,
                           dim](const Eigen::VectorXd &x) -> std::vector<cd> {
      std::vector<cd> psi(dim, cd(0.0, 0.0));
      for (std::size_t i = 0; i < L; ++i) psi[targetIdx[i]] = target[i];
      for (std::size_t k = 0; k < nAux; ++k) {
        const auto re = static_cast<Eigen::Index>(auxBase + 2 * k);
        psi[auxIdx[k]] = cd(x[re], x[re + 1]);
      }
      return psi;
    };

    // Residual: write the interior weights (and, at k=0, phases) onto ∂W's
    // complement (∂W itself is never touched), normalize ψ, and return the
    // stacked residual ([Re; Im], length 2·dim) whose ‖·‖² the
    // Levenberg–Marquardt loop drives to zero. k=0: g = Lψ - λψ (eigenvector).
    // k>=1: g = L_kψ (harmonic, ker L_k).
    const auto residual = [&es, &buildPsi, m, nPhase, dim, harmonic,
                           tunePhases](const Eigen::VectorXd &x) -> Eigen::VectorXd {
      if (m > 0) {
        std::vector<double> w(m);
        for (std::size_t i = 0; i < m; ++i)
          w[i] = x[static_cast<Eigen::Index>(i)];
        es.setInteriorWeights(w);
        if (tunePhases) {
          std::vector<double> th(m);
          for (std::size_t i = 0; i < m; ++i)
            th[i] = x[static_cast<Eigen::Index>(m + i)];
          es.setInteriorPhases(th);
        }
      }
      (void)nPhase;
      std::vector<cd> psi = buildPsi(x);
      double nrm = 0.0;
      for (const cd &z : psi) nrm += std::norm(z);
      if (nrm > 0.0) {
        const double inv = 1.0 / std::sqrt(nrm);
        for (cd &z : psi) z *= inv;
      }
      const std::vector<cd> Lp = es.apply(psi);
      double lambda = 0.0;
      if (!harmonic) {
        cd lam(0.0, 0.0);
        for (std::size_t i = 0; i < dim; ++i) lam += std::conj(psi[i]) * Lp[i];
        lambda = lam.real();
      }
      Eigen::VectorXd f(static_cast<Eigen::Index>(2 * dim));
      for (std::size_t i = 0; i < dim; ++i) {
        const cd g = harmonic ? Lp[i] : (Lp[i] - lambda * psi[i]);
        f[static_cast<Eigen::Index>(i)] = g.real();
        f[static_cast<Eigen::Index>(dim + i)] = g.imag();
      }
      return f;
    };

    // Clamp interior weights into the §4b box, the phases (k=0 only) into
    // [-kThetaBound, kThetaBound], and the auxiliary amplitudes into
    // [-kAuxBound, kAuxBound].
    const auto clamp = [m, nPhase, nAux, auxBase,
                        tunePhases](const Eigen::VectorXd &x) -> Eigen::VectorXd {
      Eigen::VectorXd c = x;
      for (std::size_t i = 0; i < m; ++i) {
        const auto wi = static_cast<Eigen::Index>(i);
        c[wi] = std::min(std::max(c[wi], kWMin), kWMax);
      }
      if (tunePhases)
        for (std::size_t i = 0; i < nPhase; ++i) {
          const auto ti = static_cast<Eigen::Index>(m + i);
          c[ti] = std::min(std::max(c[ti], -kThetaBound), kThetaBound);
        }
      for (std::size_t k = 0; k < 2 * nAux; ++k) {
        const auto ai = static_cast<Eigen::Index>(auxBase + k);
        c[ai] = std::min(std::max(c[ai], -kAuxBound), kAuxBound);
      }
      return c;
    };

    // Sample a restart: w ∈ [kWMin, kWMax], θ ∈ [-π, π] (k=0), aux ∈ [-1, 1].
    std::uniform_real_distribution<double> wDist(kWMin, kWMax);
    std::uniform_real_distribution<double> tDist(-kPi, kPi);
    std::uniform_real_distribution<double> aDist(-1.0, 1.0);
    const auto sample = [m, nPhase, nAux, auxBase, tunePhases, wDist, tDist, aDist](
                            std::mt19937_64 &rng) mutable -> Eigen::VectorXd {
      Eigen::VectorXd x0(static_cast<Eigen::Index>(auxBase + 2 * nAux));
      for (std::size_t i = 0; i < m; ++i)
        x0[static_cast<Eigen::Index>(i)] = wDist(rng);
      if (tunePhases)
        for (std::size_t i = 0; i < nPhase; ++i)
          x0[static_cast<Eigen::Index>(m + i)] = tDist(rng);
      for (std::size_t k = 0; k < 2 * nAux; ++k)
        x0[static_cast<Eigen::Index>(auxBase + k)] = aDist(rng);
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

RealizabilityOracle::Verdict RealizabilityOracle::decideBoundaryHarmonic(
    const std::vector<cd> &target, double epsilon, int restarts, int maxCones,
    std::uint64_t seed) {
  EigenstateSynthesis es(bulk_, /*degree=*/1);
  const std::size_t nBoundary = es.boundaryStateIndices().size();
  if (target.size() != nBoundary)
    throw std::invalid_argument(
        "RealizabilityOracle::decideBoundaryHarmonic: the target boundary "
        "harmonic length (" +
        std::to_string(target.size()) +
        ") must equal the number of dW boundary edges (" +
        std::to_string(nBoundary) + ")");

  Verdict v;
  v.target = target;
  const double r = fillInterior(es, target, epsilon, restarts, maxCones, seed,
                                v.state, v.conesApplied);
  v.residual = r;
  v.realizable = (r < epsilon);
  v.floor = v.realizable ? 0.0 : r;
  v.eigenvalue = es.rayleigh(v.state);
  v.interiorVertexCount = es.interiorVertexCount();
  v.witness = bulk_;
  return v;
}

}  // namespace tessera::cobordism
