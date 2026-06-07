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

#include "cobordism/BoundaryStateSynthesis.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "cobordism/EigenstateSynthesis.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Fingerprint.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

using cd = std::complex<double>;
using ::tessera::mesh::SimplexPtr;
using ::tessera::mesh::VertexPtr;
using ::tessera::mesh::VertexPtrs;

namespace {

// §4b.3 search box (matching the #133 driver): per-edge magnitudes in
// [w_min, w_max], U(1) phases in [-2π, 2π]. The two-vertex floor w_min²(|c0|²-
// |c1|²)² is taken at w_min, so a positive w_min keeps the floor demonstrably
// nonzero while the box still contains the exact eigenvector solutions on the
// coned-up complexes.
constexpr double kPi = 3.14159265358979323846;
constexpr double kWMin = 0.1;
constexpr double kWMax = 10.0;
constexpr double kThetaBound = 2.0 * kPi;

// Project a flat [w; θ] parameter vector back into the search box.
Eigen::VectorXd clampParams(Eigen::VectorXd x, std::size_t m) {
  for (std::size_t i = 0; i < m; ++i) {
    const auto wi = static_cast<Eigen::Index>(i);
    const auto ti = static_cast<Eigen::Index>(m + i);
    x[wi] = std::min(std::max(x[wi], kWMin), kWMax);
    x[ti] = std::min(std::max(x[ti], -kThetaBound), kThetaBound);
  }
  return x;
}

// Write x = [w; θ] onto the complex (via EigenstateSynthesis) and return the
// real residual vector g = Lp - λp stacked as [Re; Im] (length 2N), p unit.
// r(ψ) = ‖g‖²; this is the least-squares residual the Levenberg–Marquardt loop
// drives to zero. Reuses EigenstateSynthesis::apply (no modification of it).
Eigen::VectorXd residualVector(EigenstateSynthesis &es, const std::vector<cd> &p,
                               const Eigen::VectorXd &x, std::size_t m,
                               std::size_t N) {
  std::vector<double> w(m), th(m);
  for (std::size_t i = 0; i < m; ++i) {
    w[i] = x[static_cast<Eigen::Index>(i)];
    th[i] = x[static_cast<Eigen::Index>(m + i)];
  }
  es.setWeights(w);
  es.setPhases(th);
  const std::vector<cd> Lp = es.apply(p);
  cd lam(0.0, 0.0);
  for (std::size_t i = 0; i < N; ++i) lam += std::conj(p[i]) * Lp[i];
  const double lambda = lam.real();
  Eigen::VectorXd f(static_cast<Eigen::Index>(2 * N));
  for (std::size_t i = 0; i < N; ++i) {
    const cd g = Lp[i] - lambda * p[i];
    f[static_cast<Eigen::Index>(i)] = g.real();
    f[static_cast<Eigen::Index>(N + i)] = g.imag();
  }
  return f;
}

// One bounded Levenberg–Marquardt descent from x0 on the least-squares residual,
// numerical (central-difference) Jacobian. Leaves es written at the returned x;
// reports the achieved cost r = ‖g‖² in costOut.
Eigen::VectorXd lmDescent(EigenstateSynthesis &es, const std::vector<cd> &p,
                          Eigen::VectorXd x, std::size_t m, std::size_t N,
                          double epsilon, double &costOut) {
  const auto nParams = static_cast<Eigen::Index>(2 * m);
  const auto nRows = static_cast<Eigen::Index>(2 * N);
  constexpr double h = 1e-6;
  x = clampParams(std::move(x), m);
  Eigen::VectorXd f = residualVector(es, p, x, m, N);
  double cost = f.squaredNorm();
  double mu = 1e-3;
  for (int iter = 0; iter < 200 && cost > epsilon; ++iter) {
    Eigen::MatrixXd J(nRows, nParams);
    for (Eigen::Index j = 0; j < nParams; ++j) {
      Eigen::VectorXd xp = x;
      Eigen::VectorXd xm = x;
      xp[j] += h;
      xm[j] -= h;
      xp = clampParams(std::move(xp), m);
      xm = clampParams(std::move(xm), m);
      const double denom = xp[j] - xm[j];
      const Eigen::VectorXd fp = residualVector(es, p, xp, m, N);
      const Eigen::VectorXd fm = residualVector(es, p, xm, m, N);
      if (denom != 0.0)
        J.col(j) = (fp - fm) / denom;
      else
        J.col(j).setZero();
    }
    const Eigen::MatrixXd A = J.transpose() * J;
    const Eigen::VectorXd grad = J.transpose() * f;
    bool improved = false;
    for (int tries = 0; tries < 12; ++tries) {
      Eigen::MatrixXd H = A;
      for (Eigen::Index d = 0; d < nParams; ++d) H(d, d) += mu * (A(d, d) + 1e-12);
      const Eigen::VectorXd delta = H.ldlt().solve(-grad);
      const Eigen::VectorXd xNew = clampParams(Eigen::VectorXd(x + delta), m);
      const Eigen::VectorXd fNew = residualVector(es, p, xNew, m, N);
      const double costNew = fNew.squaredNorm();
      if (costNew < cost) {
        x = xNew;
        f = fNew;
        cost = costNew;
        mu = std::max(mu * 0.5, 1e-12);
        improved = true;
        break;
      }
      mu *= 4.0;
      if (mu > 1e12) break;
    }
    if (!improved) break;
  }
  // Leave the complex realized at the returned parameters.
  (void)residualVector(es, p, x, m, N);
  costOut = cost;
  return x;
}

// Non-convex multi-restart minimization of r(ψ) over the edge parameters of the
// (fixed) complex behind es. Leaves es at the best parameters found and returns
// the best residual. Stops early once a restart drives r below epsilon.
double multiRestart(EigenstateSynthesis &es, const std::vector<cd> &psi,
                    int restarts, std::uint64_t seed, double epsilon) {
  const std::size_t N = psi.size();
  const std::size_t m = es.numEdges();

  std::vector<cd> p = psi;  // work on a unit copy (r is scale-invariant)
  double nrm = 0.0;
  for (const cd &c : p) nrm += std::norm(c);
  if (nrm > 0.0) {
    const double inv = 1.0 / std::sqrt(nrm);
    for (cd &c : p) c *= inv;
  }

  if (m == 0) {
    const std::vector<cd> Lp = es.apply(p);
    cd lam(0.0, 0.0);
    for (std::size_t i = 0; i < N; ++i) lam += std::conj(p[i]) * Lp[i];
    const double lambda = lam.real();
    double r = 0.0;
    for (std::size_t i = 0; i < N; ++i) r += std::norm(Lp[i] - lambda * p[i]);
    return r;
  }

  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> wDist(kWMin, kWMax);
  std::uniform_real_distribution<double> tDist(-kPi, kPi);

  double bestCost = std::numeric_limits<double>::infinity();
  Eigen::VectorXd bestX;
  for (int r = 0; r < std::max(restarts, 1); ++r) {
    Eigen::VectorXd x0(static_cast<Eigen::Index>(2 * m));
    for (std::size_t i = 0; i < m; ++i) {
      x0[static_cast<Eigen::Index>(i)] = wDist(rng);
      x0[static_cast<Eigen::Index>(m + i)] = tDist(rng);
    }
    double cost = std::numeric_limits<double>::infinity();
    const Eigen::VectorXd x = lmDescent(es, p, std::move(x0), m, N, epsilon, cost);
    if (cost < bestCost) {
      bestCost = cost;
      bestX = x;
    }
    if (bestCost < epsilon) break;
  }
  (void)residualVector(es, p, bestX, m, N);  // realize the best parameters
  return bestCost;
}

}  // namespace

BoundaryStateSynthesis::BoundaryStateSynthesis(std::shared_ptr<Spacetime> seed)
    : st_(std::move(seed)) {
  if (!st_) throw std::invalid_argument("BoundaryStateSynthesis: null seed");

  std::vector<std::uint64_t> ids;
  for (const auto v : st_->getVertexList()->toVector())
    if (v != nullptr) ids.push_back(v->getId());
  if (ids.size() < 2)
    throw std::invalid_argument(
        "BoundaryStateSynthesis: the seed needs at least two vertices (the two "
        "logical vertices carrying the qubit amplitudes).");
  std::sort(ids.begin(), ids.end());
  logicalId0_ = ids[0];
  logicalId1_ = ids[1];

  // The top simplex coning extends: the largest registered simplex.
  SimplexPtr top = nullptr;
  for (const auto s : st_->getSimplices()) {
    if (s == nullptr) continue;
    if (top == nullptr || s->size() > top->size()) top = s;
  }
  if (top == nullptr)
    throw std::invalid_argument(
        "BoundaryStateSynthesis: the seed has no simplex to grow.");
  const auto &tv = top->getVertices();
  topVerts_.assign(tv.begin(), tv.end());
}

std::vector<cd> BoundaryStateSynthesis::embed(cd c0, cd c1) const {
  std::vector<std::uint64_t> ids;
  for (const auto v : st_->getVertexList()->toVector())
    if (v != nullptr) ids.push_back(v->getId());
  std::sort(ids.begin(), ids.end());
  std::vector<cd> psi(ids.size(), cd(0.0, 0.0));
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (ids[i] == logicalId0_)
      psi[i] = c0;
    else if (ids[i] == logicalId1_)
      psi[i] = c1;
  }
  return psi;
}

bool BoundaryStateSynthesis::coneInVertex() {
  // The cone raises the top simplex's vertex count by one; the Fingerprint
  // caps a simplex at kMax vertices.
  if (topVerts_.size() + 1 > ::tessera::mesh::kMax) return false;

  std::uint64_t maxId = 0;
  for (const auto v : st_->getVertexList()->toVector())
    if (v != nullptr) maxId = std::max(maxId, v->getId());

  // A fresh apex with the largest id — hence an auxiliary, leaving the logical
  // pair as the two smallest ids (the head of ψ).
  const VertexPtr apex = st_->createVertex(maxId + 1);
  VertexPtrs verts = topVerts_;
  verts.push_back(apex);
  // createSimplexTracked auto-builds the new apex→vertex edges (and re-finds the
  // existing ones); this joins the apex to the whole top simplex (Kₙ → Kₙ₊₁).
  st_->createSimplexTracked(verts);
  topVerts_ = std::move(verts);
  return true;
}

std::size_t BoundaryStateSynthesis::numVertices() const {
  return st_->getVertexCount();
}

std::size_t BoundaryStateSynthesis::numEdges() const {
  return st_->getEdgeList()->size();
}

double BoundaryStateSynthesis::optimize(cd c0, cd c1, int restarts,
                                        std::uint64_t seed) {
  EigenstateSynthesis es(st_);
  const std::vector<cd> psi = embed(c0, c1);
  // epsilon = 0 ⇒ no early-out; runs every restart so the reported value is the
  // true box minimum (the §4b.2 floor on a complex that cannot realize ψ).
  return multiRestart(es, psi, restarts, seed, 0.0);
}

BoundaryStateSynthesis::Geo BoundaryStateSynthesis::synthesize(
    cd c0, cd c1, double epsilon, int restarts, int maxCones,
    std::uint64_t seed) {
  Geo geo;
  for (int cones = 0;; ++cones) {
    EigenstateSynthesis es(st_);
    const std::vector<cd> psi = embed(c0, c1);
    const double r = multiRestart(es, psi, restarts,
                                  seed + static_cast<std::uint64_t>(cones),
                                  epsilon);
    const bool accept = (r < epsilon);
    if (accept || cones >= maxCones) {
      geo.converged = accept;
      geo.residual = r;
      geo.eigenvalue = es.rayleigh(psi);
      geo.numVertices = es.order();
      geo.numEdges = es.numEdges();
      geo.conesApplied = cones;
      geo.weights = es.weights();
      geo.phases = es.phases();
      return geo;
    }
    if (!coneInVertex()) {  // cannot grow further (capacity reached)
      geo.converged = false;
      geo.residual = r;
      geo.eigenvalue = es.rayleigh(psi);
      geo.numVertices = es.order();
      geo.numEdges = es.numEdges();
      geo.conesApplied = cones;
      geo.weights = es.weights();
      geo.phases = es.phases();
      return geo;
    }
  }
}

}  // namespace tessera::cobordism
