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

#include "cobordism/GeometrySynthesizer.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "cobordism/EigenstateSynthesis.h"
#include "cobordism/LevenbergMarquardt.h"
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

GeometrySynthesizer::GeometrySynthesizer(std::shared_ptr<Spacetime> seed)
    : st_(std::move(seed)) {
  if (!st_) throw std::invalid_argument("GeometrySynthesizer: null seed");

  std::vector<std::uint64_t> ids;
  for (const auto v : st_->getVertexList()->toVector())
    if (v != nullptr) ids.push_back(v->getId());
  if (ids.size() < 2)
    throw std::invalid_argument(
        "GeometrySynthesizer: the seed needs at least two vertices (the two "
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
        "GeometrySynthesizer: the seed has no simplex to grow.");
  const auto &tv = top->getVertices();
  topVerts_.assign(tv.begin(), tv.end());
}

std::vector<cd> GeometrySynthesizer::embed(cd c0, cd c1) const {
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

bool GeometrySynthesizer::coneInVertex() {
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

std::size_t GeometrySynthesizer::numVertices() const {
  return st_->getVertexCount();
}

std::size_t GeometrySynthesizer::numEdges() const {
  return st_->getEdgeList()->size();
}

double GeometrySynthesizer::runOptimizer(EigenstateSynthesis &es,
                                         const std::vector<cd> &psi, int restarts,
                                         std::uint64_t seed, double epsilon) const {
  const std::size_t N = psi.size();
  const std::size_t m = es.numEdges();

  // Work on a unit copy: the residual assumes a normalized target (it reads the
  // eigenvalue as λ = p†Lp, valid only for ‖p‖ = 1) and is scale-invariant.
  std::vector<cd> p = psi;
  double nrm = 0.0;
  for (const cd &c : p) nrm += std::norm(c);
  if (nrm > 0.0) {
    const double inv = 1.0 / std::sqrt(nrm);
    for (cd &c : p) c *= inv;
  }

  // Residual: write x = [w; θ] onto the complex (via EigenstateSynthesis) and
  // return g = Lp - λp stacked as [Re; Im] (length 2N). r(ψ) = ‖g‖² is what the
  // Levenberg–Marquardt loop drives to zero. Reuses es.apply (es unmodified).
  const auto residual = [&es, p, m, N](const Eigen::VectorXd &x) -> Eigen::VectorXd {
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
  };

  // Clamp: project [w; θ] back into the §4b search box.
  const auto clamp = [m](const Eigen::VectorXd &x) -> Eigen::VectorXd {
    Eigen::VectorXd c = x;
    for (std::size_t i = 0; i < m; ++i) {
      const auto wi = static_cast<Eigen::Index>(i);
      const auto ti = static_cast<Eigen::Index>(m + i);
      c[wi] = std::min(std::max(c[wi], kWMin), kWMax);
      c[ti] = std::min(std::max(c[ti], -kThetaBound), kThetaBound);
    }
    return c;
  };

  // Sample: draw a fresh restart — w ∈ [kWMin, kWMax], θ ∈ [-π, π]. The
  // distributions are built once and reused so the draw sequence matches the
  // single-rng convention exactly.
  std::uniform_real_distribution<double> wDist(kWMin, kWMax);
  std::uniform_real_distribution<double> tDist(-kPi, kPi);
  const auto sample = [m, wDist, tDist](std::mt19937_64 &rng) mutable -> Eigen::VectorXd {
    Eigen::VectorXd x0(static_cast<Eigen::Index>(2 * m));
    for (std::size_t i = 0; i < m; ++i) {
      x0[static_cast<Eigen::Index>(i)] = wDist(rng);
      x0[static_cast<Eigen::Index>(m + i)] = tDist(rng);
    }
    return x0;
  };

  const LevenbergMarquardt lm(kMaxIterations, epsilon);
  const LevenbergMarquardt::Result best =
      lm.multiRestart(residual, clamp, sample, 2 * m, restarts, seed, epsilon);
  return best.cost;
}

double GeometrySynthesizer::optimize(cd c0, cd c1, int restarts,
                                     std::uint64_t seed) {
  EigenstateSynthesis es(st_);
  const std::vector<cd> psi = embed(c0, c1);
  // epsilon = 0 ⇒ no early-out; runs every restart so the reported value is the
  // true box minimum (the §4b.2 floor on a complex that cannot realize ψ).
  return runOptimizer(es, psi, restarts, seed, 0.0);
}

GeometrySynthesizer::Geo GeometrySynthesizer::synthesize(cd c0, cd c1,
                                                         double epsilon,
                                                         int restarts,
                                                         int maxCones,
                                                         std::uint64_t seed) {
  Geo geo;
  for (int cones = 0;; ++cones) {
    EigenstateSynthesis es(st_);
    const std::vector<cd> psi = embed(c0, c1);
    const double r = runOptimizer(es, psi, restarts,
                                  seed + static_cast<std::uint64_t>(cones),
                                  epsilon);
    const bool accept = (r < epsilon);
    // Grow and retry only while unconverged and within the cone budget — and
    // only if the complex can actually grow (coneInVertex is the capacity gate
    // and is short-circuited otherwise, so it runs exactly when the loop would
    // continue). Stop on convergence, an exhausted budget, or reached capacity.
    const bool grew = !accept && cones < maxCones && coneInVertex();
    if (!grew) {
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
  }
}

}  // namespace tessera::cobordism
