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

#include "cobordism/EigenstateSynthesis.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

using cd = std::complex<double>;

EigenstateSynthesis::EigenstateSynthesis(std::shared_ptr<Spacetime> st)
    : st_(st), laplacian_(std::move(st)) {
  if (!st_) return;

  // Stable vertex order (sorted id), matching HodgeLaplacian's k=0 indexing so a
  // psi entry aligns with the operator's row/column for that vertex.
  std::unordered_set<std::uint64_t> idset;
  for (const auto v : st_->getVertexList()->toVector())
    if (v != nullptr) idset.insert(v->getId());
  order_ = idset.size();

  // Stable edge order: the tunable edges in EdgeList order — those that actually
  // carry weight in L = D - A (both endpoints present in the vertex set, not a
  // self-loop). This is exactly HodgeLaplacian::assemble's edge filter, so the
  // {w_ij, theta_ij} we expose are the parameters the Laplacian reads.
  for (const auto e : st_->getEdgeList()->toVector()) {
    if (e == nullptr) continue;
    const auto s = e->getSource();
    const auto t = e->getTarget();
    if (s == nullptr || t == nullptr) continue;
    if (s->getId() == t->getId()) continue;
    if (idset.find(s->getId()) == idset.end() ||
        idset.find(t->getId()) == idset.end())
      continue;
    edges_.push_back(e);
  }
}

std::vector<cd> EigenstateSynthesis::apply(const std::vector<cd> &psi) const {
  const std::size_t N = order_;
  if (psi.size() != N)
    throw std::runtime_error(
        "EigenstateSynthesis::apply: psi has length " +
        std::to_string(psi.size()) + ", expected " + std::to_string(N));
  std::vector<cd> out(N, cd(0.0, 0.0));
  if (N == 0) return out;
  // L = D - A reassembled from the live edge weights/phases (the k=0 magnitude
  // convention). The matrix path does not consult the eigendecomposition cache,
  // so repeated perturb-then-query is honest.
  const std::vector<cd> L = laplacian_.laplacian(0);
  for (std::size_t i = 0; i < N; ++i) {
    cd acc(0.0, 0.0);
    for (std::size_t j = 0; j < N; ++j) acc += L[i * N + j] * psi[j];
    out[i] = acc;
  }
  return out;
}

double EigenstateSynthesis::residual(const std::vector<cd> &psi) const {
  const std::size_t N = order_;
  if (psi.size() != N)
    throw std::runtime_error(
        "EigenstateSynthesis::residual: psi has length " +
        std::to_string(psi.size()) + ", expected " + std::to_string(N));
  if (N == 0) return 0.0;

  // Normalize: r and the eigenvector condition are scale-invariant, and the spec
  // writes r for a unit target.
  double nrm2 = 0.0;
  for (const cd &c : psi) nrm2 += std::norm(c);
  if (nrm2 <= 0.0) return 0.0;
  const double inv = 1.0 / std::sqrt(nrm2);
  std::vector<cd> p(N);
  for (std::size_t i = 0; i < N; ++i) p[i] = psi[i] * inv;

  const std::vector<cd> Lp = apply(p);
  // lambda = p^dagger L p (real for Hermitian L; take the real part).
  cd lam(0.0, 0.0);
  for (std::size_t i = 0; i < N; ++i) lam += std::conj(p[i]) * Lp[i];
  const double lambda = lam.real();

  // r = || L p - lambda p ||^2 = ||(I - p p^dagger) L p||^2 (p unit).
  double r = 0.0;
  for (std::size_t i = 0; i < N; ++i) r += std::norm(Lp[i] - lambda * p[i]);
  return r;
}

double EigenstateSynthesis::rayleigh(const std::vector<cd> &psi) const {
  const std::size_t N = order_;
  if (psi.size() != N)
    throw std::runtime_error(
        "EigenstateSynthesis::rayleigh: psi has length " +
        std::to_string(psi.size()) + ", expected " + std::to_string(N));
  if (N == 0) return 0.0;

  const std::vector<cd> Lp = apply(psi);
  cd num(0.0, 0.0);
  double den = 0.0;
  for (std::size_t i = 0; i < N; ++i) {
    num += std::conj(psi[i]) * Lp[i];
    den += std::norm(psi[i]);
  }
  if (den <= 0.0) return 0.0;
  return num.real() / den;
}

std::vector<double> EigenstateSynthesis::weights() const {
  std::vector<double> w;
  w.reserve(edges_.size());
  for (const auto e : edges_) w.push_back(e->getSquaredLength());
  return w;
}

std::vector<double> EigenstateSynthesis::phases() const {
  std::vector<double> th;
  th.reserve(edges_.size());
  for (const auto e : edges_) th.push_back(e->getPhase());
  return th;
}

void EigenstateSynthesis::setWeights(const std::vector<double> &w) {
  if (w.size() != edges_.size())
    throw std::runtime_error(
        "EigenstateSynthesis::setWeights: got " + std::to_string(w.size()) +
        " weights, expected " + std::to_string(edges_.size()));
  for (std::size_t i = 0; i < edges_.size(); ++i)
    edges_[i]->setSquaredLength(w[i]);
}

void EigenstateSynthesis::setPhases(const std::vector<double> &theta) {
  if (theta.size() != edges_.size())
    throw std::runtime_error(
        "EigenstateSynthesis::setPhases: got " + std::to_string(theta.size()) +
        " phases, expected " + std::to_string(edges_.size()));
  for (std::size_t i = 0; i < edges_.size(); ++i)
    edges_[i]->setPhase(theta[i]);
}

}  // namespace tessera::cobordism
