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

#include "cobordism/HodgeLaplacian.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

using cd = std::complex<double>;

HodgeLaplacian::HodgeLaplacian(std::shared_ptr<Spacetime> st) : st_(std::move(st)) {
  if (!st_) return;
  // Stable vertex order: sort by id, then id -> 0..N-1.
  const auto &verts = st_->getVertexList()->toVector();
  ids_.reserve(verts.size());
  for (const auto v : verts)
    if (v != nullptr) ids_.push_back(v->getId());
  std::sort(ids_.begin(), ids_.end());
  order_ = ids_.size();
  idToIndex_.reserve(order_);
  for (std::size_t i = 0; i < order_; ++i) idToIndex_[ids_[i]] = i;
}

void HodgeLaplacian::requireStageOne(int k) {
  if (k != 0)
    throw std::runtime_error(
        "HodgeLaplacian: Stage 2: L_k not yet implemented (only k=0, the graph "
        "Laplacian, is available); requested k=" + std::to_string(k));
}

void HodgeLaplacian::assemble(std::vector<cd> &A, std::vector<double> &D) const {
  const std::size_t N = order_;
  A.assign(N * N, cd(0.0, 0.0));
  D.assign(N, 0.0);
  if (N == 0 || !st_) return;

  for (const EdgePtr e : st_->getEdgeList()->toVector()) {
    if (e == nullptr) continue;
    const VertexPtr s = e->getSource();
    const VertexPtr t = e->getTarget();
    if (s == nullptr || t == nullptr) continue;
    const auto is = idToIndex_.find(s->getId());
    const auto it = idToIndex_.find(t->getId());
    if (is == idToIndex_.end() || it == idToIndex_.end()) continue;
    const std::size_t i = is->second;
    const std::size_t j = it->second;
    if (i == j) continue;  // a simplicial complex carries no self-loops

    const double w = e->getSquaredLength();      // signed magnitude
    const double phase = e->getPhase();          // U(1) connection on src->tgt

    // Degree uses the magnitude convention D_ii = sum |squaredLength| over
    // incident edges (phase-independent; keeps L Hermitian and e^{-iLt} unitary).
    D[i] += std::abs(w);
    D[j] += std::abs(w);

    // A_ij = sum squaredLength * e^{i*phase}; the reverse orientation negates
    // the phase, so A = A^dagger.
    const cd z = w * std::exp(cd(0.0, phase));
    A[i * N + j] += z;
    A[j * N + i] += std::conj(z);
  }
}

std::vector<cd> HodgeLaplacian::adjacency() const {
  std::vector<cd> A;
  std::vector<double> D;
  assemble(A, D);
  return A;
}

std::vector<double> HodgeLaplacian::degree() const {
  std::vector<cd> A;
  std::vector<double> D;
  assemble(A, D);
  return D;
}

std::vector<cd> HodgeLaplacian::laplacian(int k) const {
  requireStageOne(k);
  std::vector<cd> A;
  std::vector<double> D;
  assemble(A, D);
  const std::size_t N = order_;
  // L = D - A: off-diagonal entries are -A_ij; the diagonal carries D_ii (the
  // adjacency has no diagonal in a complex without self-loops).
  std::vector<cd> L(N * N, cd(0.0, 0.0));
  for (std::size_t idx = 0; idx < A.size(); ++idx) L[idx] = -A[idx];
  for (std::size_t i = 0; i < N; ++i) L[i * N + i] += D[i];
  return L;
}

void HodgeLaplacian::ensureDecomposition() const {
  if (decomposed_) return;
  const int N = static_cast<int>(order_);
  evals_.assign(static_cast<std::size_t>(N), 0.0);
  evecs_.assign(static_cast<std::size_t>(N) * N, cd(0.0, 0.0));
  if (N == 0) {
    decomposed_ = true;
    return;
  }

  std::vector<cd> A;
  std::vector<double> D;
  assemble(A, D);
  Eigen::MatrixXcd L(N, N);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      L(i, j) = -A[static_cast<std::size_t>(i) * N + j];
  for (int i = 0; i < N; ++i) L(i, i) += D[static_cast<std::size_t>(i)];

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(L);
  // Eigen yields real eigenvalues in ascending order and orthonormal columns.
  const Eigen::VectorXd &lam = es.eigenvalues();
  const Eigen::MatrixXcd &V = es.eigenvectors();
  for (int i = 0; i < N; ++i) evals_[static_cast<std::size_t>(i)] = lam[i];
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      evecs_[static_cast<std::size_t>(i) * N + j] = V(i, j);
  decomposed_ = true;
}

bool HodgeLaplacian::isHermitian(double tol) const {
  const int N = static_cast<int>(order_);
  if (N == 0) return true;
  const std::vector<cd> Lflat = laplacian(0);
  Eigen::MatrixXcd L(N, N);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      L(i, j) = Lflat[static_cast<std::size_t>(i) * N + j];
  return (L - L.adjoint()).norm() <= tol;
}

double HodgeLaplacian::unitarityResidual(double t) const {
  ensureDecomposition();
  const int N = static_cast<int>(order_);
  if (N == 0) return 0.0;

  Eigen::MatrixXcd V(N, N);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      V(i, j) = evecs_[static_cast<std::size_t>(i) * N + j];

  Eigen::VectorXcd phases(N);
  for (int k = 0; k < N; ++k)
    phases[k] = std::exp(cd(0.0, -evals_[static_cast<std::size_t>(k)] * t));

  // U = e^{-iLt} = V diag(e^{-i lambda t}) V^dagger.
  const Eigen::MatrixXcd U = V * phases.asDiagonal() * V.adjoint();
  const Eigen::MatrixXcd resid =
      U * U.adjoint() - Eigen::MatrixXcd::Identity(N, N);
  return resid.norm();
}

std::vector<double> HodgeLaplacian::eigenvalues(int k) const {
  requireStageOne(k);
  ensureDecomposition();
  return evals_;
}

std::vector<cd> HodgeLaplacian::eigenvectors(int k) const {
  requireStageOne(k);
  ensureDecomposition();
  return evecs_;
}

std::vector<cd> HodgeLaplacian::harmonics(int k, double tol) const {
  requireStageOne(k);
  ensureDecomposition();
  const std::size_t N = order_;
  if (N == 0) return {};

  std::vector<std::size_t> cols;
  for (std::size_t j = 0; j < N; ++j)
    if (std::abs(evals_[j]) < tol) cols.push_back(j);

  const std::size_t M = cols.size();
  std::vector<cd> H(N * M, cd(0.0, 0.0));
  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t jj = 0; jj < M; ++jj)
      H[i * M + jj] = evecs_[i * N + cols[jj]];
  return H;
}

}  // namespace tessera::cobordism
