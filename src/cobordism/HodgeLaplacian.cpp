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
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cobordism/ChainComplex.h"
#include "cobordism/Cochain.h"
#include "cobordism/Spectrum.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

using cd = std::complex<double>;

namespace {

using Face = std::vector<std::uint64_t>;  // sorted vertex ids

// Sorted vertex ids of a simplex — the homological reference ordering, identical
// to ChainComplex's. Distinct simplices have distinct tuples, so sorting by it is
// a canonical total order that reproduces ChainComplex's column order exactly.
Face sortedIds(const SimplexPtr &s) {
  Face ids;
  for (const auto &v : s->getVertices()) ids.push_back(v->getId());
  std::sort(ids.begin(), ids.end());
  return ids;
}

// Face-closure of the complex, bucketed by dimension and ordered within each
// dimension by sorted vertex ids — the same BFS-over-getFacets construction
// ChainComplex uses, so faces[k][j] is the simplex behind column j of
// boundaryMatrix(k). Returns the SimplexPtrs (needed for their volumes).
std::vector<std::vector<SimplexPtr>> orderedFaces(const Spacetime &K) {
  std::map<int, std::map<Face, SimplexPtr>> byDim;  // dim -> (sorted ids -> simplex)
  std::unordered_set<std::uint64_t> seen;
  std::vector<SimplexPtr> stack(K.getSimplices().begin(), K.getSimplices().end());
  while (!stack.empty()) {
    SimplexPtr s = stack.back();
    stack.pop_back();
    if (s == nullptr) continue;
    if (!seen.insert(s->fingerprint.fingerprint()).second) continue;
    const int k = static_cast<int>(s->size()) - 1;
    byDim[k][sortedIds(s)] = s;
    if (k >= 1)
      for (const auto &f : s->getFacets()) stack.push_back(f);
  }
  const int n = byDim.empty() ? -1 : byDim.rbegin()->first;
  std::vector<std::vector<SimplexPtr>> faces(static_cast<std::size_t>(n + 1));
  for (int k = 0; k <= n; ++k)
    for (const auto &[face, s] : byDim[k]) faces[static_cast<std::size_t>(k)].push_back(s);
  return faces;
}

// Diagonal weights W_k (length `count`) in ChainComplex column order: the
// per-k-simplex |volume| (Euclidean content via Simplex::volume), or all ones for
// k == 0 or the combinatorial path. A degenerate (zero) cell falls back to 1 so
// W_k stays positive-definite (W_k^{-1/2} is finite). With `lorentzian` the
// **signed** volume is used (timelike cells negative ⇒ W_k indefinite); degenerate
// cells still fall back to +1 so W_k stays invertible (W_k^{-1} is finite).
std::vector<double> simplexWeights(const std::vector<std::vector<SimplexPtr>> &faces,
                                   int k, int count, bool metric,
                                   bool lorentzian = false) {
  std::vector<double> w(static_cast<std::size_t>(std::max(count, 0)), 1.0);
  if (!metric || k == 0 || k < 0 || k >= static_cast<int>(faces.size())) return w;
  const auto &fk = faces[static_cast<std::size_t>(k)];
  for (int j = 0; j < count && j < static_cast<int>(fk.size()); ++j) {
    const double signedVol = fk[static_cast<std::size_t>(j)]->volume();
    const double vol = lorentzian ? signedVol : std::abs(signedVol);
    w[static_cast<std::size_t>(j)] = (std::abs(vol) > 0.0) ? vol : 1.0;
  }
  return w;
}

// Symmetric (W_k-orthonormal) metric Hodge Laplacian for k >= 1:
//   L_k^sym = B_k^T B_k + B_{k+1} B_{k+1}^T,  B_k = W_{k-1}^{1/2} d_k W_k^{-1/2}.
// With metric == false all W = I, giving the combinatorial d_k^T d_k +
// d_{k+1} d_{k+1}^T. Returns a |C_k| x |C_k| real SPD matrix (0 x 0 if no k-cells).
Eigen::MatrixXd metricLaplacian(const Spacetime &K, int k, bool metric) {
  const ChainComplex cc = ChainComplex::fromSpacetime(K);
  const int n = cc.dimension();
  const int nk = static_cast<int>(cc.numSimplices(k));
  Eigen::MatrixXd L = Eigen::MatrixXd::Zero(nk, nk);
  if (nk == 0) return L;

  const std::vector<std::vector<SimplexPtr>> faces = orderedFaces(K);
  const auto weightArr = [&](int kk) {
    const std::vector<double> wv =
        simplexWeights(faces, kk, static_cast<int>(cc.numSimplices(kk)), metric);
    Eigen::ArrayXd a(static_cast<Eigen::Index>(wv.size()));
    for (std::size_t i = 0; i < wv.size(); ++i) a[static_cast<Eigen::Index>(i)] = wv[i];
    return a;
  };
  const auto boundary = [&](int kk, int rows, int cols) {
    const std::vector<long> &flat = cc.boundaryMatrix(kk);
    Eigen::MatrixXd d(rows, cols);
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < cols; ++c)
        d(r, c) = static_cast<double>(flat[static_cast<std::size_t>(r) * cols + c]);
    return d;
  };

  const Eigen::ArrayXd wk = weightArr(k);
  const Eigen::ArrayXd invSqrtWk = wk.sqrt().inverse();

  // Term 1: B_k^T B_k (always present for k >= 1, since k-simplices have faces).
  const int rows = static_cast<int>(cc.numSimplices(k - 1));
  if (rows > 0) {
    const Eigen::MatrixXd dk = boundary(k, rows, nk);
    const Eigen::ArrayXd sqrtWkm1 = weightArr(k - 1).sqrt();
    const Eigen::MatrixXd Bk =
        sqrtWkm1.matrix().asDiagonal() * dk * invSqrtWk.matrix().asDiagonal();
    L.noalias() += Bk.transpose() * Bk;
  }

  // Term 2: B_{k+1} B_{k+1}^T (absent when there are no (k+1)-cells, e.g. k = n).
  const int cols = (k + 1 <= n) ? static_cast<int>(cc.numSimplices(k + 1)) : 0;
  if (cols > 0) {
    const Eigen::MatrixXd dkp1 = boundary(k + 1, nk, cols);
    const Eigen::ArrayXd invSqrtWkp1 = weightArr(k + 1).sqrt().inverse();
    const Eigen::MatrixXd Bkp1 =
        wk.sqrt().matrix().asDiagonal() * dkp1 * invSqrtWkp1.matrix().asDiagonal();
    L.noalias() += Bkp1 * Bkp1.transpose();
  }
  return L;
}

// Signed-weight (Lorentzian) metric Hodge Laplacian for k >= 0 — the discrete
// d'Alembertian. With W indefinite the symmetric W^{1/2} similarity breaks, so the
// operator is assembled directly from the signed metric adjoint
// d_k* = W_k^{-1} d_k^T W_{k-1}:
//   L_k = W_k^{-1} d_k^T W_{k-1} d_k + d_{k+1} W_{k+1}^{-1} d_{k+1}^T W_k.
// This is similar to the symmetric metricLaplacian when every weight is positive
// (W_k^{-1/2} L_k W_k^{1/2} = L_k^sym), so the spectrum/kernel coincide there; with
// signed weights it is generally NON-symmetric (a true d'Alembertian). Returns a
// |C_k| x |C_k| real matrix (0 x 0 if no k-cells). `metric == false` ⇒ unit weights
// (the positive combinatorial operator, no Lorentzian content).
Eigen::MatrixXd signedLaplacian(const Spacetime &K, int k, bool metric) {
  const ChainComplex cc = ChainComplex::fromSpacetime(K);
  const int n = cc.dimension();
  const int nk = static_cast<int>(cc.numSimplices(k));
  Eigen::MatrixXd L = Eigen::MatrixXd::Zero(nk, nk);
  if (nk == 0) return L;

  const std::vector<std::vector<SimplexPtr>> faces = orderedFaces(K);
  const auto weightArr = [&](int kk) {
    const std::vector<double> wv = simplexWeights(
        faces, kk, static_cast<int>(cc.numSimplices(kk)), metric, /*lorentzian=*/true);
    Eigen::ArrayXd a(static_cast<Eigen::Index>(wv.size()));
    for (std::size_t i = 0; i < wv.size(); ++i) a[static_cast<Eigen::Index>(i)] = wv[i];
    return a;
  };
  const auto boundary = [&](int kk, int rows, int cols) {
    const std::vector<long> &flat = cc.boundaryMatrix(kk);
    Eigen::MatrixXd d(rows, cols);
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < cols; ++c)
        d(r, c) = static_cast<double>(flat[static_cast<std::size_t>(r) * cols + c]);
    return d;
  };

  const Eigen::ArrayXd wk = weightArr(k);
  const Eigen::ArrayXd invWk = wk.inverse();

  // Term 1: W_k^{-1} d_k^T W_{k-1} d_k (present once there are (k-1)-faces).
  const int rows = static_cast<int>(cc.numSimplices(k - 1));
  if (k >= 1 && rows > 0) {
    const Eigen::MatrixXd dk = boundary(k, rows, nk);
    const Eigen::ArrayXd wkm1 = weightArr(k - 1);
    L.noalias() += invWk.matrix().asDiagonal() * dk.transpose() *
                   wkm1.matrix().asDiagonal() * dk;
  }

  // Term 2: d_{k+1} W_{k+1}^{-1} d_{k+1}^T W_k (absent when there are no (k+1)-cells).
  const int cols = (k + 1 <= n) ? static_cast<int>(cc.numSimplices(k + 1)) : 0;
  if (cols > 0) {
    const Eigen::MatrixXd dkp1 = boundary(k + 1, nk, cols);
    const Eigen::ArrayXd invWkp1 = weightArr(k + 1).inverse();
    L.noalias() += dkp1 * invWkp1.matrix().asDiagonal() * dkp1.transpose() *
                   wk.matrix().asDiagonal();
  }
  return L;
}

}  // namespace

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

void HodgeLaplacian::requireNonNegativeDegree(int k) {
  if (k < 0)
    throw std::runtime_error(
        "HodgeLaplacian: degree k must be non-negative; requested k=" +
        std::to_string(k));
}

std::vector<std::vector<std::uint64_t>> HodgeLaplacian::cochainOrdering(
    int k, bool useVertexSet) const {
  std::vector<std::vector<std::uint64_t>> ord;
  if (k < 0 || !st_) return ord;
  if (useVertexSet && k == 0) {
    // The Hermitian k=0 operator is indexed over the full sorted-id vertex set
    // (it reads every vertex, including any lone vertices ChainComplex omits).
    ord.reserve(ids_.size());
    for (const std::uint64_t id : ids_) ord.push_back({id});
    return ord;
  }
  // The metric (k>=1) and signed-weight (any k) operators are assembled from the
  // ChainComplex boundary maps, so the eigenvector components are indexed in the
  // canonical ChainComplex k-simplex column order — exactly kSimplexVertices(k),
  // whose count always matches the operator dimension numSimplices(k).
  return ChainComplex::fromSpacetime(*st_).kSimplexVertices(k);
}

Spectrum HodgeLaplacian::makeSpectrum(
    int degree, std::vector<std::vector<std::uint64_t>> ordering,
    const std::vector<cd> &evals, const std::vector<cd> &evecsFlat, int dim,
    bool hermitian) {
  if (dim <= 0) return Spectrum();
  if (ordering.size() != static_cast<std::size_t>(dim))
    throw std::runtime_error(
        "HodgeLaplacian::makeSpectrum: k-simplex ordering size " +
        std::to_string(ordering.size()) + " != operator dimension " +
        std::to_string(dim));
  Eigen::VectorXcd eigenvalues(dim);
  for (int j = 0; j < dim; ++j)
    eigenvalues[j] = evals[static_cast<std::size_t>(j)];
  std::vector<Cochain> eigenvectors;
  eigenvectors.reserve(static_cast<std::size_t>(dim));
  for (int j = 0; j < dim; ++j) {
    Eigen::VectorXcd col(dim);
    for (int i = 0; i < dim; ++i)
      col[i] = evecsFlat[static_cast<std::size_t>(i) * dim + j];
    eigenvectors.emplace_back(degree, ordering, std::move(col));
  }
  return Spectrum(std::move(eigenvalues), std::move(eigenvectors), hermitian);
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

    const cd w = e->getSquaredLength();  // exact complex squared length l^2
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

std::vector<cd> HodgeLaplacian::laplacian(int k, bool metric, bool lorentzian) const {
  requireNonNegativeDegree(k);
  if (lorentzian && k >= 1) {
    // The signed-weight d'Alembertian (generally non-symmetric); real operator
    // returned complex (imag 0) for type parity with the other paths.
    if (!st_) return {};
    const Eigen::MatrixXd L = signedLaplacian(*st_, k, metric);
    const int nk = static_cast<int>(L.rows());
    std::vector<cd> out(static_cast<std::size_t>(nk) * nk, cd(0.0, 0.0));
    for (int i = 0; i < nk; ++i)
      for (int j = 0; j < nk; ++j)
        out[static_cast<std::size_t>(i) * nk + j] = cd(L(i, j), 0.0);
    return out;
  }
  if (k == 0) {
    // k = 0 (unchanged): L = D - A. Off-diagonal entries are -A_ij; the diagonal
    // carries D_ii (the adjacency has no diagonal in a complex without self-loops).
    std::vector<cd> A;
    std::vector<double> D;
    assemble(A, D);
    const std::size_t N = order_;
    std::vector<cd> L(N * N, cd(0.0, 0.0));
    for (std::size_t idx = 0; idx < A.size(); ++idx) L[idx] = -A[idx];
    for (std::size_t i = 0; i < N; ++i) L[i * N + i] += D[i];
    return L;
  }
  // k >= 1: the symmetric metric (or, with metric == false, combinatorial)
  // Laplacian L_k^sym, returned complex (imaginary parts zero) for type parity
  // with the k = 0 operator.
  if (!st_) return {};
  const Eigen::MatrixXd L = metricLaplacian(*st_, k, metric);
  const int nk = static_cast<int>(L.rows());
  std::vector<cd> out(static_cast<std::size_t>(nk) * nk, cd(0.0, 0.0));
  for (int i = 0; i < nk; ++i)
    for (int j = 0; j < nk; ++j)
      out[static_cast<std::size_t>(i) * nk + j] = cd(L(i, j), 0.0);
  return out;
}

std::vector<double> HodgeLaplacian::weights(int k, bool lorentzian) const {
  if (k < 0 || !st_) return {};
  const ChainComplex cc = ChainComplex::fromSpacetime(*st_);
  if (k > cc.dimension()) return {};
  const int m = static_cast<int>(cc.numSimplices(k));
  if (k == 0) return std::vector<double>(static_cast<std::size_t>(m), 1.0);
  return simplexWeights(orderedFaces(*st_), k, m, /*metric=*/true, lorentzian);
}

const HodgeLaplacian::MetricSpectrum &HodgeLaplacian::ensureMetricSpectrum(
    int k, bool metric) const {
  const long long key = static_cast<long long>(k) * 2 + (metric ? 1 : 0);
  const auto cached = metricCache_.find(key);
  if (cached != metricCache_.end()) return cached->second;

  MetricSpectrum sp;
  if (st_) {
    const Eigen::MatrixXd L = metricLaplacian(*st_, k, metric);
    const int nk = static_cast<int>(L.rows());
    sp.dim = nk;
    sp.evals.assign(static_cast<std::size_t>(nk), 0.0);
    sp.evecs.assign(static_cast<std::size_t>(nk) * nk, cd(0.0, 0.0));
    if (nk > 0) {
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(L);
      const Eigen::VectorXd &lam = es.eigenvalues();   // ascending
      const Eigen::MatrixXd &V = es.eigenvectors();    // orthonormal columns
      for (int i = 0; i < nk; ++i) sp.evals[static_cast<std::size_t>(i)] = lam[i];
      for (int i = 0; i < nk; ++i)
        for (int j = 0; j < nk; ++j)
          sp.evecs[static_cast<std::size_t>(i) * nk + j] = cd(V(i, j), 0.0);
    }
  }
  return metricCache_.emplace(key, std::move(sp)).first->second;
}

const HodgeLaplacian::LorentzianSpectrum &HodgeLaplacian::ensureLorentzianSpectrum(
    int k, bool metric) const {
  const long long key = static_cast<long long>(k) * 2 + (metric ? 1 : 0);
  const auto cached = lorentzianCache_.find(key);
  if (cached != lorentzianCache_.end()) return cached->second;

  LorentzianSpectrum sp;
  if (st_) {
    const Eigen::MatrixXd L = signedLaplacian(*st_, k, metric);
    const int nk = static_cast<int>(L.rows());
    sp.dim = nk;
    sp.evals.assign(static_cast<std::size_t>(nk), cd(0.0, 0.0));
    sp.evecs.assign(static_cast<std::size_t>(nk) * nk, cd(0.0, 0.0));
    sp.wk = weights(k, /*lorentzian=*/true);
    if (nk > 0) {
      // Indefinite metric ⇒ the operator is non-self-adjoint; a general solver is
      // needed (eigenvalues may be negative or complex-conjugate pairs).
      Eigen::EigenSolver<Eigen::MatrixXd> es(L);
      const Eigen::VectorXcd lam = es.eigenvalues();
      const Eigen::MatrixXcd V = es.eigenvectors();

      // Stable, reproducible order: ascending by (Re, Im) of the eigenvalue.
      std::vector<int> order(static_cast<std::size_t>(nk));
      for (int i = 0; i < nk; ++i) order[static_cast<std::size_t>(i)] = i;
      std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (lam[a].real() != lam[b].real()) return lam[a].real() < lam[b].real();
        return lam[a].imag() < lam[b].imag();
      });
      for (int c = 0; c < nk; ++c) {
        const int src = order[static_cast<std::size_t>(c)];
        sp.evals[static_cast<std::size_t>(c)] = lam[src];
        for (int r = 0; r < nk; ++r)
          sp.evecs[static_cast<std::size_t>(r) * nk + c] = V(r, src);
      }
    }
  }
  return lorentzianCache_.emplace(key, std::move(sp)).first->second;
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

Spectrum HodgeLaplacian::spectrum(int k, bool metric) const {
  requireNonNegativeDegree(k);
  if (k == 0) {
    ensureDecomposition();
    std::vector<cd> evalsC(evals_.size());
    for (std::size_t i = 0; i < evals_.size(); ++i) evalsC[i] = cd(evals_[i], 0.0);
    return makeSpectrum(0, cochainOrdering(0, /*useVertexSet=*/true), evalsC,
                        evecs_, static_cast<int>(order_), /*hermitian=*/true);
  }
  const MetricSpectrum &sp = ensureMetricSpectrum(k, metric);
  std::vector<cd> evalsC(sp.evals.size());
  for (std::size_t i = 0; i < sp.evals.size(); ++i) evalsC[i] = cd(sp.evals[i], 0.0);
  return makeSpectrum(k, cochainOrdering(k, /*useVertexSet=*/true), evalsC,
                      sp.evecs, sp.dim, /*hermitian=*/true);
}

std::vector<double> HodgeLaplacian::eigenvalues(int k, bool metric) const {
  requireNonNegativeDegree(k);
  if (k == 0) {
    ensureDecomposition();
    return evals_;
  }
  return ensureMetricSpectrum(k, metric).evals;
}

std::vector<cd> HodgeLaplacian::eigenvectors(int k, bool metric) const {
  requireNonNegativeDegree(k);
  if (k == 0) {
    ensureDecomposition();
    return evecs_;
  }
  return ensureMetricSpectrum(k, metric).evecs;
}

std::vector<Cochain> HodgeLaplacian::harmonics(int k, double tol,
                                               bool metric) const {
  // ker L_k as Cochains: the eigenvectors with (near-)zero eigenvalue, a basis
  // for H_k (the count is b_k). requireNonNegativeDegree runs inside spectrum().
  return spectrum(k, metric).harmonics(tol);
}

std::vector<cd> HodgeLaplacian::harmonicMatrix(int k, double tol,
                                               bool metric) const {
  requireNonNegativeDegree(k);
  // The same cached eigendecompositions harmonics() reads, emitted column-by-
  // selected-column so no Cochain objects are materialized.
  const std::vector<double> *evals = nullptr;
  const std::vector<cd> *evecs = nullptr;
  int dim = 0;
  if (k == 0) {
    ensureDecomposition();
    evals = &evals_;
    evecs = &evecs_;
    dim = static_cast<int>(order_);
  } else {
    const MetricSpectrum &sp = ensureMetricSpectrum(k, metric);
    evals = &sp.evals;
    evecs = &sp.evecs;
    dim = sp.dim;
  }
  std::vector<cd> rows;
  for (int j = 0; j < dim; ++j) {
    if (std::abs((*evals)[static_cast<std::size_t>(j)]) >= tol) continue;
    for (int i = 0; i < dim; ++i)
      rows.push_back((*evecs)[static_cast<std::size_t>(i) * dim + j]);
  }
  return rows;
}

Spectrum HodgeLaplacian::lorentzianSpectrum(int k, bool metric) const {
  requireNonNegativeDegree(k);
  const LorentzianSpectrum &sp = ensureLorentzianSpectrum(k, metric);
  return makeSpectrum(k, cochainOrdering(k, /*useVertexSet=*/false), sp.evals,
                      sp.evecs, sp.dim, /*hermitian=*/false);
}

std::vector<cd> HodgeLaplacian::lorentzianEigenvalues(int k, bool metric) const {
  requireNonNegativeDegree(k);
  return ensureLorentzianSpectrum(k, metric).evals;
}

std::vector<cd> HodgeLaplacian::lorentzianEigenvectors(int k, bool metric) const {
  requireNonNegativeDegree(k);
  return ensureLorentzianSpectrum(k, metric).evecs;
}

std::vector<Cochain> HodgeLaplacian::lorentzianHarmonics(int k, double tol,
                                                         bool metric) const {
  // The near-kernel (pseudo-Hodge) representatives as Cochains; the matching
  // indefinite W-norms come from lorentzianNullNorms (same order).
  return lorentzianSpectrum(k, metric).harmonics(tol);
}

std::vector<double> HodgeLaplacian::lorentzianNullNorms(int k, double tol,
                                                        bool metric) const {
  requireNonNegativeDegree(k);
  const LorentzianSpectrum &sp = ensureLorentzianSpectrum(k, metric);
  const std::size_t N = static_cast<std::size_t>(sp.dim);
  if (N == 0) return {};

  std::vector<double> norms;
  for (std::size_t j = 0; j < N; ++j) {
    if (std::abs(sp.evals[j]) >= tol) continue;
    // Indefinite W-norm <h,h>_W = sum_i W_{k,i} |h_i|^2 (real; signed W_k). A
    // value ≈ 0 marks a null (lightlike) harmonic direction.
    double nrm = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
      const cd hi = sp.evecs[i * N + j];
      nrm += sp.wk[i] * std::norm(hi);  // std::norm = |hi|^2
    }
    norms.push_back(nrm);
  }
  return norms;
}

}  // namespace tessera::cobordism
