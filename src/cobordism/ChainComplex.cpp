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

#include "cobordism/ChainComplex.h"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cobordism/IntegerLinalg.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

using Face = std::vector<std::uint64_t>;  // sorted vertex ids

namespace {
// Sorted vertex ids of a simplex — the homological reference ordering.
Face sortedIds(const SimplexPtr &s) {
  Face ids;
  for (const auto &v : s->getVertices()) ids.push_back(v->getId());
  std::sort(ids.begin(), ids.end());
  return ids;
}
}  // namespace

ChainComplex ChainComplex::fromSpacetime(const Spacetime &K) {
  ChainComplex cc;

  // Collect the face lattice through the mesh's own facet operation
  // (Simplex::getFacets) — a BFS down from the registered simplices,
  // de-duplicated by fingerprint and bucketed by dimension. We do NOT
  // re-derive faces; getFacets is the single source of truth for "the faces of
  // a simplex". The only thing ChainComplex adds is the homological boundary
  // sign below, which mesh facets don't carry.
  std::map<int, std::vector<SimplexPtr>> byDim;
  std::unordered_set<std::uint64_t> seen;
  std::vector<SimplexPtr> stack(K.getSimplices().begin(), K.getSimplices().end());
  while (!stack.empty()) {
    SimplexPtr s = stack.back();
    stack.pop_back();
    if (s == nullptr) continue;
    if (!seen.insert(s->fingerprint.fingerprint()).second) continue;
    const int k = static_cast<int>(s->size()) - 1;
    byDim[k].push_back(s);
    if (k >= 1)
      for (const auto &f : s->getFacets()) stack.push_back(f);
  }

  if (byDim.empty()) return cc;  // empty complex
  const int n = byDim.rbegin()->first;
  cc.dimension_ = n;

  // Order each dimension deterministically (by sorted vertex ids) and index
  // the simplices by fingerprint for boundary lookups.
  std::vector<std::vector<SimplexPtr>> faces(n + 1);
  std::vector<std::unordered_map<std::uint64_t, int>> index(n + 1);
  cc.counts_.assign(n + 1, 0);
  cc.faceVerts_.assign(n + 1, {});
  for (int k = 0; k <= n; ++k) {
    auto &vec = byDim[k];
    std::sort(vec.begin(), vec.end(), [](const SimplexPtr &a, const SimplexPtr &b) {
      return sortedIds(a) < sortedIds(b);
    });
    faces[k] = vec;
    cc.counts_[k] = vec.size();
    cc.faceVerts_[k].reserve(vec.size());
    for (int j = 0; j < static_cast<int>(vec.size()); ++j) {
      index[k][vec[j]->fingerprint.fingerprint()] = j;
      cc.faceVerts_[k].push_back(sortedIds(vec[j]));
    }
  }

  // Boundary ∂_k (rows = |C_{k-1}|, cols = |C_k|): each column is a k-simplex,
  // its nonzero rows are its facets, and the orientation is already carried by
  // getFacets()'s canonical order — facet at index i is the i-th vertex
  // dropped, so its coefficient is (-1)^i (see Simplex::getFacets). We read it
  // off the index rather than recomputing any sign.
  cc.boundary_.assign(n + 1, {});
  for (int k = 1; k <= n; ++k) {
    const int rows = static_cast<int>(cc.counts_[k - 1]);
    const int cols = static_cast<int>(cc.counts_[k]);
    std::vector<long> M(static_cast<std::size_t>(rows) * cols, 0);
    for (int j = 0; j < cols; ++j) {
      const auto &facets = faces[k][j]->getFacets();
      for (int i = 0; i < static_cast<int>(facets.size()); ++i) {
        const int r = index[k - 1].at(facets[i]->fingerprint.fingerprint());
        M[static_cast<std::size_t>(r) * cols + j] = (i % 2 == 0) ? 1 : -1;
      }
    }
    cc.boundary_[k] = std::move(M);
  }
  return cc;
}

std::size_t ChainComplex::numSimplices(int k) const noexcept {
  if (k < 0 || k > dimension_) return 0;
  return counts_[static_cast<std::size_t>(k)];
}

int ChainComplex::eulerCharacteristic() const noexcept {
  int chi = 0;
  for (int k = 0; k <= dimension_; ++k)
    chi += (k % 2 == 0 ? 1 : -1) * static_cast<int>(counts_[static_cast<std::size_t>(k)]);
  return chi;
}

const std::vector<long> &ChainComplex::boundaryMatrix(int k) const {
  static const std::vector<long> kEmpty{};
  if (k < 0 || k > dimension_) return kEmpty;
  return boundary_[static_cast<std::size_t>(k)];
}

bool ChainComplex::boundaryComposesToZero() const {
  // ∂_{k-1} ∘ ∂_k = 0 : (|C_{k-2}| x |C_{k-1}|) · (|C_{k-1}| x |C_k|).
  for (int k = 2; k <= dimension_; ++k) {
    const int a = static_cast<int>(counts_[k - 2]);  // rows of ∂_{k-1}
    const int b = static_cast<int>(counts_[k - 1]);  // shared dim
    const int c = static_cast<int>(counts_[k]);      // cols of ∂_k
    const auto &L = boundary_[static_cast<std::size_t>(k - 1)];
    const auto &R = boundary_[static_cast<std::size_t>(k)];
    for (int i = 0; i < a; ++i)
      for (int j = 0; j < c; ++j) {
        long acc = 0;
        for (int m = 0; m < b; ++m)
          acc += L[static_cast<std::size_t>(i) * b + m] * R[static_cast<std::size_t>(m) * c + j];
        if (acc != 0) return false;
      }
  }
  return true;
}

int ChainComplex::rankOfBoundary(int k) const {
  if (k < 1 || k > dimension_) return 0;
  return integerRank(boundary_[static_cast<std::size_t>(k)],
                     static_cast<int>(counts_[k - 1]), static_cast<int>(counts_[k]));
}

int ChainComplex::gf2RankOfBoundary(int k) const {
  if (k < 1 || k > dimension_) return 0;
  const auto &M = boundary_[static_cast<std::size_t>(k)];
  std::vector<int> bits(M.size());
  for (std::size_t i = 0; i < M.size(); ++i) bits[i] = static_cast<int>(M[i] & 1);
  return gf2Rank(std::move(bits), static_cast<int>(counts_[k - 1]),
                 static_cast<int>(counts_[k]));
}

std::vector<int> ChainComplex::bettiNumbers() const {
  std::vector<int> b;
  if (dimension_ < 0) return b;
  b.assign(dimension_ + 1, 0);
  for (int k = 0; k <= dimension_; ++k)
    b[k] = static_cast<int>(counts_[k]) - rankOfBoundary(k) - rankOfBoundary(k + 1);
  return b;
}

std::vector<int> ChainComplex::bettiNumbersGF2() const {
  std::vector<int> b;
  if (dimension_ < 0) return b;
  b.assign(dimension_ + 1, 0);
  for (int k = 0; k <= dimension_; ++k)
    b[k] = static_cast<int>(counts_[k]) - gf2RankOfBoundary(k) - gf2RankOfBoundary(k + 1);
  return b;
}

std::vector<double> ChainComplex::intersectionForm() const {
  if (dimension_ != 4) return {};
  const int b2 = bettiNumbers()[2];
  if (b2 == 0) return {};

  const int nC1 = static_cast<int>(counts_[1]);
  const int nC2 = static_cast<int>(counts_[2]);
  const int nC3 = static_cast<int>(counts_[3]);
  const int nC4 = static_cast<int>(counts_[4]);

  auto eigenOf = [&](int k, int rows, int cols) {
    Eigen::MatrixXd M(rows, cols);
    const auto &flat = boundary_[static_cast<std::size_t>(k)];
    for (int i = 0; i < rows; ++i)
      for (int j = 0; j < cols; ++j)
        M(i, j) = static_cast<double>(flat[static_cast<std::size_t>(i) * cols + j]);
    return M;
  };
  const Eigen::MatrixXd d2 = eigenOf(2, nC1, nC2);  // \partial_2 : C_2 -> C_1
  const Eigen::MatrixXd d3 = eigenOf(3, nC2, nC3);  // \partial_3 : C_3 -> C_2
  const Eigen::MatrixXd d4 = eigenOf(4, nC3, nC4);  // \partial_4 : C_4 -> C_3

  // Fundamental class [K]: generator of ker \partial_4. For a closed oriented
  // 4-manifold this is 1-dimensional with ±1 entries; otherwise [K] is ill-
  // defined and the signature is not.
  const Eigen::MatrixXd K4 = Eigen::FullPivLU<Eigen::MatrixXd>(d4).kernel();
  if (K4.cols() != 1)
    throw std::runtime_error(
        "ChainComplex::intersectionForm: a closed orientable 4-manifold is "
        "required (dim ker d_4 != 1)");
  Eigen::VectorXd c = K4.col(0);
  int piv = 0;
  for (int i = 1; i < c.size(); ++i)
    if (std::abs(c[i]) > std::abs(c[piv])) piv = i;
  c /= c[piv];  // normalize: entries become ±1 (orientation fixed up to overall sign)

  // 2-cocycles Z^2 = ker(\delta_2 = \partial_3^T); 2-coboundaries B^2 =
  // im(\delta_1 = \partial_2^T). H^2 reps = cocycles independent modulo
  // coboundaries.
  const Eigen::MatrixXd Zco =
      Eigen::FullPivLU<Eigen::MatrixXd>(d3.transpose()).kernel();  // |C_2| x dimZ
  const Eigen::MatrixXd Bco = d2.transpose();                      // |C_2| x |C_1|

  const double tol = 1e-9;
  auto rankOf = [&](const Eigen::MatrixXd &M) {
    if (M.cols() == 0) return 0;
    Eigen::FullPivLU<Eigen::MatrixXd> lu(M);
    lu.setThreshold(tol);
    return static_cast<int>(lu.rank());
  };
  Eigen::MatrixXd span = Bco;
  int spanRank = rankOf(span);
  std::vector<Eigen::VectorXd> reps;
  for (int j = 0; j < Zco.cols() && static_cast<int>(reps.size()) < b2; ++j) {
    Eigen::MatrixXd aug(nC2, span.cols() + 1);
    if (span.cols() > 0) aug.leftCols(span.cols()) = span;
    aug.col(span.cols()) = Zco.col(j);
    if (rankOf(aug) > spanRank) {
      reps.push_back(Zco.col(j));
      span = aug;
      ++spanRank;
    }
  }

  // Index 2-simplices (sorted vertex triples) for cup-product lookups.
  std::map<std::array<std::uint64_t, 3>, int> triIndex;
  for (int j = 0; j < nC2; ++j) {
    const auto &v = faceVerts_[2][static_cast<std::size_t>(j)];
    triIndex[{v[0], v[1], v[2]}] = j;
  }

  // Alexander–Whitney cup product on [K]:
  //   Q(\alpha,\beta) = \sum_\sigma c_\sigma \, \alpha(v_0 v_1 v_2)\,\beta(v_2 v_3 v_4)
  // over each 4-simplex \sigma = [v_0 < ... < v_4].
  const int B = static_cast<int>(reps.size());
  std::vector<double> Q(static_cast<std::size_t>(B) * B, 0.0);
  for (int s = 0; s < nC4; ++s) {
    const auto &v = faceVerts_[4][static_cast<std::size_t>(s)];
    const int fi = triIndex.at({v[0], v[1], v[2]});
    const int bi = triIndex.at({v[2], v[3], v[4]});
    const double cs = c[s];
    for (int i = 0; i < B; ++i)
      for (int j = 0; j < B; ++j)
        Q[static_cast<std::size_t>(i) * B + j] += cs * reps[i][fi] * reps[j][bi];
  }
  // The cup product on H^2 is symmetric; clean up numerical asymmetry.
  for (int i = 0; i < B; ++i)
    for (int j = i + 1; j < B; ++j) {
      const double avg = 0.5 * (Q[static_cast<std::size_t>(i) * B + j] +
                                Q[static_cast<std::size_t>(j) * B + i]);
      Q[static_cast<std::size_t>(i) * B + j] = avg;
      Q[static_cast<std::size_t>(j) * B + i] = avg;
    }
  return Q;
}

int ChainComplex::signature() const {
  const std::vector<double> Q = intersectionForm();
  if (Q.empty()) return 0;
  const int B = static_cast<int>(std::lround(std::sqrt(static_cast<double>(Q.size()))));
  Eigen::MatrixXd M(B, B);
  for (int i = 0; i < B; ++i)
    for (int j = 0; j < B; ++j)
      M(i, j) = Q[static_cast<std::size_t>(i) * B + j];
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(M, Eigen::EigenvaluesOnly);
  double scale = 0.0;
  for (int i = 0; i < B; ++i) scale = std::max(scale, std::abs(es.eigenvalues()[i]));
  const double tol = 1e-7 * (scale > 0 ? scale : 1.0);  // relative: form is unimodular
  int pos = 0, neg = 0;
  for (int i = 0; i < B; ++i) {
    const double l = es.eigenvalues()[i];
    if (l > tol) ++pos;
    else if (l < -tol) ++neg;
  }
  return pos - neg;
}

std::vector<long> ChainComplex::torsion(int k) const {
  std::vector<long> out;
  if (k < 0 || k + 1 > dimension_) return out;  // torsion of H_k comes from ∂_{k+1}
  const int kk = k + 1;
  auto snf = smithNormalForm(boundary_[static_cast<std::size_t>(kk)],
                             static_cast<int>(counts_[kk - 1]),
                             static_cast<int>(counts_[kk]));
  for (long d : snf.invariantFactors)
    if (d > 1) out.push_back(d);
  return out;
}

}  // namespace tessera::cobordism
