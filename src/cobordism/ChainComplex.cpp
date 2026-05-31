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

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <vector>

#include "cobordism/IntegerLinalg.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

using Face = std::vector<std::uint64_t>;  // sorted vertex ids

ChainComplex ChainComplex::fromSpacetime(const Spacetime &K) {
  ChainComplex cc;

  // Collect the full face-closure: every sub-face of every registered simplex,
  // bucketed by cardinality (card = k+1 for a k-simplex).
  std::vector<std::set<Face>> faceSets;  // faceSets[card-1]
  for (const auto &simplex : K.getSimplices()) {
    Face verts;
    for (const auto &v : simplex->getVertices())
      verts.push_back(v->getId());
    std::sort(verts.begin(), verts.end());
    const std::size_t sz = verts.size();
    if (faceSets.size() < sz) faceSets.resize(sz);
    // Enumerate every nonempty subset by cardinality.
    for (std::size_t card = 1; card <= sz; ++card) {
      std::vector<std::size_t> idx(card);
      for (std::size_t i = 0; i < card; ++i) idx[i] = i;
      for (;;) {
        Face f;
        f.reserve(card);
        for (auto i : idx) f.push_back(verts[i]);
        faceSets[card - 1].insert(std::move(f));
        // next combination of indices over [0, sz)
        std::size_t pos = card;
        while (pos > 0) {
          --pos;
          if (idx[pos] != pos + sz - card) {
            ++idx[pos];
            for (std::size_t j = pos + 1; j < card; ++j) idx[j] = idx[j - 1] + 1;
            break;
          }
          if (pos == 0) { pos = card + 1; break; }
        }
        if (pos == card + 1) break;
      }
    }
  }

  cc.dimension_ = static_cast<int>(faceSets.size()) - 1;
  if (cc.dimension_ < 0) return cc;  // empty complex

  // Per-dimension ordered face lists + index maps (sorted order is canonical).
  const int n = cc.dimension_;
  std::vector<std::vector<Face>> faces(n + 1);
  std::vector<std::map<Face, int>> index(n + 1);
  cc.counts_.assign(n + 1, 0);
  for (int k = 0; k <= n; ++k) {
    faces[k].assign(faceSets[k].begin(), faceSets[k].end());  // std::set is sorted
    cc.counts_[k] = faces[k].size();
    for (int j = 0; j < static_cast<int>(faces[k].size()); ++j)
      index[k][faces[k][j]] = j;
  }

  // Boundary matrices ∂_k for k = 1..n. boundary_[0] stays empty.
  cc.boundary_.assign(n + 1, {});
  for (int k = 1; k <= n; ++k) {
    const int rows = static_cast<int>(cc.counts_[k - 1]);
    const int cols = static_cast<int>(cc.counts_[k]);
    std::vector<long> M(static_cast<std::size_t>(rows) * cols, 0);
    for (int j = 0; j < cols; ++j) {
      const Face &s = faces[k][j];  // (k+1) sorted vertices
      for (std::size_t i = 0; i < s.size(); ++i) {
        Face f;
        f.reserve(s.size() - 1);
        for (std::size_t m = 0; m < s.size(); ++m)
          if (m != i) f.push_back(s[m]);
        const int r = index[k - 1].at(f);
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
