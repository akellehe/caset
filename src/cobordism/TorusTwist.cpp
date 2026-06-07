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

#include "cobordism/TorusTwist.h"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <vector>

#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

namespace {

// Top simplices of a complex, as sorted vertex-id tuples ("top" = maximal
// dimension present). Mirrors the helper in Cobordism.cpp (kept local so this
// translation unit has no dependence on that one's anonymous namespace).
std::set<std::vector<std::uint64_t>> topSimplexSet(const Spacetime &complex) {
  std::vector<std::vector<std::uint64_t>> all;
  std::size_t topVertexCount = 0;
  for (const auto &simplex : complex.getSimplices()) {
    std::vector<std::uint64_t> vertices;
    for (const auto &v : simplex->getVertices()) vertices.push_back(v->getId());
    std::sort(vertices.begin(), vertices.end());
    topVertexCount = std::max(topVertexCount, vertices.size());
    all.push_back(std::move(vertices));
  }
  std::set<std::vector<std::uint64_t>> tops;
  for (auto &s : all)
    if (s.size() == topVertexCount) tops.insert(std::move(s));
  return tops;
}

}  // namespace

TorusTwist TorusTwist::identity() { return TorusTwist(1, 0, 0, 1); }
TorusTwist TorusTwist::S() { return TorusTwist(0, -1, 1, 0); }
TorusTwist TorusTwist::T() { return TorusTwist(1, 1, 0, 1); }
TorusTwist TorusTwist::flip() { return TorusTwist(0, 1, 1, 0); }

TorusTwist TorusTwist::compose(const TorusTwist &rhs) const {
  // [[a b][c d]] * [[e f][g h]]
  return TorusTwist(a_ * rhs.a_ + b_ * rhs.c_, a_ * rhs.b_ + b_ * rhs.d_,
                    c_ * rhs.a_ + d_ * rhs.c_, c_ * rhs.b_ + d_ * rhs.d_);
}

TorusTwist TorusTwist::power(int k) const {
  if (k < 0)
    throw std::invalid_argument("TorusTwist::power: exponent must be >= 0");
  TorusTwist result = identity();
  for (int i = 0; i < k; ++i) result = result.compose(*this);
  return result;
}

bool TorusTwist::equals(const TorusTwist &rhs) const {
  return a_ == rhs.a_ && b_ == rhs.b_ && c_ == rhs.c_ && d_ == rhs.d_;
}

long TorusTwist::determinant() const { return a_ * d_ - b_ * c_; }

std::array<long, 4> TorusTwist::matrix() const { return {a_, b_, c_, d_}; }

std::map<std::uint64_t, std::uint64_t> TorusTwist::vertexPermutation(
    int n) const {
  if (n < 1)
    throw std::invalid_argument(
        "TorusTwist::vertexPermutation: n must be >= 1");
  const long m = n;
  auto mod = [m](long x) { return static_cast<std::uint64_t>(((x % m) + m) % m); };
  std::map<std::uint64_t, std::uint64_t> perm;
  for (long i = 0; i < m; ++i)
    for (long j = 0; j < m; ++j) {
      const std::uint64_t ni = mod(a_ * i + b_ * j);
      const std::uint64_t nj = mod(c_ * i + d_ * j);
      perm[static_cast<std::uint64_t>(i * m + j)] =
          ni * static_cast<std::uint64_t>(m) + nj;
    }
  return perm;
}

bool TorusTwist::isSimplicialAutomorphism(
    const Spacetime &torus,
    const std::map<std::uint64_t, std::uint64_t> &permutation) {
  const std::set<std::vector<std::uint64_t>> tops = topSimplexSet(torus);
  std::set<std::vector<std::uint64_t>> image;
  for (const auto &top : tops) {
    std::vector<std::uint64_t> mapped;
    mapped.reserve(top.size());
    for (std::uint64_t v : top) {
      auto it = permutation.find(v);
      if (it == permutation.end()) return false;  // not defined on a vertex
      mapped.push_back(it->second);
    }
    std::sort(mapped.begin(), mapped.end());
    // A simplicial map must stay injective on each simplex's vertices.
    if (std::adjacent_find(mapped.begin(), mapped.end()) != mapped.end())
      return false;
    image.insert(std::move(mapped));
  }
  return image == tops;
}

bool TorusTwist::satisfiesModularRelations() {
  const bool s4 = S().power(4).equals(identity());
  const bool st3 = S().compose(T()).power(3).equals(S().power(2));
  return s4 && st3;
}

}  // namespace tessera::cobordism
