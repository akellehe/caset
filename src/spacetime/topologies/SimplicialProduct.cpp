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

#include "spacetime/topologies/SimplicialProduct.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "spacetime/Spacetime.h"

namespace tessera::spacetime {

namespace {

// Build a factor topology into a scratch Spacetime and return its top-simplex
// vertex-id tuples (sorted) along with its vertex count. The fixture
// topologies number vertices 0..|V|-1, so |V| = getVertexCount().
struct FactorTops {
  int numVertices = 0;
  std::vector<std::vector<std::uint64_t>> tops;
};

FactorTops factorTops(const std::shared_ptr<Topology> &t) {
  auto scratch = std::make_shared<Spacetime>();
  t->build(scratch.get(), 0);
  FactorTops out;
  out.numVertices = static_cast<int>(scratch->getVertexCount());
  std::size_t maxSize = 0;
  for (const auto &s : scratch->getSimplices())
    maxSize = std::max(maxSize, static_cast<std::size_t>(s->size()));
  for (const auto &s : scratch->getSimplices()) {
    if (s->size() != maxSize) continue;
    std::vector<std::uint64_t> ids;
    for (const auto &v : s->getVertices()) ids.push_back(v->getId());
    std::sort(ids.begin(), ids.end());
    out.tops.push_back(std::move(ids));
  }
  return out;
}

}  // namespace

void SimplicialProduct::build(Spacetime *spacetime, int /*numSimplices*/) {
  const FactorTops A = factorTops(left_);
  const FactorTops B = factorTops(right_);
  const int nB = B.numVertices;
  auto productId = [nB](std::uint64_t u, std::uint64_t v) -> std::uint64_t {
    return u * static_cast<std::uint64_t>(nB) + v;
  };

  std::vector<std::vector<std::uint64_t>> tops;
  for (const auto &sigma : A.tops) {
    for (const auto &tau : B.tops) {
      const int p = static_cast<int>(sigma.size()) - 1;
      const int q = static_cast<int>(tau.size()) - 1;
      // Each monotone lattice path (0,0)->(p,q) is an interleaving of p
      // i-steps and q j-steps; enumerate by choosing which of the p+q step
      // slots are i-steps.
      std::vector<int> iStep(static_cast<std::size_t>(p + q));
      for (int s = 0; s < p; ++s) iStep[s] = 1;  // first p are i-steps...
      std::sort(iStep.begin(), iStep.end());     // ...then permute (000..111..)
      do {
        std::vector<std::uint64_t> cell;
        cell.reserve(static_cast<std::size_t>(p + q + 1));
        int i = 0, j = 0;
        cell.push_back(productId(sigma[i], tau[j]));
        for (int step = 0; step < p + q; ++step) {
          if (iStep[static_cast<std::size_t>(step)]) ++i; else ++j;
          cell.push_back(productId(sigma[i], tau[j]));
        }
        tops.push_back(std::move(cell));
      } while (std::next_permutation(iStep.begin(), iStep.end()));
    }
  }

  buildExplicit(spacetime, static_cast<std::size_t>(A.numVertices) * nB, tops);
}

} // namespace tessera::spacetime
