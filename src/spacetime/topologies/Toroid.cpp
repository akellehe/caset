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

#include "SimplexOrientation.h"
#include "spacetime/topologies/Toroid.h"
#include "spacetime/Spacetime.h"
#include "utils.h"
#include <deque>
#include <ATen/core/interned_strings.h>

namespace caset {
void Toroid::build(Spacetime *spacetime, int nSimplices) {
  auto dimensions = spacetime->getMetric()->getSignature()->getDimensions();
  SimplexOrientation orientation{1, static_cast<std::uint8_t>(dimensions - 1)};
  spacetime->reserve(nSimplices);
  auto [seed, created] = spacetime->createSimplex(orientation.numeric());
  const auto &facets = seed->getFacets();
  std::deque<SimplexPtr> exteriorFacets{facets.begin(), facets.end()};
  auto complexSize = 1;
  std::vector<double> plusT{1.};
  std::vector<double> minusT{-1};
  while (complexSize < nSimplices) {
    SimplexPtr &exteriorFacet = exteriorFacets.front();
    exteriorFacets.pop_front();
    if (exteriorFacet->isTimelike()) continue;
    if (random_uniform() > 0) {
      auto vertex = spacetime->createVertex(plusT);
      auto [kSimplex, newFacets] = exteriorFacet->cone(vertex);
      exteriorFacets.insert(exteriorFacets.end(), newFacets.begin(), newFacets.end());
      ++complexSize;
    } else {
      auto vertex = spacetime->createVertex(minusT);
      auto [kSimplex, newFacets] = exteriorFacet->cone(vertex);
      exteriorFacets.insert(exteriorFacets.end(), newFacets.begin(), newFacets.end());
      ++complexSize;
    }
  }
}
}
