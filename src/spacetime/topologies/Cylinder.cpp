// MIT License
// Copyright (c) 2025 Andrew Kelleher

#include "SimplexOrientation.h"
#include "spacetime/topologies/Cylinder.h"
#include "spacetime/Spacetime.h"
#include "utils.h"
#include <deque>
#include <cmath>

namespace caset {
void Cylinder::build(Spacetime *spacetime, int nSimplices) {
  auto dimensions = spacetime->getMetric()->getSignature()->getDimensions();
  SimplexOrientation orientation{1, static_cast<std::uint8_t>(dimensions)};
  spacetime->reserve(nSimplices);

  int numLayers = std::max(2, static_cast<int>(std::cbrt(nSimplices)));
  int perLayer = std::max(1, nSimplices / numLayers);
  int complexSize = 0;

  for (int layer = 0; layer < numLayers && complexSize < nSimplices; ++layer) {
    auto [seed, created] = spacetime->createSimplex(orientation.numeric());
    if (!created) continue;
    const auto &facets = seed->getFacets();
    std::deque<SimplexPtr> exteriorFacets{facets.begin(), facets.end()};
    ++complexSize;

    int layerTarget = std::min(complexSize + perLayer, nSimplices);
    // Cylinder: always cone forward
    while (complexSize < layerTarget && !exteriorFacets.empty()) {
      SimplexPtr &exteriorFacet = exteriorFacets.front();
      exteriorFacets.pop_front();
      if (exteriorFacet->isTimelike()) continue;
      auto vertex = spacetime->createVertex(std::vector<double>{1.});
      auto [kSimplex, newFacets] = exteriorFacet->cone(vertex);
      exteriorFacets.insert(exteriorFacets.end(), newFacets.begin(), newFacets.end());
      ++complexSize;
    }
    spacetime->incrementTime();
  }
}
}
