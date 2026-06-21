// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "mesh/TemporalOrientation.h"
#include "spacetime/topologies/Cylinder.h"
#include "spacetime/Spacetime.h"
#include "utils.h"
#include <deque>
#include <cmath>

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;
void Cylinder::build(Spacetime *spacetime, int nSimplices) {
  auto dimensions = spacetime->getMetric()->getSignature()->getDimensions();
  TemporalOrientation orientation{1, static_cast<std::uint8_t>(dimensions)};
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
      if (exteriorFacet->isSpatial()) continue;
      auto vertex = spacetime->createVertex(std::vector<double>{1.});
      auto [kSimplex, newFacets] = exteriorFacet->cone(vertex);
      exteriorFacets.insert(exteriorFacets.end(), newFacets.begin(), newFacets.end());
      ++complexSize;
    }
    spacetime->incrementTime();
  }
}
}
