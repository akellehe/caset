// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "spacetime/topologies/StellarSubdivision.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "spacetime/Spacetime.h"

namespace tessera::spacetime {

void StellarSubdivision::build(Spacetime *spacetime, int /*numSimplices*/) {
  // Build the base into a scratch spacetime; its fixture vertices are numbered
  // 0..|V|-1 (see SimplicialProduct / buildExplicit), so |V| = getVertexCount().
  auto scratch = std::make_shared<Spacetime>();
  base_->build(scratch.get(), 0);
  const std::uint64_t numVertices = scratch->getVertexCount();

  // Collect the top (highest-dimensional) simplices as sorted vertex tuples.
  std::size_t topSize = 0;
  for (const auto &s : scratch->getSimplices())
    topSize = std::max(topSize, static_cast<std::size_t>(s->size()));

  std::vector<std::vector<std::uint64_t>> tops;
  for (const auto &s : scratch->getSimplices()) {
    if (static_cast<std::size_t>(s->size()) != topSize) continue;
    std::vector<std::uint64_t> ids;
    ids.reserve(topSize);
    for (const auto &v : s->getVertices()) ids.push_back(v->getId());
    std::sort(ids.begin(), ids.end());
    tops.push_back(std::move(ids));
  }

  // Nothing to refine (e.g. an empty complex): rebuild the base verbatim.
  if (tops.empty()) {
    buildExplicit(spacetime, static_cast<std::size_t>(numVertices), tops);
    return;
  }

  // Deterministic choice: star the lexicographically smallest top simplex so
  // the subdivision is reproducible regardless of getSimplices() order.
  std::sort(tops.begin(), tops.end());
  const std::vector<std::uint64_t> starred = tops.front();
  tops.erase(tops.begin());

  // Stellar 1 -> (n+1) move: replace `starred` with the cones from a fresh
  // interior vertex `center` over each of its facets.
  const std::uint64_t center = numVertices;
  for (std::size_t omit = 0; omit < starred.size(); ++omit) {
    std::vector<std::uint64_t> cell;
    cell.reserve(starred.size());
    for (std::size_t i = 0; i < starred.size(); ++i)
      if (i != omit) cell.push_back(starred[i]);
    cell.push_back(center);
    tops.push_back(std::move(cell));
  }

  buildExplicit(spacetime, static_cast<std::size_t>(numVertices) + 1, tops);
}

} // namespace tessera::spacetime
