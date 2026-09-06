// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "spacetime/topologies/PolygonCircle.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace tessera::spacetime {

void PolygonCircle::build(Spacetime *spacetime, int /*numSimplices*/) {
  if (n_ < 3)
    throw std::invalid_argument(
        "PolygonCircle requires n >= 3 (a 2-gon repeats the vertex pair {0,1} and is "
        "not a simplicial complex)");
  // S^1 as the n-gon: edge i joins vertex i to vertex i+1, the last edge closes
  // the loop. Each edge is written with ascending vertex ids, the reference
  // orientation every ChainComplex built over it uses, so the wrap edge is
  // {0, n-1}.
  const std::size_t numV = static_cast<std::size_t>(n_);
  std::vector<std::vector<std::uint64_t>> tops;
  tops.reserve(numV);
  for (std::size_t i = 0; i + 1 < numV; ++i)
    tops.push_back({static_cast<std::uint64_t>(i), static_cast<std::uint64_t>(i + 1)});
  tops.push_back({0, static_cast<std::uint64_t>(numV - 1)});
  buildExplicit(spacetime, numV, tops);
}

} // namespace tessera::spacetime
