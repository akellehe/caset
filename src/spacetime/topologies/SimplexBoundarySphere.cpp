// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "spacetime/topologies/SimplexBoundarySphere.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace tessera::spacetime {

void SimplexBoundarySphere::build(Spacetime *spacetime, int /*numSimplices*/) {
  if (n_ < 1) throw std::invalid_argument("SimplexBoundarySphere requires n >= 1");
  // S^n = ∂Δ^{n+1}: n+2 vertices; each top n-simplex omits exactly one vertex.
  const std::size_t numV = static_cast<std::size_t>(n_) + 2;
  std::vector<std::vector<std::uint64_t>> tops;
  tops.reserve(numV);
  for (std::size_t omit = 0; omit < numV; ++omit) {
    std::vector<std::uint64_t> s;
    s.reserve(numV - 1);
    for (std::size_t v = 0; v < numV; ++v)
      if (v != omit) s.push_back(static_cast<std::uint64_t>(v));
    tops.push_back(std::move(s));
  }
  buildExplicit(spacetime, numV, tops);
}

} // namespace tessera::spacetime
