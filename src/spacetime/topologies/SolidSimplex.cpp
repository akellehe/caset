// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "spacetime/topologies/SolidSimplex.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace tessera::spacetime {

void SolidSimplex::build(Spacetime *spacetime, int /*numSimplices*/) {
  if (n_ < 1) throw std::invalid_argument("SolidSimplex requires n >= 1");
  // Δ^n: a single top simplex on n+1 vertices.
  const std::size_t numV = static_cast<std::size_t>(n_) + 1;
  std::vector<std::uint64_t> s;
  s.reserve(numV);
  for (std::size_t v = 0; v < numV; ++v) s.push_back(static_cast<std::uint64_t>(v));
  buildExplicit(spacetime, numV, {std::move(s)});
}

} // namespace tessera::spacetime
