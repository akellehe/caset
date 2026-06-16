// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "spacetime/topologies/RealProjectivePlane.h"

#include <cstdint>
#include <vector>

namespace tessera::spacetime {

void RealProjectivePlane::build(Spacetime *spacetime, int /*numSimplices*/) {
  // Minimal 6-vertex RP² (hemi-icosahedron): 10 triangles on K_6. Every edge
  // lies in exactly two triangles (closed surface); χ = 6 - 15 + 10 = 1.
  const std::vector<std::vector<std::uint64_t>> tops{
      {0, 1, 2}, {0, 2, 3}, {0, 3, 4}, {0, 4, 5}, {0, 1, 5},
      {1, 2, 4}, {2, 3, 5}, {1, 3, 4}, {1, 3, 5}, {2, 4, 5}};
  buildExplicit(spacetime, 6, tops);
}

} // namespace tessera::spacetime
