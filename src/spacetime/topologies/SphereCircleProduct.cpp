// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "spacetime/topologies/SphereCircleProduct.h"

#include <memory>

#include "spacetime/topologies/SimplexBoundarySphere.h"
#include "spacetime/topologies/SimplicialProduct.h"

namespace tessera::spacetime {

void SphereCircleProduct::build(Spacetime *spacetime, int /*numSimplices*/) {
  // S^2 x S^1 = ∂Δ^3 × ∂Δ^2, triangulated by the staircase product.
  SimplicialProduct product(std::make_shared<SimplexBoundarySphere>(2),
                            std::make_shared<SimplexBoundarySphere>(1));
  product.build(spacetime, 0);
}

} // namespace tessera::spacetime
