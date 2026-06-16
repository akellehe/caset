// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/CombinatorialDimension.h"

#include <algorithm>
#include <cstddef>

#include "mesh/Simplex.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

double CombinatorialDimension::compute(const std::shared_ptr<Spacetime> &spacetime) {
  if (spacetime == nullptr) return -1.0;
  std::size_t maxVerts = 0;
  for (const auto &simplex : spacetime->getSimplices()) {
    maxVerts = std::max(maxVerts, static_cast<std::size_t>(simplex->size()));
  }
  return maxVerts == 0 ? -1.0 : static_cast<double>(maxVerts) - 1.0;
}

}  // namespace tessera::cobordism
