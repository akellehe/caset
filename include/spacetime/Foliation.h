// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

//
// Created by Andrew Kelleher on 12/21/25.
//

#ifndef TESSERA_FOLIATION_H
#define TESSERA_FOLIATION_H
#include <cstdint>

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
enum class Foliation : std::uint8_t {
  NONE = 0,
  PREFERRED = 1
};
} // namespace tessera::spacetime

#endif //TESSERA_FOLIATION_H