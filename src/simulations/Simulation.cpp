// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "simulations/Simulation.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::spacetime {}
namespace tessera::simulations {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::quantum;

void Simulation::tune(std::function<void(int,int)> /*progress*/) {}
void Simulation::thermalize() {}

} // namespace tessera::simulations
