// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "observables/Observable.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::observables {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;

double Observable::compute(const std::shared_ptr<Spacetime> &spacetime) { return 0.0; }
double Observable::update(const std::shared_ptr<Spacetime> &spacetime) { return 0.0; }

} // namespace tessera::observables

