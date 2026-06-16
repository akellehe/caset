// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

//
// Created by Andrew Kelleher on 11/10/25.
//

#ifndef TESSERA_SPACETIMEVOLUME_H
#define TESSERA_SPACETIMEVOLUME_H

#include <memory>
#include "Observable.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
// === cross-subsystem fwd-decls ===
namespace tessera::spacetime {
  class Spacetime;
}
namespace tessera::observables {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;
class SpacetimeVolume : public Observable {
  public:
    double compute(const std::shared_ptr<Spacetime> &spacetime) override;
    double update(const std::shared_ptr<Spacetime> &spacetime) override;
};
}

#endif //TESSERA_SPACETIMEVOLUME_H