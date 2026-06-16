// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_TOROID_H
#define TESSERA_TOROID_H

#include "Topology.h"

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
class Spacetime;

/// # Toroidal Topology \f$ T^{d-1} \f$
///
/// Spatial slices have the topology of a \f$(d\!-\!1)\f$-torus, giving a spacetime
/// manifold \f$ \mathcal{M} \cong T^{d-1} \times S^1 \f$ with periodic boundary
/// conditions in both space and time. This is the default topology for CDT
/// simulations and the most commonly used in the literature.
///
/// Periodic time means the last time slice \f$ t = T \f$ is identified with the
/// first \f$ t = 0 \f$, so the triangulation has no temporal boundaries.
///
/// The build creates multiple time layers, each grown from a seed \f$ d \f$-simplex
/// via iterated coning in random (forward/backward) time directions.
///
class Toroid : public Topology {
  public:
    /// Build a toroidal triangulation with multiple time layers.
    ///
    /// Cones exterior facets in random \f$ \pm t \f$ directions within each layer,
    /// then advances to the next time slice. The result is a multi-layer simplicial
    /// complex where each layer spans one unit of coordinate time.
    ///
    /// @param spacetime The spacetime to populate
    /// @param numSimplices Target number of top-dimensional simplices
    void build(Spacetime *spacetime, int numSimplices) override;
};

} // namespace tessera::spacetime

#endif //TESSERA_TOROID_H
