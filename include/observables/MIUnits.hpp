// Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
//
// Shared MI normalisation constant. Promoted from
// ``src/simulations/InteractionSimulation.cpp`` so the same value is used by
// every consumer of the MI → length map ℓ = −log(I / I_max):
//
//   • InteractionSimulation (Pachner cells, Regge action)
//   • WeightedSparseGraph::fromSpacetimeSkeleton (heat-kernel weights)
//   • MutualInformationProfile::buildSpacetime (edge lengths)
//
// kIMax = 2·log(2) is the maximum mutual information for a maximally-
// entangled qubit pair (each subsystem dim 2). For qudit-basis runs the
// correct value is 2·log(d); a follow-up (#39) makes this a config field
// rather than a global constant.

#pragma once

#include <cmath>

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

/// I_max for maximally-entangled qubit pairs. Used as the normalisation
/// in ℓ = −log(I / kIMax). Hardcoded 2·log(2) for now; #39 promotes this
/// to a HolographyConfig / InteractionSimulation parameter so qudit-basis
/// (v0.2) runs can override it to 2·log(4).
inline const double kIMax = 2.0 * std::log(2.0);

} // namespace tessera::observables
