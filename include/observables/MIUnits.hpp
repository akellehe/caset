// MIT License -- Copyright (c) 2025 Andrew Kelleher
//
// Shared MI normalisation constant. Promoted from
// ``src/quantum/interaction_simulation.cpp`` so the same value is used by
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

namespace tessera::observables {

/// I_max for maximally-entangled qubit pairs. Used as the normalisation
/// in ℓ = −log(I / kIMax). Hardcoded 2·log(2) for now; #39 promotes this
/// to a HolographyConfig / InteractionSimulation parameter so qudit-basis
/// (v0.2) runs can override it to 2·log(4).
inline const double kIMax = 2.0 * std::log(2.0);

} // namespace tessera::observables
