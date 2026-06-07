// MIT License
// Copyright (c) 2025 Andrew Kelleher

#ifndef TESSERA_PACHNER_FLIPMOVE_H
#define TESSERA_PACHNER_FLIPMOVE_H

#include <memory>
#include <random>
#include <vector>

#include "mesh/ForwardDeclarations.h"
#include "spacetime/PachnerMove.h"
#include "spacetime/Spacetime.h"

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

/// (2, d) Pachner flip with apply / rollback.
///
/// Removes 2 d-simplices sharing a (d-1)-face and creates d new
/// d-simplices sharing an edge.  ``dN0 = 0``; ``ΔN4 = d - 2 = +2``
/// in 4D.  Inverse: :class:`IFlipMove` (the (d, 2) move).
///
/// Metropolis log prefactor: ``log(N4 / (N4 + d - 2))``
/// (combinatorial selection ratio between forward and reverse moves).
/// See ``CDT::flip`` for the original (non-transactional) implementation.
class FlipMove : public PachnerMove {
public:
  FlipMove(Spacetime *st, std::mt19937 *rng,
           PachnerMode mode = PachnerMode::CDT, bool boundaryFixed = false);
  FlipMove(Spacetime *st, std::uint64_t seed,
           PachnerMode mode = PachnerMode::CDT, bool boundaryFixed = false);

  bool propose() override;
  int dN0() const override { return 0; }
  int dN41() const override { return dN41_; }
  int dN32() const override { return dN32_; }
  double metropolisLogPrefactor() const override { return logPrefactor_; }
  bool apply() override;
  void rollback() override;
  bool isApplied() const override { return applied_; }
  std::vector<std::uint64_t> touchedVertexIds() const override;
  std::string moveType() const override { return "flip"; }

private:
  Spacetime *st_;
  std::unique_ptr<std::mt19937> ownedRng_;
  std::mt19937 *rng_;

  /// Pre-geometric 2→(d+1) flip: the same combinatorial replacement as
  /// the CDT path but without the time-slice / CDT-orientation guard,
  /// with the move dimension read off the chosen top cell and a
  /// manifold-preservation check (apex edge must not pre-exist).  In
  /// boundary-fixed mode the operative facet must be interior.
  bool proposePreGeometric();

  bool proposed_ = false;
  std::vector<VertexPtrs> oldSimplexVerts_;
  std::vector<VertexPtrs> newSimplexVerts_;
  std::vector<std::uint64_t> touchedIds_;
  int dN41_ = 0;
  int dN32_ = 0;
  double logPrefactor_ = 0.0;

  bool applied_ = false;
  std::vector<SimplexPtr> oldSimplices_;
  // Vertex tuples of simplices we actually created (see ShiftMove for the
  // staleness-bug rationale).
  std::vector<VertexPtrs> createdSimplexVerts_;
  Edges createdEdges_;
};

}  // namespace tessera

#endif  // TESSERA_PACHNER_FLIPMOVE_H
