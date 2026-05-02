// MIT License
// Copyright (c) 2025 Andrew Kelleher

#ifndef TESSERA_PACHNER_ADDMOVE_H
#define TESSERA_PACHNER_ADDMOVE_H

#include <memory>
#include <random>
#include <vector>

#include "mesh/ForwardDeclarations.h"
#include "spacetime/PachnerMove.h"
#include "spacetime/Spacetime.h"

namespace tessera {

/// (2, 2d) Pachner add (vertex insertion) with apply / rollback.
///
/// Picks a random N41 top simplex, finds its spatial face and the
/// adjacent simplex of opposite orientation.  Inserts a new vertex at
/// the shared spatial time slice, replacing the 2 simplices with 2d
/// new ones.  ``dN0 = +1``; ``dN41 = +(2d - 2) = +6`` in 4D;
/// ``dN32 = 0``.
///
/// Vertex relabeling: after the move commits, the new vertex's ID is
/// optionally swapped with a randomly-chosen existing vertex (per
/// [BGL] Sec. 2.2.1).  Toggle via ``setRelabelEnabled(bool)`` at
/// construction time; default is enabled.  Rollback un-swaps before
/// removing the new vertex.
class AddMove : public PachnerMove {
public:
  AddMove(Spacetime *st, std::mt19937 *rng, bool relabelEnabled = true);
  AddMove(Spacetime *st, std::uint64_t seed, bool relabelEnabled = true);

  bool propose() override;
  int dN0() const override { return 1; }
  int dN41() const override { return dN41_; }
  int dN32() const override { return 0; }
  double metropolisLogPrefactor() const override { return logPrefactor_; }
  bool apply() override;
  void rollback() override;
  bool isApplied() const override { return applied_; }
  std::vector<std::uint64_t> touchedVertexIds() const override;
  std::string moveType() const override { return "add"; }

private:
  Spacetime *st_;
  std::unique_ptr<std::mt19937> ownedRng_;
  std::mt19937 *rng_;
  bool relabelEnabled_;

  // Filled by propose()
  bool proposed_ = false;
  SimplexPtr sigma_, sigmaAdj_, spatialFacet_;
  VertexPtr vertA_, vertB_;
  VertexPtrs spatialVerts_;
  VertexPtrs sigmaVerts_, sigmaAdjVerts_;
  double spatialTime_ = 0.0;
  std::vector<std::uint64_t> touchedIds_;
  int dN41_ = 0;
  double logPrefactor_ = 0.0;

  // Filled by apply()
  bool applied_ = false;
  VertexPtr newVert_;                       // the inserted vertex
  VertexPtr swapPartner_ = nullptr;         // null if no relabel happened
  std::vector<SimplexPtr> createdSimplices_;
  Edges createdEdges_;
};

}  // namespace tessera

#endif  // TESSERA_PACHNER_ADDMOVE_H
