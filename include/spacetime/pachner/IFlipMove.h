// MIT License
// Copyright (c) 2025 Andrew Kelleher

#ifndef TESSERA_PACHNER_IFLIPMOVE_H
#define TESSERA_PACHNER_IFLIPMOVE_H

#include <memory>
#include <random>
#include <vector>

#include "mesh/ForwardDeclarations.h"
#include "spacetime/PachnerMove.h"
#include "spacetime/Spacetime.h"

namespace tessera {

/// Inverse (d, 2) Pachner flip with apply / rollback.
///
/// Removes d d-simplices sharing an edge and creates 2 new
/// d-simplices sharing a (d-1)-face.  ``dN0 = 0``;
/// ``ΔN4 = -(d - 2) = -2`` in 4D.  Inverse: :class:`FlipMove`.
///
/// Includes an explicit manifold-preservation check (matches
/// CDT::iflip): rejects if either of the two proposed new simplices
/// would already exist in the lattice (preventing duplicate-simplex
/// creation).  This makes iflip the only Pachner move that detects
/// dedupe in propose() — apply() will always make exactly the
/// advertised changes.
class IFlipMove : public PachnerMove {
public:
  IFlipMove(Spacetime *st, std::mt19937 *rng);
  IFlipMove(Spacetime *st, std::uint64_t seed);

  bool propose() override;
  int dN0() const override { return 0; }
  int dN41() const override { return dN41_; }
  int dN32() const override { return dN32_; }
  double metropolisLogPrefactor() const override { return logPrefactor_; }
  bool apply() override;
  void rollback() override;
  bool isApplied() const override { return applied_; }
  std::vector<std::uint64_t> touchedVertexIds() const override;
  std::string moveType() const override { return "iflip"; }

private:
  Spacetime *st_;
  std::unique_ptr<std::mt19937> ownedRng_;
  std::mt19937 *rng_;

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

#endif  // TESSERA_PACHNER_IFLIPMOVE_H
