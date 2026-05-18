// MIT License
// Copyright (c) 2025 Andrew Kelleher

#ifndef TESSERA_PACHNER_REMOVEMOVE_H
#define TESSERA_PACHNER_REMOVEMOVE_H

#include <memory>
#include <random>
#include <vector>

#include "mesh/ForwardDeclarations.h"
#include "spacetime/PachnerMove.h"
#include "spacetime/Spacetime.h"

namespace tessera {

/// (2d, 2) Pachner remove (vertex deletion) with apply / rollback.
///
/// Picks a random vertex with order 2d (all incident top simplices are
/// N41-type), removes the 2d simplices and the vertex, and creates 2
/// replacement simplices.  ``dN0 = -1``; ``dN41 = -(2d-2) = -6`` in 4D;
/// ``dN32 = 0``.  Inverse: :class:`AddMove`.
///
/// Rollback is the most involved of the move types: it has to recreate
/// the deleted vertex (with its original ID and coordinates), reinsert
/// the d edges incident to it (with their original squared lengths),
/// and recreate the 2d removed simplices.  All of that data is captured
/// by ``apply()`` before the geometry is mutated.
class RemoveMove : public PachnerMove {
public:
  RemoveMove(Spacetime *st, std::mt19937 *rng);
  RemoveMove(Spacetime *st, std::uint64_t seed);

  bool propose() override;
  int dN0() const override { return -1; }
  int dN41() const override { return dN41_; }
  int dN32() const override { return 0; }
  double metropolisLogPrefactor() const override { return logPrefactor_; }
  bool apply() override;
  void rollback() override;
  bool isApplied() const override { return applied_; }
  std::vector<std::uint64_t> touchedVertexIds() const override;
  std::string moveType() const override { return "remove"; }

private:
  Spacetime *st_;
  std::unique_ptr<std::mt19937> ownedRng_;
  std::mt19937 *rng_;

  // Filled by propose()
  bool proposed_ = false;
  VertexPtr v_;                                  // vertex to be removed
  std::vector<SimplexPtr> incident_;             // 2d incident top simplices
  std::vector<VertexPtrs> incidentVerts_;        // their vertex tuples
  VertexPtr vertA_, vertB_;                      // the two non-spatial vertices
  VertexPtrs spatialVerts_;                      // the d spatial vertices
  std::vector<std::uint64_t> touchedIds_;
  int dN41_ = 0;
  double logPrefactor_ = 0.0;

  // Captured by apply() — for rollback
  bool applied_ = false;
  // Edge data captured before deletion: (sourcePtr, targetPtr,
  // squaredLength).  EdgePtr alone is not enough because EdgeList::
  // remove invalidates the slot.
  struct EdgeRecord {
    VertexPtr source;
    VertexPtr target;
    double squaredLength;
  };
  std::vector<EdgeRecord> deletedEdges_;
  // Vertex tuples of the 2 replacement simplices we actually created in
  // apply() (see ShiftMove for the staleness-bug rationale).
  std::vector<VertexPtrs> createdSimplexVerts_;
  // Edges freshly inserted by createSimplexTracked when we built the
  // 2 replacement simplices (likely empty since they reuse existing
  // edges, but track for safety).
  Edges createdEdges_;
  // Captured vertex coordinates for rollback (unique vertex with
  // captured ID).
  std::uint64_t vertexId_ = 0;
  std::vector<double> vertexCoords_;
};

}  // namespace tessera

#endif  // TESSERA_PACHNER_REMOVEMOVE_H
