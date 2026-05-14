// MIT License
// Copyright (c) 2025 Andrew Kelleher

#ifndef TESSERA_PACHNERMOVE_H
#define TESSERA_PACHNERMOVE_H

#include <cstdint>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "mesh/ForwardDeclarations.h"
#include "mesh/Simplex.h"
#include "mesh/SimplexOrientation.h"
#include "mesh/Vertex.h"
#include "spacetime/Spacetime.h"

namespace tessera {

namespace pachner_detail {

/// Check that a proposed simplex vertex set has a valid CDT
/// orientation — one of (d,1), (1,d), (d-1,2), (2,d-1) — and spans
/// exactly 2 time slices.  Mirrors the static helper in CDT.cpp; lives
/// here so the move classes can share it.
inline bool isValidCDTOrientation(const VertexPtrs &verts, int d) {
  std::unordered_set<std::uint64_t> times;
  for (const auto &v : verts) {
    times.insert(static_cast<std::uint64_t>(v->getTime()));
  }
  if (times.size() != 2) return false;
  auto orient = SimplexOrientation::orientationOf(verts);
  auto [ti, tf] = orient.numeric();
  if ((ti == d && tf == 1) || (ti == 1 && tf == d)) return true;
  if ((ti == d - 1 && tf == 2) || (ti == 2 && tf == d - 1)) return true;
  return false;
}

inline bool isN41Type(const SimplexPtr &s, int d) {
  auto [ti, tf] = s->getOrientation().numeric();
  return (ti == d && tf == 1) || (ti == 1 && tf == d);
}

inline bool isN32Type(const SimplexPtr &s, int d) {
  auto [ti, tf] = s->getOrientation().numeric();
  return (ti == d - 1 && tf == 2) || (ti == 2 && tf == d - 1);
}

inline bool isN41TypeVerts(const VertexPtrs &verts, int d) {
  auto [ti, tf] = SimplexOrientation::orientationOf(verts).numeric();
  return (ti == d && tf == 1) || (ti == 1 && tf == d);
}

inline bool isN32TypeVerts(const VertexPtrs &verts, int d) {
  auto [ti, tf] = SimplexOrientation::orientationOf(verts).numeric();
  return (ti == d - 1 && tf == 2) || (ti == 2 && tf == d - 1);
}

inline int spacetimeDim(const Spacetime &st) {
  return st.getMetric()->getSignature()->getDimensions();
}

/// Detach every edge in ``edges`` from its endpoints, remove it from
/// the spacetime's EdgeList, then clear the container. Used by all
/// five Pachner moves at the end of ``rollback()`` to undo edges that
/// ``apply()`` freshly inserted.
inline void removeAndClearEdges(Edges &edges, Spacetime *st) {
  for (const auto &e : edges) {
    e->getSource()->removeOutEdge(e);
    e->getTarget()->removeInEdge(e);
    st->getEdgeList()->remove(e);
  }
  edges.clear();
}

/// Order-preserving union of every simplex's vertex list,
/// de-duplicated by vertex ID. Used by FlipMove / IFlipMove /
/// ShiftMove during ``propose()`` to build the (d+2)-vertex span of
/// adjacent simplices before checking the orientation constraint.
/// Hash-set dedup makes this O(n) over the total vertex count instead
/// of the inlined O(n²) loop the move classes used to carry.
template <typename SimplexRange>
inline VertexPtrs unionVerticesAcross(SimplexRange const &simplices) {
  VertexPtrs out;
  std::unordered_set<std::uint64_t> seen;
  for (const auto &s : simplices) {
    for (const auto &v : s->getVertices()) {
      if (seen.insert(v->getId()).second) out.push_back(v);
    }
  }
  return out;
}

}  // namespace pachner_detail


/// Transactional Pachner move: a propose-apply-rollback wrapper around
/// the geometric mutations of CDT::add / remove / flip / iflip / shift.
///
/// The interface separates a move's three life-cycle phases:
///
///   1. ``propose()`` — read-only.  Selects a target (random simplex /
///      facet / edge / vertex), validates that the move can be applied
///      to it, and records all the data needed to commit.  Does not
///      mutate the spacetime.  Returns ``false`` if no eligible target
///      can be found within the move's retry budget.
///
///   2. ``apply()`` — mutating.  Commits the proposed move to the
///      spacetime.  Builds an internal undo log (created simplices,
///      freshly-inserted edges, removed simplex vertex tuples, etc.)
///      that ``rollback()`` consumes.  Returns ``true`` on success.
///
///   3. ``rollback()`` — mutating.  Replays the undo log in reverse,
///      restoring the spacetime to the byte-identical state it was in
///      before ``apply()``.  Idempotent: a second call is a no-op.
///
/// The combinatorial Δ in (N0, N41, N32) is published by
/// ``dN0() / dN41() / dN32()`` after a successful ``propose()``, so the
/// caller (typically ``CDT::add()`` / etc.) can plug those into its
/// own action computation.  The base class deliberately *does not*
/// compute ΔS — the move is purely about geometry, not the action it
/// happens to be sampled against.
///
/// Locked-in characterization (see
/// docs/source/modularity-plan.md, "Discoveries from the safety-net pass"):
///
/// * Edges added by ``apply()`` are recorded by EdgePtr identity (not
///   by fingerprint hash, which is unstable under
///   ``swapVertexLabels``).
/// * ``apply()`` does not force facet/coface registration on newly
///   created simplices.  Coface registration is tessera's lazy
///   responsibility (triggered on the next ``getFacets`` walk).
/// * ``rollback()`` for moves that delete edges
///   (``RemoveMove``) must capture the deleted edges'
///   ``(sourceId, targetId, squaredLength)`` so it can reinsert them.
class PachnerMove {
public:
  virtual ~PachnerMove() = default;

  /// Pick a target and validate it.  No spacetime mutation.
  /// Returns ``true`` on success, ``false`` if no eligible target.
  virtual bool propose() = 0;

  /// Combinatorial change in vertex count if this move is applied.
  /// Valid only after a successful ``propose()``.
  virtual int dN0() const = 0;
  /// Combinatorial change in N41-type top-simplex count.
  virtual int dN41() const = 0;
  /// Combinatorial change in N32-type top-simplex count.
  virtual int dN32() const = 0;

  /// Log of the Metropolis combinatorial prefactor
  /// ``log(g(T'→T) · P_l(T') / [g(T→T') · P_l(T)])`` ([BGL] eq. 26).
  /// 0.0 for self-symmetric moves (shift) and the (2,d)/(d,2)
  /// flips; non-trivial for add/remove because of vertex selection.
  /// Valid only after a successful ``propose()``.
  virtual double metropolisLogPrefactor() const = 0;

  /// Commit the proposed move.  Builds the undo log.
  /// Must be called at most once per object.  Returns ``true`` on
  /// success.
  virtual bool apply() = 0;

  /// Replay the undo log in reverse.  After this returns,
  /// the spacetime is byte-identical to its state before ``apply()``.
  /// Idempotent.
  virtual void rollback() = 0;

  /// True iff ``apply()`` has been called and not yet rolled back.
  virtual bool isApplied() const = 0;

  /// IDs of the vertices whose neighborhood the move
  /// re-arranges, for informed proposal scoring (community-aware
  /// modularity sweeps).  Valid after a successful ``propose()``.
  virtual std::vector<std::uint64_t> touchedVertexIds() const = 0;

  /// Move-type tag for logging / acceptance-rate accounting.
  /// One of: ``"add"``, ``"remove"``, ``"flip"``, ``"iflip"``,
  /// ``"shift"``.
  virtual std::string moveType() const = 0;
};

}  // namespace tessera

#endif  // TESSERA_PACHNERMOVE_H
