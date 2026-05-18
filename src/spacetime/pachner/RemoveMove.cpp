// MIT License
// Copyright (c) 2025 Andrew Kelleher

#include "spacetime/pachner/RemoveMove.h"

#include <cmath>
#include <map>

#include "mesh/Edge.h"
#include "mesh/Simplex.h"
#include "mesh/SimplexOrientation.h"
#include "mesh/Vertex.h"

namespace tessera {

RemoveMove::RemoveMove(Spacetime *st, std::mt19937 *rng)
    : st_(st), ownedRng_(nullptr), rng_(rng) {}

RemoveMove::RemoveMove(Spacetime *st, std::uint64_t seed)
    : st_(st),
      ownedRng_(std::make_unique<std::mt19937>(seed)),
      rng_(ownedRng_.get()) {}

bool RemoveMove::propose() {
  if (proposed_) return false;
  using namespace pachner_detail;
  const int d = spacetimeDim(*st_);
  const int dPlus1 = d + 1;
  const int requiredOrder = 2 * d;

  // Mirrors CDT::remove's blind-guessing strategy.
  // Use the spacetime's RNG-free getRandomVertex for parity with the
  // existing code; we don't get to influence vertex selection here.
  // (CDT::remove does the same.)
  VertexPtr v = st_->getRandomVertex();
  if (!v) return false;

  std::vector<SimplexPtr> incident;
  for (const auto &s : v->getSimplices()) {
    if (static_cast<int>(s->size()) == dPlus1) incident.push_back(s);
  }
  if (static_cast<int>(incident.size()) != requiredOrder) return false;

  // Verify structural prerequisites: all incident must be N41-type.
  for (const auto &s : incident) {
    if (!isN41Type(s, d)) return false;
  }

  // Collect d+2 other vertices (excluding v) and their per-simplex
  // counts.  Spatial vertices appear in 2(d-1) of the 2d simplices;
  // the 2 non-spatial each appear in d.
  std::map<std::uint64_t, VertexPtr> otherVerts;
  std::map<std::uint64_t, int> vertCounts;
  for (const auto &s : incident) {
    for (const auto &vert : s->getVertices()) {
      if (vert->getId() == v->getId()) continue;
      otherVerts[vert->getId()] = vert;
      vertCounts[vert->getId()]++;
    }
  }
  if (static_cast<int>(otherVerts.size()) != d + 2) return false;

  VertexPtr vertA = nullptr, vertB = nullptr;
  VertexPtrs spatialVerts;
  for (const auto &[vid, count] : vertCounts) {
    if (count == d) {
      if (!vertA) vertA = otherVerts[vid];
      else if (!vertB) vertB = otherVerts[vid];
      else return false;
    } else if (count == 2 * (d - 1)) {
      spatialVerts.push_back(otherVerts[vid]);
    } else {
      return false;
    }
  }
  if (!vertA || !vertB || static_cast<int>(spatialVerts.size()) != d)
    return false;

  // Combinatorial deltas.
  dN41_ = -(2 * d - 2);

  // Combinatorial prefactor (matches CDT::remove): log(N0 / N41after).
  double N41 = static_cast<double>(st_->getN41());
  double N0 = static_cast<double>(st_->getVertexCount());
  double N41after = N41 + dN41_;
  if (N41after <= 0) return false;
  logPrefactor_ = std::log(N0) - std::log(N41after);

  // Capture for apply.
  v_ = v;
  incident_ = std::move(incident);
  vertA_ = vertA;
  vertB_ = vertB;
  spatialVerts_ = std::move(spatialVerts);

  // Capture vertex tuples for rollback.
  incidentVerts_.reserve(incident_.size());
  for (const auto &s : incident_) {
    const auto &verts = s->getVertices();
    incidentVerts_.emplace_back(verts.begin(), verts.end());
  }

  // Touched: v + d spatial + vertA + vertB.
  touchedIds_.reserve(d + 3);
  touchedIds_.push_back(v_->getId());
  for (const auto &sv : spatialVerts_) touchedIds_.push_back(sv->getId());
  touchedIds_.push_back(vertA_->getId());
  touchedIds_.push_back(vertB_->getId());

  proposed_ = true;
  return true;
}

bool RemoveMove::apply() {
  if (!proposed_ || applied_) return false;

  // 1. Capture edge data BEFORE deletion (for rollback).
  // Mirrors CDT::remove's edge-cleanup loop.
  vertexId_ = v_->getId();
  vertexCoords_ = v_->getCoordinates();

  // Snapshot incident edges (in + out) and their squared lengths.
  // (We can't capture EdgePtr — those slots get freed by EdgeList::
  // remove.)
  for (const auto &e : v_->getInEdges()) {
    deletedEdges_.push_back({e->getSource(), e->getTarget(),
                             e->getSquaredLength()});
  }
  for (const auto &e : v_->getOutEdges()) {
    deletedEdges_.push_back({e->getSource(), e->getTarget(),
                             e->getSquaredLength()});
  }

  // 2. Remove the 2d incident simplices.
  for (const auto &s : incident_) st_->removeSimplex(s);

  // 3. Remove edges incident to v from both endpoints + the global list.
  // Mirrors CDT::remove's cleanup.
  Edges inCopy(v_->getInEdges().begin(), v_->getInEdges().end());
  for (const auto &e : inCopy) {
    e->getSource()->removeOutEdge(e);
    v_->removeInEdge(e);
    st_->getEdgeList()->remove(e);
  }
  Edges outCopy(v_->getOutEdges().begin(), v_->getOutEdges().end());
  for (const auto &e : outCopy) {
    e->getTarget()->removeInEdge(e);
    v_->removeOutEdge(e);
    st_->getEdgeList()->remove(e);
  }
  (void)st_->removeIfIsolated(v_);

  // 4. Create 2 replacement simplices.
  VertexPtrs verts1(spatialVerts_.begin(), spatialVerts_.end());
  verts1.push_back(vertA_);
  VertexPtrs verts2(spatialVerts_.begin(), spatialVerts_.end());
  verts2.push_back(vertB_);

  auto r1 = st_->createSimplexTracked(verts1);
  if (r1.created) createdSimplexVerts_.push_back(verts1);
  for (const auto &e : r1.newEdges) createdEdges_.push_back(e);

  auto r2 = st_->createSimplexTracked(verts2);
  if (r2.created) createdSimplexVerts_.push_back(verts2);
  for (const auto &e : r2.newEdges) createdEdges_.push_back(e);

  applied_ = true;
  return true;
}

void RemoveMove::rollback() {
  if (!applied_) return;

  // 1. Remove the 2 replacement simplices.  Resolve by verts at
  // rollback time (see ShiftMove for the staleness-bug rationale).
  for (const auto &verts : createdSimplexVerts_) {
    if (auto s = st_->findSimplexByVerts(verts)) st_->removeSimplex(s);
  }
  createdSimplexVerts_.clear();

  // 2. Remove freshly-inserted edges (from the replacement simplices).
  pachner_detail::removeAndClearEdges(createdEdges_, st_);

  // 3. Recreate the deleted vertex with its original ID and coordinates.
  v_ = st_->createVertex(vertexId_, vertexCoords_);

  // 4. Reinsert the deleted edges.  All endpoints still exist
  // (only v_ was removed, and we just recreated it).  Need to be
  // careful: the EdgeRecord's source/target pointers refer to
  // pre-deletion VertexPtrs.  v_'s pointer is fresh (from step 3),
  // so swap any reference to the *old* v_ pointer with the new one.
  // Actually — the way createVertex works, it allocates a new Vertex
  // and returns its pointer.  The OLD v_ pointer (from before
  // step 3) is invalid, but we already overwrote v_ with the new
  // pointer in step 3.  EdgeRecord captured pointers BEFORE we
  // overwrote v_, so those still point to the old (deleted) Vertex.
  //
  // To handle this cleanly: identify edges that reference the deleted
  // vertex by ID match (vertexId_) rather than by pointer comparison.
  // For each EdgeRecord:
  //   - If src->getId() == vertexId_ → use new v_ as source, target unchanged
  //   - Else if tgt->getId() == vertexId_ → use new v_ as target
  //   - Else: shouldn't happen (every captured edge was incident to v).
  for (const auto &er : deletedEdges_) {
    VertexPtr src = (er.source->getId() == vertexId_) ? v_ : er.source;
    VertexPtr tgt = (er.target->getId() == vertexId_) ? v_ : er.target;
    // After re-creating the vertex with the original ID, the OLD
    // pointer captured in EdgeRecord may have been freed.  We rely on
    // the ID match above; ID-stable accessors are sufficient.
    auto r = st_->getEdgeList()->tryAdd(src, tgt, er.squaredLength);
    src->addOutEdge(r.first);
    tgt->addInEdge(r.first);
  }
  deletedEdges_.clear();

  // 5. Recreate the 2d removed simplices.  Their vertex tuples
  // contain the OLD v_ pointer (captured pre-deletion).  Swap to the
  // new pointer based on ID.
  for (const auto &origVerts : incidentVerts_) {
    VertexPtrs verts;
    verts.reserve(origVerts.size());
    for (const auto &vp : origVerts) {
      verts.push_back(vp->getId() == vertexId_ ? v_ : vp);
    }
    st_->createSimplexTracked(verts);
  }

  applied_ = false;
}

std::vector<std::uint64_t> RemoveMove::touchedVertexIds() const {
  return touchedIds_;
}

}  // namespace tessera
