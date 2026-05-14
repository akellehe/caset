// MIT License
// Copyright (c) 2025 Andrew Kelleher

#include "spacetime/pachner/IFlipMove.h"

#include <cmath>
#include <random>

#include "mesh/Edge.h"
#include "mesh/Simplex.h"
#include "mesh/SimplexOrientation.h"
#include "mesh/Vertex.h"

namespace tessera {

IFlipMove::IFlipMove(Spacetime *st, std::mt19937 *rng)
    : st_(st), ownedRng_(nullptr), rng_(rng) {}

IFlipMove::IFlipMove(Spacetime *st, std::uint64_t seed)
    : st_(st),
      ownedRng_(std::make_unique<std::mt19937>(seed)),
      rng_(ownedRng_.get()) {}

bool IFlipMove::propose() {
  if (proposed_) return false;
  using namespace pachner_detail;
  const int d = spacetimeDim(*st_);
  const int dPlus1 = d + 1;

  SimplexPtr sigma = st_->getRandomTopSimplex();
  if (!sigma) return false;

  const auto &edges = sigma->getEdges();
  if (edges.empty()) return false;
  std::uniform_int_distribution<std::size_t> edgeDist(0, edges.size() - 1);
  EdgePtr edge = edges[edgeDist(*rng_)];

  VertexPtr v1 = edge->getSource();
  VertexPtr v2 = edge->getTarget();

  // Find all top simplices containing both endpoints.
  std::vector<SimplexPtr> sharing;
  for (const auto &s : v1->getSimplices()) {
    if (static_cast<int>(s->size()) == dPlus1 && s->hasVertex(v2)) {
      sharing.push_back(s);
    }
  }
  if (static_cast<int>(sharing.size()) != d) return false;

  // Collect all vertices: should be d+2 total.
  auto allVerts = pachner_detail::unionVerticesAcross(sharing);
  if (static_cast<int>(allVerts.size()) != d + 2) return false;

  // Separate shared (the 2 edge endpoints) and unique (the d others).
  VertexPtrs shared, unique;
  for (const auto &v : allVerts) {
    if (v->getId() == v1->getId() || v->getId() == v2->getId())
      shared.push_back(v);
    else
      unique.push_back(v);
  }
  if (shared.size() != 2 || static_cast<int>(unique.size()) != d) return false;

  // Manifold check: would either proposed new simplex already exist?
  // We look for a top simplex incident to unique[0] that contains all
  // unique vertices plus one shared vertex but is NOT one of the d
  // we're about to remove.  Matches CDT::iflip's check exactly.
  for (int i = 0; i < 2; ++i) {
    for (const auto &s : unique[0]->getSimplices()) {
      if (static_cast<int>(s->size()) != dPlus1) continue;
      bool isSharing = false;
      for (const auto &sh : sharing) {
        if (s == sh) { isSharing = true; break; }
      }
      if (isSharing) continue;
      if (!s->hasVertex(shared[i])) continue;
      bool hasAll = true;
      for (const auto &u : unique) {
        if (!s->hasVertex(u)) { hasAll = false; break; }
      }
      if (hasAll) return false;  // would create duplicate
    }
  }

  // Old orientation counts.
  int oldN41 = 0, oldN32 = 0;
  for (const auto &s : sharing) {
    if (isN41Type(s, d)) ++oldN41;
    else if (isN32Type(s, d)) ++oldN32;
  }

  // Build 2 new simplex tuples: each has all d unique + 1 of 2 shared.
  std::vector<VertexPtrs> proposedNew;
  for (int i = 0; i < 2; ++i) {
    VertexPtrs nv(unique.begin(), unique.end());
    nv.push_back(shared[i]);
    if (static_cast<int>(nv.size()) != dPlus1) return false;
    proposedNew.push_back(std::move(nv));
  }

  for (const auto &nv : proposedNew) {
    if (!isValidCDTOrientation(nv, d)) return false;
  }

  int newN41 = 0, newN32 = 0;
  for (const auto &nv : proposedNew) {
    if (isN41TypeVerts(nv, d)) ++newN41;
    else if (isN32TypeVerts(nv, d)) ++newN32;
  }

  oldSimplices_ = std::move(sharing);
  oldSimplexVerts_.reserve(oldSimplices_.size());
  for (const auto &s : oldSimplices_) {
    const auto &verts = s->getVertices();
    oldSimplexVerts_.emplace_back(verts.begin(), verts.end());
  }
  newSimplexVerts_ = std::move(proposedNew);
  dN41_ = newN41 - oldN41;
  dN32_ = newN32 - oldN32;

  // Combinatorial prefactor (matches CDT::iflip): log(N4 / (N4 - d + 2)).
  double N4 = static_cast<double>(st_->getSimplexCount());
  logPrefactor_ = std::log(N4) - std::log(N4 - d + 2);

  touchedIds_.reserve(d + 2);
  for (const auto &v : shared) touchedIds_.push_back(v->getId());
  for (const auto &v : unique) touchedIds_.push_back(v->getId());

  proposed_ = true;
  return true;
}

bool IFlipMove::apply() {
  if (!proposed_ || applied_) return false;

  for (const auto &s : oldSimplices_) st_->removeSimplex(s);

  createdSimplices_.reserve(newSimplexVerts_.size());
  for (const auto &nv : newSimplexVerts_) {
    auto r = st_->createSimplexTracked(nv);
    if (r.created) createdSimplices_.push_back(r.simplex);
    for (const auto &e : r.newEdges) createdEdges_.push_back(e);
  }

  applied_ = true;
  return true;
}

void IFlipMove::rollback() {
  if (!applied_) return;

  for (const auto &s : createdSimplices_) st_->removeSimplex(s);
  createdSimplices_.clear();

  pachner_detail::removeAndClearEdges(createdEdges_, st_);

  for (const auto &verts : oldSimplexVerts_) {
    st_->createSimplexTracked(verts);
  }

  applied_ = false;
}

std::vector<std::uint64_t> IFlipMove::touchedVertexIds() const {
  return touchedIds_;
}

}  // namespace tessera
