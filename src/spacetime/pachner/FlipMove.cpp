// MIT License
// Copyright (c) 2025 Andrew Kelleher

#include "spacetime/pachner/FlipMove.h"

#include <cmath>
#include <random>

#include "mesh/Simplex.h"
#include "mesh/SimplexOrientation.h"
#include "mesh/Vertex.h"

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

FlipMove::FlipMove(Spacetime *st, std::mt19937 *rng)
    : st_(st), ownedRng_(nullptr), rng_(rng) {}

FlipMove::FlipMove(Spacetime *st, std::uint64_t seed)
    : st_(st),
      ownedRng_(std::make_unique<std::mt19937>(seed)),
      rng_(ownedRng_.get()) {}

bool FlipMove::propose() {
  if (proposed_) return false;
  using namespace pachner_detail;
  const int d = spacetimeDim(*st_);
  const int dPlus1 = d + 1;

  SimplexPtr sigma = st_->getRandomTopSimplex();
  if (!sigma) return false;

  const auto &facets = sigma->getFacets();
  if (facets.empty()) return false;

  std::uniform_int_distribution<std::size_t> facetDist(0, facets.size() - 1);
  SimplexPtr facet = facets[facetDist(*rng_)];

  // Need exactly 2 d-simplex cofaces.
  std::vector<SimplexPtr> topCofaces;
  for (const auto &cf : facet->getCofaces()) {
    if (static_cast<int>(cf->size()) == dPlus1) topCofaces.push_back(cf);
  }
  if (topCofaces.size() != 2) return false;

  SimplexPtr s1 = topCofaces[0];
  SimplexPtr s2 = topCofaces[1];

  // Collect d+2 unique vertices: d shared + 2 unique.
  auto allVerts = pachner_detail::unionVerticesAcross(
      Simplices{s1, s2});
  if (static_cast<int>(allVerts.size()) != d + 2) return false;

  VertexPtrs shared, unique;
  for (const auto &v : allVerts) {
    if (s1->hasVertex(v) && s2->hasVertex(v)) shared.push_back(v);
    else unique.push_back(v);
  }
  if (static_cast<int>(shared.size()) != d ||
      static_cast<int>(unique.size()) != 2) return false;

  // Old orientation counts.
  int oldN41 = 0, oldN32 = 0;
  for (const auto &s : {s1, s2}) {
    if (isN41Type(s, d)) ++oldN41;
    else if (isN32Type(s, d)) ++oldN32;
  }

  // Build d new simplex tuples: each has both unique + (d-1) of d shared.
  std::vector<VertexPtrs> proposedNew;
  for (int skip = 0; skip < d; ++skip) {
    VertexPtrs nv;
    nv.reserve(dPlus1);
    for (int i = 0; i < d; ++i) {
      if (i != skip) nv.push_back(shared[i]);
    }
    nv.push_back(unique[0]);
    nv.push_back(unique[1]);
    if (static_cast<int>(nv.size()) != dPlus1) return false;
    proposedNew.push_back(std::move(nv));
  }

  // Reject if any new simplex would have a non-CDT orientation.
  for (const auto &nv : proposedNew) {
    if (!isValidCDTOrientation(nv, d)) return false;
  }

  int newN41 = 0, newN32 = 0;
  for (const auto &nv : proposedNew) {
    if (isN41TypeVerts(nv, d)) ++newN41;
    else if (isN32TypeVerts(nv, d)) ++newN32;
  }

  // Capture state for apply / rollback.
  oldSimplices_ = {s1, s2};
  oldSimplexVerts_.reserve(2);
  for (const auto &s : oldSimplices_) {
    const auto &verts = s->getVertices();
    oldSimplexVerts_.emplace_back(verts.begin(), verts.end());
  }
  newSimplexVerts_ = std::move(proposedNew);
  dN41_ = newN41 - oldN41;
  dN32_ = newN32 - oldN32;

  // Combinatorial prefactor (matches CDT::flip): log(N4 / (N4 + d - 2)).
  double N4 = static_cast<double>(st_->getSimplexCount());
  logPrefactor_ = std::log(N4) - std::log(N4 + d - 2);

  touchedIds_.reserve(d + 2);
  for (const auto &v : shared) touchedIds_.push_back(v->getId());
  for (const auto &v : unique) touchedIds_.push_back(v->getId());

  proposed_ = true;
  return true;
}

bool FlipMove::apply() {
  if (!proposed_ || applied_) return false;

  for (const auto &s : oldSimplices_) st_->removeSimplex(s);

  createdSimplexVerts_.reserve(newSimplexVerts_.size());
  for (const auto &nv : newSimplexVerts_) {
    auto r = st_->createSimplexTracked(nv);
    if (r.created) createdSimplexVerts_.push_back(nv);
    for (const auto &e : r.newEdges) createdEdges_.push_back(e);
  }

  applied_ = true;
  return true;
}

void FlipMove::rollback() {
  if (!applied_) return;

  // Resolve created simplices by verts at rollback time (see ShiftMove).
  for (const auto &verts : createdSimplexVerts_) {
    if (auto s = st_->findSimplexByVerts(verts)) st_->removeSimplex(s);
  }
  createdSimplexVerts_.clear();

  pachner_detail::removeAndClearEdges(createdEdges_, st_);

  for (const auto &verts : oldSimplexVerts_) {
    st_->createSimplexTracked(verts);
  }

  applied_ = false;
}

std::vector<std::uint64_t> FlipMove::touchedVertexIds() const {
  return touchedIds_;
}

}  // namespace tessera
