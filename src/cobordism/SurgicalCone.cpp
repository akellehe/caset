// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/SurgicalCone.h"

#include <algorithm>
#include <bit>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "cobordism/ChainComplex.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
#include "spacetime/Metric.h"
#include "spacetime/Signature.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

SurgicalCone::SurgicalCone(Spacetime *spacetime) : st_(spacetime) {}

SurgicalCone::~SurgicalCone() = default;

std::size_t SurgicalCone::topVerts() const {
  if (st_ == nullptr) return 0;
  const int d = st_->getMetric()->getSignature()->getDimensions();
  return (d >= 0) ? static_cast<std::size_t>(d) + 1 : 0;
}

std::vector<std::vector<std::uint64_t>> SurgicalCone::topCells() const {
  std::vector<std::vector<std::uint64_t>> cells;
  const std::size_t tv = topVerts();
  if (tv == 0) return cells;
  for (const auto &s : st_->getSimplices()) {
    if (s == nullptr || s->size() != tv) continue;
    std::vector<std::uint64_t> ids;
    ids.reserve(tv);
    for (const auto &v : s->getVertices())
      if (v != nullptr) ids.push_back(v->getId());
    if (ids.size() != tv) continue;
    std::sort(ids.begin(), ids.end());
    cells.push_back(std::move(ids));
  }
  std::sort(cells.begin(), cells.end());
  cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
  return cells;
}

std::pair<bool, std::string> SurgicalCone::validate() const {
  const auto tops = topCells();
  if (tops.empty()) return {false, "no top cells"};
  const int dim = static_cast<int>(tops.front().size()) - 1;
  return ChainComplex::dualComplexIsValid(tops, dim);
}

std::vector<int> SurgicalCone::bettiNumbers() const {
  if (st_ == nullptr) return {};
  return ChainComplex::fromSpacetime(*st_).bettiNumbers();
}

std::size_t SurgicalCone::depth() const { return moves_.size(); }

bool SurgicalCone::isApplied() const { return !moves_.empty(); }

namespace {

/// Build {id -> Vertex*} over the live vertex list.
std::unordered_map<std::uint64_t, ::tessera::mesh::Vertex *> vertexIndex(
    Spacetime *st) {
  std::unordered_map<std::uint64_t, ::tessera::mesh::Vertex *> idx;
  for (const auto v : st->getVertexList()->toVector())
    if (v != nullptr) idx.emplace(v->getId(), v);
  return idx;
}

/// Build {min(u,v),max(u,v) -> Edge*} over the live edge list.
std::map<std::pair<std::uint64_t, std::uint64_t>, ::tessera::mesh::Edge *>
edgeIndex(Spacetime *st) {
  std::map<std::pair<std::uint64_t, std::uint64_t>, ::tessera::mesh::Edge *> idx;
  for (const auto e : st->getEdgeList()->toVector()) {
    if (e == nullptr || e->getSource() == nullptr || e->getTarget() == nullptr)
      continue;
    const std::uint64_t a = e->getSource()->getId();
    const std::uint64_t b = e->getTarget()->getId();
    idx[{std::min(a, b), std::max(a, b)}] = e;
  }
  return idx;
}

/// Coordinates of a vertex, or an empty vector for a coordinate-free one.
std::vector<double> coordsOf(::tessera::mesh::Vertex *v) {
  try {
    return v->getCoordinates();
  } catch (const std::exception &) {
    return {};
  }
}

}  // namespace

std::pair<bool, std::string> SurgicalCone::coneOut(
    const std::vector<std::uint64_t> &cell) {
  if (st_ == nullptr) return {false, "no spacetime"};
  const std::size_t tv = topVerts();
  if (tv < 2) return {false, "degenerate dimension"};
  if (cell.size() != tv)
    return {false, "cell is not a top cell (" + std::to_string(cell.size()) +
                       " vertices, expected " + std::to_string(tv) + ")"};
  std::vector<std::uint64_t> want(cell.begin(), cell.end());
  std::sort(want.begin(), want.end());

  // Locate the target top cell; collect the OTHER top cells (to know which of
  // want's edges survive). Refuse to remove the last top cell of the dimension.
  ::tessera::mesh::Simplex *target = nullptr;
  std::vector<std::vector<std::uint64_t>> otherTop;
  for (const auto s : st_->getSimplices()) {
    if (s == nullptr || s->size() != tv) continue;
    std::vector<std::uint64_t> ids;
    ids.reserve(tv);
    for (const auto v : s->getVertices())
      if (v != nullptr) ids.push_back(v->getId());
    std::sort(ids.begin(), ids.end());
    if (target == nullptr && ids == want)
      target = s;
    else
      otherTop.push_back(std::move(ids));
  }
  if (target == nullptr) return {false, "no such top cell"};
  if (otherTop.empty()) return {false, "refusing to remove the last top cell"};

  // An edge {u,v} of the cell is orphaned iff no surviving top cell covers both
  // endpoints. Capture (u, v, complex l2, phase) for each orphan so rollback
  // restores it bit-exactly (#581: the full complex value, never Re alone).
  const auto covered = [&](std::uint64_t u, std::uint64_t v) {
    for (const auto &c : otherTop) {
      const bool hu = std::find(c.begin(), c.end(), u) != c.end();
      const bool hv = std::find(c.begin(), c.end(), v) != c.end();
      if (hu && hv) return true;
    }
    return false;
  };
  auto eidx = edgeIndex(st_);
  Move m;
  m.kind = Move::Kind::ConeOut;
  m.cell = want;
  m.hadFacets = target->hasFacets();
  std::vector<::tessera::mesh::Edge *> toRemove;
  for (std::size_t i = 0; i + 1 < want.size(); ++i)
    for (std::size_t j = i + 1; j < want.size(); ++j) {
      const std::uint64_t u = want[i], v = want[j];
      if (covered(u, v)) continue;
      const auto it = eidx.find({std::min(u, v), std::max(u, v)});
      if (it == eidx.end()) continue;  // already absent
      m.edges.emplace_back(u, v, it->second->getLength(),
                           it->second->getPhase());
      toRemove.push_back(it->second);
    }

  // Snapshot the cell vertices' coords (for any that go isolated).
  auto vidx = vertexIndex(st_);
  std::unordered_map<std::uint64_t, std::vector<double>> coordSnap;
  for (const std::uint64_t id : want) {
    const auto it = vidx.find(id);
    if (it != vidx.end()) coordSnap[id] = coordsOf(it->second);
  }

  // Mutate: drop the top cell, then its orphaned faces, then its orphaned
  // edges. The face prune must precede the edge removal (removeEdge's
  // contract: no simplex may still contain the edge) — a registered face
  // stripped of its edge would stay wired into the hinges' coface walk while
  // reading l2 = 0 in every Gram-matrix computation, the #587 drift.
  st_->removeSimplex(target);
  // If anything was pruned, the undo must re-materialize even when the cell
  // itself carried no facet cache (a partially materialized host can hold
  // faces registered by a neighbor cell's getFacets) — registered faces are
  // never lost across a round trip.
  if (st_->pruneOrphanedSimplices(want) > 0) m.hadFacets = true;
  for (auto *e : toRemove)
    if (e != nullptr) st_->removeEdge(e);

  // Any cell vertex now carrying no edge is deleted (and recorded for rollback).
  vidx = vertexIndex(st_);
  for (const std::uint64_t id : want) {
    const auto it = vidx.find(id);
    if (it == vidx.end()) continue;
    if (it->second->degree() == 0) {
      m.verts.emplace_back(id, coordSnap[id]);
      st_->removeIfIsolated(it->second);
    }
  }

  // Gate: the result must be a valid manifold-with-boundary. On rejection,
  // restore the cell exactly (the inverse of this same move) and report.
  const auto verdict = validate();
  if (!verdict.first) {
    undoConeOut(m);
    return verdict;
  }
  moves_.push_back(std::move(m));
  return {true, "ok"};
}

std::pair<bool, std::string> SurgicalCone::coneIn(
    const std::vector<std::uint64_t> &targetVerts, bool timelike) {
  if (st_ == nullptr) return {false, "no spacetime"};
  const std::size_t tv = topVerts();
  if (tv < 2) return {false, "degenerate dimension"};
  if (targetVerts.size() != tv - 1)
    return {false, "cone-in needs " + std::to_string(tv - 1) +
                       " target vertices (got " +
                       std::to_string(targetVerts.size()) + ")"};
  // The targets must be distinct existing vertices.
  auto vidx = vertexIndex(st_);
  std::unordered_set<std::uint64_t> seen;
  ::tessera::mesh::VertexPtrs targets;
  targets.reserve(tv);
  for (const std::uint64_t id : targetVerts) {
    if (!seen.insert(id).second) return {false, "duplicate target vertex"};
    const auto it = vidx.find(id);
    if (it == vidx.end()) return {false, "no such target vertex"};
    targets.push_back(it->second);
  }

  // The fresh apex + the d targets form the new top cell.
  ::tessera::mesh::Vertex *apex = st_->createVertex();
  Move m;
  m.kind = Move::Kind::ConeIn;
  m.verts.emplace_back(apex->getId(), coordsOf(apex));
  ::tessera::mesh::VertexPtrs verts = targets;
  verts.push_back(apex);
  std::vector<std::uint64_t> cellIds;
  cellIds.reserve(tv);
  for (const auto v : verts) cellIds.push_back(v->getId());
  std::sort(cellIds.begin(), cellIds.end());
  m.cell = cellIds;

  auto r = st_->createSimplexTracked(verts);
  // #613: seed the apex edges' causal disposition BEFORE they are recorded below,
  // so the rollback record and the complex never disagree. Only edges incident to
  // the fresh apex are written; every pre-existing edge is left exactly as it was.
  // `timelike == false` (the default) writes nothing at all.
  if (timelike) {
    const std::uint64_t apexId = apex->getId();
    for (const auto &e : r.newEdges)
      if (e != nullptr && e->getSource() != nullptr &&
          e->getTarget() != nullptr &&
          (e->getSource()->getId() == apexId ||
           e->getTarget()->getId() == apexId))
        // Under balanced wiring the timelike class takes the OTHER root, so
        // l^2 = -i|m| rather than +i|m| (#741). Passing -kTimelikeSquaredLength
        // to the unbranched form landed on +1 — kTimelikeSquaredLength is
        // already negative — which is exactly the spacelike auto-wiring value,
        // so a timelike cone-in produced edges identical to a spacelike one.
        e->setLength(st_->balancedEdgeWiring()
                         ? ::tessera::spacetime::Spacetime::balancedLength(
                               kTimelikeSquaredLength, /*timelikeBranch=*/true)
                         : std::sqrt(std::complex<double>(
                               kTimelikeSquaredLength, 0.0)));
  }
  for (const auto &e : r.newEdges)
    if (e != nullptr && e->getSource() != nullptr && e->getTarget() != nullptr)
      m.edges.emplace_back(e->getSource()->getId(), e->getTarget()->getId(),
                           (e->getLength() * e->getLength()), e->getPhase());

  const auto verdict = validate();
  if (!verdict.first) {
    undoConeIn(m);
    return verdict;
  }
  moves_.push_back(std::move(m));
  return {true, "ok"};
}

namespace {

/// The sorted vertex ids selected by a bit mask over `cell`.
std::vector<std::uint64_t> maskedSubset(const std::vector<std::uint64_t> &cell,
                                        std::uint64_t mask) {
  std::vector<std::uint64_t> ids;
  for (std::size_t i = 0; i < cell.size(); ++i)
    if (mask & (std::uint64_t{1} << i)) ids.push_back(cell[i]);
  return ids;  // `cell` is sorted, so the subset is too
}

}  // namespace

std::pair<bool, std::string> SurgicalCone::bridge(
    const std::vector<std::uint64_t> &cellVertices) {
  if (st_ == nullptr) return {false, "no spacetime"};
  const std::size_t tv = topVerts();
  if (tv < 2) return {false, "degenerate dimension"};
  if (cellVertices.size() != tv)
    return {false, "bridge needs " + std::to_string(tv) + " vertices (got " +
                       std::to_string(cellVertices.size()) + ")"};
  // The vertices must be distinct and already exist: a bridge mints nothing.
  auto vidx = vertexIndex(st_);
  std::unordered_set<std::uint64_t> seen;
  ::tessera::mesh::VertexPtrs verts;
  verts.reserve(tv);
  for (const std::uint64_t id : cellVertices) {
    if (!seen.insert(id).second) return {false, "duplicate bridge vertex"};
    const auto it = vidx.find(id);
    if (it == vidx.end()) return {false, "no such bridge vertex"};
    verts.push_back(it->second);
  }
  if (auto existing = st_->findSimplexByVerts(verts);
      existing != nullptr && !existing->isStale())
    return {false, "cell already exists"};

  Move m;
  m.kind = Move::Kind::Bridge;
  m.cell = cellVertices;
  std::sort(m.cell.begin(), m.cell.end());
  // Which proper sub-faces of the cell are registered BEFORE the move (the
  // surfaces' own triangles among them, plus whatever a neighbouring cell's
  // lattice already materialized). Everything else the move's lifetime
  // registers under this cell is the move's own and goes on undo.
  {
    std::unordered_map<std::uint64_t, ::tessera::mesh::Vertex *> byId;
    for (const auto v : verts) byId.emplace(v->getId(), v);
    const std::uint64_t full = (std::uint64_t{1} << tv) - 1;
    for (std::uint64_t mask = 1; mask < full; ++mask) {
      const auto ids = maskedSubset(m.cell, mask);
      ::tessera::mesh::VertexPtrs sub;
      for (const std::uint64_t id : ids) sub.push_back(byId.at(id));
      const auto face = st_->findSimplexByVerts(sub);
      if (face != nullptr && !face->isStale() && face->size() == ids.size())
        m.preexistingFaces.push_back(ids);
    }
  }

  auto r = st_->createSimplexTracked(verts);
  if (!r.created) return {false, "cell already exists"};
  for (const auto &e : r.newEdges)
    if (e != nullptr && e->getSource() != nullptr && e->getTarget() != nullptr)
      m.edges.emplace_back(e->getSource()->getId(), e->getTarget()->getId(),
                           e->getLength(), e->getPhase());

  // Gate: the manifold check over the top cells that exist, and nothing else.
  const auto verdict = validate();
  if (!verdict.first) {
    undoBridge(m);
    return verdict;
  }
  moves_.push_back(std::move(m));
  return {true, "ok"};
}

void SurgicalCone::undoBridge(const Move &m) {
  // Drop the top cell first: removeSimplex clears its facets' coface links,
  // so a sub-face the move introduced is then recognisable by having no
  // coface left at all.
  auto vidx = vertexIndex(st_);
  ::tessera::mesh::VertexPtrs verts;
  verts.reserve(m.cell.size());
  for (const std::uint64_t id : m.cell) {
    const auto it = vidx.find(id);
    if (it != vidx.end()) verts.push_back(it->second);
  }
  if (verts.size() == m.cell.size()) {
    if (auto s = st_->findSimplexByVerts(verts); s != nullptr && !s->isStale())
      st_->removeSimplex(s);
  }
  // Sub-faces the move introduced: registered now, not registered before the
  // move, and held by no surviving simplex. Largest first, so a face's own
  // facets still see it when its coface links are cleaned. A pre-existing
  // face — a surface triangle no top cell covers, or a neighbour's
  // materialized facet — is never touched, which is why this is not
  // Spacetime::pruneOrphanedSimplices (that would delete the uncovered
  // surface triangles the bulk is being drawn onto).
  const std::set<std::vector<std::uint64_t>> preexisting(
      m.preexistingFaces.begin(), m.preexistingFaces.end());
  const std::size_t n = m.cell.size();
  std::unordered_map<std::uint64_t, ::tessera::mesh::Vertex *> byId;
  for (const auto v : verts) byId.emplace(v->getId(), v);
  for (std::size_t faceSize = n - 1; faceSize >= 1; --faceSize) {
    for (std::uint64_t mask = 1; mask < (std::uint64_t{1} << n); ++mask) {
      if (static_cast<std::size_t>(std::popcount(mask)) != faceSize) continue;
      const auto ids = maskedSubset(m.cell, mask);
      if (preexisting.count(ids)) continue;
      ::tessera::mesh::VertexPtrs sub;
      sub.reserve(ids.size());
      for (const std::uint64_t id : ids) {
        const auto it = byId.find(id);
        if (it == byId.end()) break;
        sub.push_back(it->second);
      }
      if (sub.size() != ids.size()) continue;
      const auto face = st_->findSimplexByVerts(sub);
      if (face == nullptr || face->isStale() || face->size() != faceSize) continue;
      if (!face->getCofaces().empty()) continue;  // held by a survivor
      st_->removeSimplex(face);
    }
  }
  // The edges this move alone inserted. Every simplex that contained one of
  // them was introduced by the move too and is gone by now, so removeEdge's
  // contract (no simplex still holds the edge) is met.
  auto eidx = edgeIndex(st_);
  for (const auto &[u, v, w, theta] : m.edges) {
    (void)w;
    (void)theta;
    const auto it = eidx.find({std::min(u, v), std::max(u, v)});
    if (it != eidx.end()) st_->removeEdge(it->second);
  }
}

void SurgicalCone::undoConeOut(const Move &m) {
  // Re-create any isolated vertices first, then the top cell, then restore the
  // removed edges' lengths/phases bit-exactly.
  for (const auto &[id, coords] : m.verts) {
    if (coords.empty())
      (void)st_->createVertex(id);
    else
      (void)st_->createVertex(id, coords);
  }
  auto vidx = vertexIndex(st_);
  ::tessera::mesh::VertexPtrs verts;
  verts.reserve(m.cell.size());
  for (const std::uint64_t id : m.cell) {
    const auto it = vidx.find(id);
    if (it == vidx.end()) return;  // a cell vertex vanished — cannot restore
    verts.push_back(it->second);
  }
  const auto restored = st_->createSimplexTracked(verts);

  auto eidx = edgeIndex(st_);
  for (const auto &[u, v, w, theta] : m.edges) {
    const auto it = eidx.find({std::min(u, v), std::max(u, v)});
    if (it != eidx.end()) {
      it->second->setLength(w);  // the recorded complex LENGTH, bit-exact
      it->second->setPhase(theta);
    }
  }

  // Restore the cell's facet/coface lattice: createSimplexTracked wires the
  // cell to vertices and edges only, so without this the restored cell is
  // nobody's coface and every surrounding hinge's dualVolume misses its
  // wedges until some global re-materialization — and the pruned faces
  // (fresh objects bound to the fresh edges) would never come back at all.
  // Skipped when the pre-move cell had no materialized facets, so the undo
  // never creates bookkeeping the pre-move complex lacked.
  if (m.hadFacets) st_->materializeFacets(restored.simplex);
}

void SurgicalCone::undoConeIn(const Move &m) {
  // Drop the added top cell, its orphaned faces, its fresh edges, then the
  // fresh apex vertex. The face prune covers the case where the lattice was
  // materialized between the accepted move and this rollback (e.g. a solver
  // scoring the probe): the apex's faces would otherwise stay registered
  // with their edges stripped — the same zombie class as the cone-out side.
  // Faces shared with surviving top cells are kept.
  auto vidx = vertexIndex(st_);
  ::tessera::mesh::VertexPtrs verts;
  verts.reserve(m.cell.size());
  for (const std::uint64_t id : m.cell) {
    const auto it = vidx.find(id);
    if (it != vidx.end()) verts.push_back(it->second);
  }
  if (verts.size() == m.cell.size()) {
    if (auto s = st_->findSimplexByVerts(verts)) st_->removeSimplex(s);
  }
  st_->pruneOrphanedSimplices(m.cell);
  auto eidx = edgeIndex(st_);
  for (const auto &[u, v, w, theta] : m.edges) {
    (void)w;
    (void)theta;
    const auto it = eidx.find({std::min(u, v), std::max(u, v)});
    if (it != eidx.end()) st_->removeEdge(it->second);
  }
  for (const auto &[id, coords] : m.verts) {
    (void)coords;
    const auto it = vidx.find(id);
    if (it != vidx.end()) st_->removeIfIsolated(it->second);
  }
}

bool SurgicalCone::rollback() {
  if (moves_.empty()) return false;
  const Move m = std::move(moves_.back());
  moves_.pop_back();
  if (m.kind == Move::Kind::ConeOut)
    undoConeOut(m);
  else if (m.kind == Move::Kind::ConeIn)
    undoConeIn(m);
  else
    undoBridge(m);
  return true;
}

std::size_t SurgicalCone::rollbackAll() {
  std::size_t n = 0;
  while (rollback()) ++n;
  return n;
}

}  // namespace tessera::cobordism
