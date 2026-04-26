// MIT License
// Copyright (c) 2025 Andrew Kelleher
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//
// Created by andrew on 10/23/25.
//

// (was: #include <pybind11/pybind11.h> — removed; unreferenced.)
#include "Logger.h"
#include <cmath>
#include <memory>
#include <queue>
#include <set>
#include "spacetime/Spacetime.h"
#include "mesh/SimplexOrientation.h"
#include "mesh/ForwardDeclarations.h"
#include "mesh/EdgeList.h"
#include "mesh/Edge.h"
#include "spacetime/topologies/Toroid.h"

namespace tessera {

// ========================================
// Constructors
// ========================================

Spacetime::Spacetime() {
  Signature signature(4, SignatureType::Lorentzian);
  metric = std::make_shared<Metric>(true, signature);
  spacetimeType = SpacetimeType::CDT;
  alpha = 1.;
  topology = std::make_shared<Toroid>();
  a = 1.;
  foliation = Foliation::PREFERRED;
}

Spacetime::Spacetime(
  std::shared_ptr<Metric> metric_,
  const SpacetimeType spacetimeType_,
  std::optional<double> alpha_,
  std::optional<double> a_,
  Foliation foliation_,
  std::optional<std::shared_ptr<Topology> > topology_) : metric(std::move(metric_)), spacetimeType(spacetimeType_), foliation(foliation_) {
  alpha = alpha_.value_or(1.);
  a = a_.value_or(1.);
  topology = topology_.has_value() ? std::move(*topology_) : std::make_shared<Toroid>();
}

// ========================================
// Creation Methods
// ========================================

std::pair<SimplexPtr, bool> Spacetime::createSimplex(
  const VertexPtrs &vertices,
  const Edges &edges
) {
  // Compute hash directly without allocating a temporary Fingerprint.
  std::uint64_t hash = 0;
  for (const auto &v : vertices) {
    hash ^= Fingerprint::mix64(v->getId());
  }
  auto *found = simplexIndex_.find(hash);
  if (found) {
#ifdef TESSERA_ASSERTIONS
    CLOG(DEBUG_LEVEL, "You attempted to create a simplex that already exists: ", (*found)->toString());
#endif
    return {*found, false};
  }
  SimplexPtr simplex = Simplex::create(this, vertices, edges);
  registerSimplex(simplex, false);
  return {simplex, true};
}

std::pair<SimplexPtr, bool> Spacetime::createSimplex(
  const VertexPtrs &vertices
) {
  std::vector<EdgePtr> edges_{};
  bool isLorentzian = metric->getSignature()->getSignatureType() == SignatureType::Lorentzian;
  for (std::size_t i=0; i<vertices.size()-1; i++) {
    for (std::size_t j=i+1; j<vertices.size(); j++) {
      double squaredLen = a;  // spacelike: ℓ² = a
      if (isLorentzian && vertices[i]->getTime() != vertices[j]->getTime()) {
        squaredLen = -alpha * a;  // timelike: ℓ² = -α·a
      }
      EdgePtr edge = createEdge(vertices[i], vertices[j], squaredLen);
      edges_.push_back(edge);
    }
  }
  return createSimplex(vertices, edges_);
}

std::pair<SimplexPtr, bool> Spacetime::createSimplex(const std::tuple<uint8_t, uint8_t> &numericOrientation) {
  double spacelikeSquaredLength = a;           // ℓ² = a
  double timelikeSquaredLength = -alpha * a;    // ℓ² = -α·a
  if (getMetric()->getSignature()->getSignatureType() != SignatureType::Lorentzian) {
    timelikeSquaredLength = a;  // Euclidean: all edges positive
  }
  SimplexOrientation orientation = {
    std::get<0>(numericOrientation),
    std::get<1>(numericOrientation)
  };
  std::uint8_t k = orientation.getK();
  auto [ti, tf] = orientation.numeric();
  VertexPtrs vertices = {};
  vertices.reserve(k);
  Edges edges = {};
  edges.reserve(Simplex::computeNumberOfEdges(k));
  for (int i = 0; i < ti; i++) {
    // Create ti vertices at currentTime (initial time slice).
    VertexPtr newVertex = vertexList->add(vertexIdCounter++, {static_cast<double>(currentTime)});
    for (const auto &existingVertex : vertices) {
      EdgePtr edge = edgeList->
          add(existingVertex, newVertex, spacelikeSquaredLength);
      existingVertex->addOutEdge(edge);
      newVertex->addInEdge(edge);
      edges.push_back(edge);
    }
    vertices.push_back(newVertex);
  }
  for (int i = 0; i < tf; i++) {
    // Create ti Spacelike vertices
    // Use coning to construct the vertex edges. For each new vertex; draw an edge to each existing vertex.
    /// We can't just use the vertexList .size() here, because some vertices can be removed. We need to keep a
    /// counter:
    VertexPtr newVertex = vertexList->add(vertexIdCounter++, {static_cast<double>(currentTime + 1)});
    for (const auto &existingVertex : vertices) {
      EdgePtr edge;
      if (existingVertex->getTime() < newVertex->getTime()) {
        edge = edgeList->add(existingVertex, newVertex, timelikeSquaredLength);
      } else {
        edge = edgeList->add(existingVertex, newVertex, spacelikeSquaredLength);
      }
      existingVertex->addOutEdge(edge);
      newVertex->addInEdge(edge);
      edges.push_back(edge);
    }
    vertices.push_back(newVertex);
  }
  return createSimplex(vertices, edges);
}

double Spacetime::getAlpha() const noexcept {
  return alpha;
}

double Spacetime::getA() const noexcept {
  return a;
}

std::pair<SimplexPtr, bool> Spacetime::createSimplex(std::size_t k) {
  double squaredLength = a;  // all same-time → spacelike: ℓ² = a
  VertexPtrs vertices = {};
  vertices.reserve(k);
  Edges edges = {};
  edges.reserve(Simplex::computeNumberOfEdges(k));
  for (int i = 0; i < k; i++) {
    // Use coning to construct the vertex edges. For each new vertex; draw an edge to each existing vertex.
    VertexPtr newVertex = vertexList->add(vertexIdCounter++, {static_cast<double>(currentTime)});
    for (const auto &existingVertex : vertices) {
      EdgePtr edge = edgeList->add(existingVertex, newVertex, squaredLength);
      existingVertex->addOutEdge(edge);
      newVertex->addInEdge(edge);
      edges.push_back(edge);
    }
    vertices.push_back(newVertex);
  }
  return createSimplex(vertices, edges);
}

VertexPtr Spacetime::createVertex() noexcept {
  return vertexList->add(vertexIdCounter++);
}

VertexPtr Spacetime::createVertex(const std::uint64_t id) const noexcept {
  return vertexList->add(id);
}

VertexPtr Spacetime::createVertex(const std::uint64_t id, const std::vector<double> &coords) const noexcept {
  return vertexList->add(id, coords);
}

VertexPtr Spacetime::createVertex(const std::vector<double> &coords) noexcept {
  return vertexList->add(vertexIdCounter++, coords);
}

EdgePtr Spacetime::createEdge(
  const VertexPtr &src,
  const VertexPtr &tgt
) const noexcept {
  EdgePtr edge = edgeList->add(src, tgt);
  src->addOutEdge(edge);
  tgt->addInEdge(edge);
  return edge;
}

EdgePtr Spacetime::createEdge(
  const VertexPtr &src,
  const VertexPtr &tgt,
  double squaredLength
) const noexcept {
#ifdef TESSERA_ASSERTIONS
  if (src->getTime() == tgt->getTime() && squaredLength <= 0) {
    CLOG(INFO_LEVEL, "You attempted to create a same-time (spacelike) edge with non-positive squared length");
    std::abort();
  }
#endif
  EdgePtr edge = edgeList->add(src, tgt, squaredLength);
  src->addOutEdge(edge);
  tgt->addInEdge(edge);
  return edge;
}

// ========================================
// Complex Building Methods
// ========================================
void Spacetime::build(int numSimplices) {
  return topology->build(this, numSimplices);
}

// ========================================
// Query Methods
// ========================================
SpacetimeType Spacetime::getSpacetimeType() const noexcept {
  return spacetimeType;
}

double Spacetime::getCurrentTime() const noexcept {
  return static_cast<double>(currentTime);
}

const std::shared_ptr<EdgeList> &Spacetime::getEdgeList() const noexcept {
  return edgeList;
}

Foliation Spacetime::getFoliation() const noexcept {
  return foliation;
}

const std::shared_ptr<Metric> &Spacetime::getMetric() const noexcept {
  return metric;
}

// ========================================
// Time-Slice & Spatial-Subgraph Queries
// ========================================

std::vector<int> Spacetime::getTimeSlices() const {
  std::set<int> times;
  for (auto *v : vertexList->liveVector())
    times.insert(static_cast<int>(v->getTime()));
  return {times.begin(), times.end()};
}

VertexPtrs Spacetime::getVerticesAtTime(int t) const {
  VertexPtrs result;
  for (auto *v : vertexList->liveVector())
    if (static_cast<int>(v->getTime()) == t)
      result.push_back(v);
  return result;
}

std::pair<VertexPtrs, Edges> Spacetime::getSpatialSubgraph(int t) const {
  auto verts = getVerticesAtTime(t);
  std::unordered_set<std::uint64_t> vidSet;
  for (auto *v : verts) vidSet.insert(v->getId());

  Edges spatial;
  std::unordered_set<std::uint64_t> seen;
  for (auto *v : verts) {
    for (const auto &e : v->getEdges()) {
      auto fp = e->fingerprint.fingerprint();
      if (seen.count(fp)) continue;
      seen.insert(fp);
      if (vidSet.count(e->getSource()->getId())
          && vidSet.count(e->getTarget()->getId())
          && e->getSquaredLength() > 0)
        spatial.push_back(e);
    }
  }
  return {verts, spatial};
}

std::unordered_map<std::uint64_t, int>
Spacetime::bfsDistances(VertexPtr center, int maxDepth) const {
  // Build adjacency from spacelike edges at center's time slice
  int t = static_cast<int>(center->getTime());
  auto [verts, edges] = getSpatialSubgraph(t);

  std::unordered_map<std::uint64_t, std::vector<std::uint64_t>> adj;
  for (auto *v : verts) adj[v->getId()]; // ensure entry
  for (auto *e : edges) {
    auto s = e->getSource()->getId();
    auto tgt = e->getTarget()->getId();
    adj[s].push_back(tgt);
    adj[tgt].push_back(s);
  }

  std::unordered_map<std::uint64_t, int> dist;
  dist[center->getId()] = 0;
  std::queue<std::uint64_t> q;
  q.push(center->getId());
  while (!q.empty()) {
    auto vid = q.front(); q.pop();
    if (maxDepth >= 0 && dist[vid] >= maxDepth) continue;
    for (auto nbr : adj[vid]) {
      if (!dist.count(nbr)) {
        dist[nbr] = dist[vid] + 1;
        q.push(nbr);
      }
    }
  }
  return dist;
}

const std::shared_ptr<VertexList> &Spacetime::getVertexList() const noexcept {
  return vertexList;
}

SimplexSet Spacetime::getExternalSimplices() noexcept {
  SimplexSet result{};
  for (const auto &simplex : simplicesVec) {
    if (simplex->hasBoundaryFacet()) result.insert(simplex);
  }
  return result;
}

std::vector<SimplexPtr> Spacetime::getSimplicesWithOrientation(std::tuple<uint8_t, uint8_t> orientation) const {
  SimplexOrientation o{std::get<0>(orientation), std::get<1>(orientation)};
  std::vector<SimplexPtr> result{};
  for (const auto &simplex : simplicesVec) {
    if (simplex->getOrientation() == o) result.push_back(simplex);
  }
  return result;
}

std::tuple<std::vector<std::uint32_t>,
           std::vector<std::uint32_t>,
           std::uint32_t>
Spacetime::getDualAdjacency() const {
  const std::uint32_t N = static_cast<std::uint32_t>(topSimplicesVec.size());

  // Map fingerprint → index in topSimplicesVec
  // (topSimplexVecIndex already exists but maps to pool slots; rebuild a clean one)
  std::unordered_map<std::uint64_t, std::uint32_t> fpToIdx;
  fpToIdx.reserve(N);
  for (std::uint32_t i = 0; i < N; ++i) {
    fpToIdx[topSimplicesVec[i]->fingerprint.fingerprint()] = i;
  }

  std::vector<std::uint32_t> rows, cols;
  rows.reserve(N * 5);  // ~d+1 neighbours per simplex in d dimensions
  cols.reserve(N * 5);

  for (std::uint32_t i = 0; i < N; ++i) {
    const auto &simplex = topSimplicesVec[i];
    const auto &facets = simplex->getFacets();
    for (const auto &facet : facets) {
      const auto &cofaces = facet->getCofaces();
      for (const auto &coface : cofaces) {
        if (coface == simplex) continue;
        const auto fp = coface->fingerprint.fingerprint();
        auto it = fpToIdx.find(fp);
        if (it != fpToIdx.end()) {
          rows.push_back(it->second);
          cols.push_back(i);
        }
      }
    }
  }

  return {std::move(rows), std::move(cols), N};
}

std::vector<VertexPtrs> Spacetime::getConnectedComponents() const {
  VertexPtrSet seen{};
  std::vector<VertexPtrs> components{};
  for (auto vertex : vertexList->toVector()) {
    if (seen.contains(vertex)) {
      continue;
    }
    VertexPtrs component{};
    VertexPtrs stack{vertex};
    while (!stack.empty()) {
      VertexPtr current = stack.back();
      stack.pop_back();
      if (seen.contains(current)) {
        continue;
      }
      seen.insert(current);
      component.push_back(current);
      for (const auto &edge : current->getOutEdges()) {
        VertexPtr neighbor = vertexList->get(edge->getTarget()->getId());
        if (neighbor != nullptr && !seen.contains(neighbor)) {
          stack.push_back(neighbor);
        }
      }
      for (const auto &edge : current->getInEdges()) {
        VertexPtr neighbor = vertexList->get(edge->getSource()->getId());
        if (neighbor != nullptr && !seen.contains(neighbor)) {
          stack.push_back(neighbor);
        }
      }
    }
    components.push_back(component);
  }
  return components;
}

SimplexPtr Spacetime::getSimplex(SimplexPtr simplex) const {
  if (simplex && simplex->vecIdx_ != UINT32_MAX) return simplex;
  return nullptr;
}

SimplexPtr Spacetime::getSimplex(std::uint64_t fingerprint) const {
  auto *s = simplexIndex_.find(fingerprint);
  if (!s) return nullptr;
  return *s;
}

// ========================================
// Manipulation & Helper Methods
// ========================================

double Spacetime::incrementTime() noexcept {
  currentTime++;
  return static_cast<double>(currentTime);
}

void Spacetime::swapVertexLabels(VertexPtr v1, VertexPtr v2) {
  if (v1 == v2 || v1->getId() == v2->getId()) return;

  auto id1 = v1->getId();
  auto id2 = v2->getId();

  // Collect simplices containing EXACTLY ONE of v1, v2 and record which
  // vertex they belong to BEFORE swapping IDs (hasVertex compares by ID).
  // Simplices containing both are unaffected (XOR fingerprint is symmetric).
  struct AffectedSimplex { SimplexPtr ptr; bool hadV1; };
  std::vector<AffectedSimplex> affected;

  for (const auto &s : v1->getSimplices()) {
    if (!s->hasVertex(v2))
      affected.push_back({s, true});
  }
  for (const auto &s : v2->getSimplices()) {
    if (!s->hasVertex(v1))
      affected.push_back({s, false});
  }

  // Swap vertex IDs, rekey vertex list
  v1->setId(id2);
  v2->setId(id1);
  vertexList->swapKeys(id1, id2);

  // Update edge fingerprints and rekey in EdgeList.
  // Edges incident to exactly one of v1, v2 need fingerprint updates.
  // Edges between v1 and v2 are unaffected (XOR is commutative).
  //
  // Must batch: extract-all, update-all, reinsert-all to avoid transient
  // collisions when v1 and v2 share a neighbor.
  struct AffectedEdge { EdgePtr ptr; bool fromV1; };
  std::vector<AffectedEdge> affectedEdges;

  for (const auto &e : v1->getEdges()) {
    if (!e->hasVertex(id1))  // id1 is now v2's id; skip v1-v2 edge
      affectedEdges.push_back({e, true});
  }
  for (const auto &e : v2->getEdges()) {
    if (!e->hasVertex(id2))  // id2 is now v1's id; skip v1-v2 edge
      affectedEdges.push_back({e, false});
  }

  std::vector<std::pair<std::uint32_t, std::size_t>> edgeSlots;
  edgeSlots.reserve(affectedEdges.size());
  for (std::size_t i = 0; i < affectedEdges.size(); ++i) {
    auto slot = edgeList->detachEdge(affectedEdges[i].ptr->fingerprint.fingerprint());
    if (slot != UINT32_MAX)
      edgeSlots.push_back({slot, i});
  }

  for (auto &[slot, idx] : edgeSlots) {
    auto &[e, fromV1] = affectedEdges[idx];
    if (fromV1) {
      e->fingerprint.removeId(id1);
      e->fingerprint.addId(id2);
    } else {
      e->fingerprint.removeId(id2);
      e->fingerprint.addId(id1);
    }
    e->fingerprint.refresh();
  }

  for (auto &[slot, idx] : edgeSlots) {
    edgeList->reattachEdge(slot);
  }

  // Re-key simplexIndex_: erase old fingerprints, update, re-insert.
  // Only re-key simplices that are registered (vecIdx_ != UINT32_MAX).
  for (auto &[s, hadV1] : affected) {
    if (s->vecIdx_ != UINT32_MAX)
      simplexIndex_.erase(s->fingerprint.fingerprint());
  }
  for (auto &[s, hadV1] : affected) {
    if (hadV1) {
      s->fingerprint.removeId(id1);
      s->fingerprint.addId(id2);
    } else {
      s->fingerprint.removeId(id2);
      s->fingerprint.addId(id1);
    }
    s->fingerprint.refresh();
    if (s->vecIdx_ != UINT32_MAX)
      simplexIndex_.insert(s->fingerprint.fingerprint(), s);
  }
}

bool Spacetime::removeIfIsolated(const VertexPtr &vertex) noexcept {
  if (vertex->degree() == 0) {
    CLOG(DEBUG_LEVEL, "Removing vertex: ", vertex->toString());
    vertexList->remove(vertex);
    return true;
  }
  CLOG(DEBUG_LEVEL, "NOT Removing vertex: ", vertex->toString());
  return false;
}

void Spacetime::addObservable(const std::shared_ptr<Observable> &observable) {
  observables.push_back(observable);
}

// ========================================
// Internal Management
// ========================================

SimplexPtr Spacetime::registerSimplex(const SimplexPtr &simplex, bool internal) {
  auto fp = simplex->fingerprint.fingerprint();
  auto *existing = simplexIndex_.find(fp);
  if (existing) {
    return *existing;
  }

  std::uint32_t slot;
  if (!simplexFreeSlots_.empty()) {
    slot = simplexFreeSlots_.back();
    simplexFreeSlots_.pop_back();
  } else {
    slot = static_cast<std::uint32_t>(simplexPool_.size());
    simplexPool_.push_back(nullptr);
  }
  simplexPool_[slot] = simplex;
  simplex->poolSlot_ = slot;

  simplex->vecIdx_ = static_cast<std::uint32_t>(simplicesVec.size());
  simplicesVec.push_back(simplex);
  simplexIndex_.insert(fp, simplex);

  auto d = metric->getSignature()->getDimensions();
  if (simplex->size() == static_cast<std::size_t>(d + 1)) {
    simplex->topVecIdx_ = static_cast<std::uint32_t>(topSimplicesVec.size());
    topSimplicesVec.push_back(simplex);
  }
  updateOrientationCounters(simplex, +1);
  return simplex;
}

void Spacetime::unregisterSimplex(const SimplexPtr &simplex) {
  auto vecIdx = simplex->vecIdx_;
  if (vecIdx == UINT32_MAX) {
#ifdef TESSERA_ASSERTIONS
    CLOG(CRITICAL_LEVEL, "You attempted to unregister a simplex that does not exist!");
#endif
    return;
  }

  updateOrientationCounters(simplex, -1);

  auto fp = simplex->fingerprint.fingerprint();
  auto poolSlot = simplex->poolSlot_;

  // Swap-and-pop from simplicesVec (O(1) via stored index)
  if (vecIdx + 1 < static_cast<std::uint32_t>(simplicesVec.size())) {
    simplicesVec[vecIdx] = simplicesVec.back();
    simplicesVec[vecIdx]->vecIdx_ = vecIdx;
  }
  simplicesVec.pop_back();
  simplex->vecIdx_ = UINT32_MAX;
  simplexIndex_.erase(fp);

  // Swap-and-pop from topSimplicesVec (O(1) via stored index)
  auto topIdx = simplex->topVecIdx_;
  if (topIdx != UINT32_MAX) {
    if (topIdx + 1 < static_cast<std::uint32_t>(topSimplicesVec.size())) {
      topSimplicesVec[topIdx] = topSimplicesVec.back();
      topSimplicesVec[topIdx]->topVecIdx_ = topIdx;
    }
    topSimplicesVec.pop_back();
    simplex->topVecIdx_ = UINT32_MAX;
  }

  // Release pool slot and free the allocation
  delete simplexPool_[poolSlot];
  simplexPool_[poolSlot] = nullptr;
  simplex->poolSlot_ = UINT32_MAX;
  simplexFreeSlots_.push_back(poolSlot);
}

void Spacetime::reserve(int nSimplices) {
  simplexIndex_.reserve(nSimplices);
  simplicesVec.reserve(nSimplices);
  topSimplicesVec.reserve(nSimplices);
  edgeList->reserve(nSimplices);
  vertexList->reserve(nSimplices);
}

// ========================================
// Counting & Access
// ========================================

std::size_t Spacetime::getSimplexCount() const noexcept {
  return n41Count + n32Count;
}

std::size_t Spacetime::getVertexCount() const noexcept {
  return vertexList->size();
}

std::size_t Spacetime::getN41() const noexcept {
  return n41Count;
}

std::size_t Spacetime::getN32() const noexcept {
  return n32Count;
}

const std::vector<SimplexPtr>& Spacetime::getSimplices() const noexcept {
  return simplicesVec;
}

VertexPtr Spacetime::getRandomVertex() {
  const auto &verts = vertexList->liveVector();
  if (verts.empty()) return nullptr;
  std::uniform_int_distribution<std::size_t> dist(0, verts.size() - 1);
  return verts[dist(rng)];
}

SimplexPtr Spacetime::getRandomSimplex() {
  if (simplicesVec.empty()) return nullptr;
  std::uniform_int_distribution<std::size_t> dist(0, simplicesVec.size() - 1);
  return simplicesVec[dist(rng)];
}

SimplexPtr Spacetime::getRandomTopSimplex() {
  if (topSimplicesVec.empty()) return nullptr;
  std::uniform_int_distribution<std::size_t> dist(0, topSimplicesVec.size() - 1);
  return topSimplicesVec[dist(rng)];
}

SimplexPtr Spacetime::getRandomSimplexWithOrientation(uint8_t ti, uint8_t tf) {
  SimplexOrientation target{ti, tf};
  // Try random sampling first (fast if many match)
  for (int attempt = 0; attempt < 100; ++attempt) {
    auto s = getRandomSimplex();
    if (s && s->getOrientation() == target) return s;
  }
  // Fallback: linear scan and pick random from matches
  std::vector<SimplexPtr> matches;
  for (const auto &s : simplicesVec) {
    if (s->getOrientation() == target) matches.push_back(s);
  }
  if (matches.empty()) return nullptr;
  std::uniform_int_distribution<std::size_t> dist(0, matches.size() - 1);
  return matches[dist(rng)];
}

void Spacetime::updateOrientationCounters(const SimplexPtr &simplex, int delta) {
  auto d = metric->getSignature()->getDimensions();
  auto nVerts = simplex->size();
  // Only count top-dimensional simplices (d-simplices have d+1 vertices)
  if (nVerts != static_cast<std::size_t>(d + 1)) return;
  auto [ti, tf] = simplex->getOrientation().numeric();
  // (d, 1) or (1, d) type
  if ((ti == d && tf == 1) || (ti == 1 && tf == d)) {
    n41Count += delta;
  }
  // (d-1, 2) or (2, d-1) type
  else if ((ti == d - 1 && tf == 2) || (ti == 2 && tf == d - 1)) {
    n32Count += delta;
  }
}

void Spacetime::removeSimplex(const SimplexPtr &simplex) {
  // Remove from vertex simplex lists
  for (const auto &v : simplex->getVertices()) {
    v->removeSimplex(simplex);
  }
  // Remove coface references from facets
  if (simplex->hasFacets()) {
    for (const auto &facet : simplex->getFacets()) {
      facet->removeCoface(simplex);
    }
  }
  unregisterSimplex(simplex);
}

} // tessera
