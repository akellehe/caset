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

#include <pybind11/pybind11.h>
// #include <torch/torch.h>
#include "Logger.h"
#include <memory>
#include "spacetime/Spacetime.h"
#include "SimplexOrientation.h"
#include "ForwardDeclarations.h"
#include "EdgeList.h"
#include "Edge.h"
#include "spacetime/topologies/Toroid.h"

namespace caset {

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
  std::optional<std::shared_ptr<Topology> > topology_) : metric(metric_), spacetimeType(spacetimeType_), foliation(foliation_) {
  alpha = alpha_.value_or(1.);
  a = a_.value_or(1.);
  topology = topology_.value_or(std::make_shared<Toroid>());
}

// ========================================
// Creation Methods
// ========================================

std::pair<SimplexPtr, bool> Spacetime::createSimplex(
  const VertexPtrs &vertices,
  const Edges &edges
) {
  // Compute hash directly without allocating a temporary Fingerprint.
  // Uses heterogeneous lookup (is_transparent) on the SimplexSet.
  std::uint64_t hash = 0;
  for (const auto &v : vertices) {
    hash ^= Fingerprint::mix64(v->getId());
  }
  const auto found = simplices.find(hash);
  if (found == simplices.end()) {
#ifdef CASET_ASSERTIONS
    std::unordered_set<std::uint64_t> seen{};
    for (const auto &s : simplices) {
      if (seen.contains(s->fingerprint.fingerprint())) {
        CLOG(CRITICAL_LEVEL, "attempted to create a new simplex with the same fingerprint as an existing one!");
        CLOG(CRITICAL_LEVEL, "Attempted vertices: ");
        for (const auto &v : vertices) {
          CLOG(CRITICAL_LEVEL, "    - ", v->toString());
        }
        CLOG(CRITICAL_LEVEL,
             "Existing simplex: ",
             s->toString(),
             " with fingerprint ",
             std::to_string(s->fingerprint.fingerprint()),
             " vs ",
             std::to_string(hash));
        throw std::runtime_error("Duplicate simplex: " + s->toString());
      }
      seen.insert(s->fingerprint.fingerprint());
    }
#endif
    SimplexPtr simplex = Simplex::create(this, vertices, edges);
    registerSimplex(simplex, false);
    return {simplex, true};
  }
#ifdef CASET_ASSERTIONS
  CLOG(DEBUG_LEVEL, "You attempted to create a simplex that already exists: ", (*found)->toString());
#endif
  return {*found, false};
}

std::pair<SimplexPtr, bool> Spacetime::createSimplex(
  const VertexPtrs &vertices
) {
  std::vector<EdgePtr> edges_{};
  bool isLorentzian = metric->getSignature()->getSignatureType() == SignatureType::Lorentzian;
  for (std::size_t i=0; i<vertices.size()-1; i++) {
    for (std::size_t j=i+1; j<vertices.size(); j++) {
      double squaredLen = alpha;
      if (isLorentzian && vertices[i]->getTime() == vertices[j]->getTime()) {
        squaredLen = -alpha;
      }
      EdgePtr edge = createEdge(vertices[i], vertices[j], squaredLen);
      edges_.push_back(edge);
    }
  }
  return createSimplex(vertices, edges_);
}

std::pair<SimplexPtr, bool> Spacetime::createSimplex(const std::tuple<uint8_t, uint8_t> &numericOrientation) {
  double squaredLength = alpha;
  double timelikeSquaredLength = alpha;
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
    // Create ti Timelike vertices
    // Use coning to construct the vertex edges. For each new vertex; draw an edge to each existing vertex.
    VertexPtr newVertex = vertexList->add(vertexIdCounter++, {static_cast<double>(currentTime)});
    if (getMetric()->getSignature()->getSignatureType() == SignatureType::Lorentzian) {
      timelikeSquaredLength = -alpha;
    }
    for (const auto &existingVertex : vertices) {
      EdgePtr edge = edgeList->
          add(existingVertex, newVertex, timelikeSquaredLength);
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
        edge = edgeList->add(existingVertex, newVertex, squaredLength);
      } else {
        edge = edgeList->add(existingVertex, newVertex, timelikeSquaredLength);
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
  double squaredLength = alpha;
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
#ifdef CASET_ASSERTIONS
  if (src->getTime() == tgt->getTime() && squaredLength >= 0) {
    CLOG(INFO_LEVEL, "You attempted to create an edge for which the start and end vertices have the same time, but the squared length is greater than 0");
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

std::shared_ptr<EdgeList> Spacetime::getEdgeList() const noexcept {
  return edgeList;
}

Foliation Spacetime::getFoliation() const noexcept {
  return foliation;
}

std::shared_ptr<Metric> Spacetime::getMetric() const noexcept {
  return metric;
}

std::shared_ptr<VertexList> Spacetime::getVertexList() const noexcept {
  return vertexList;
}

SimplexSet Spacetime::getExternalSimplices() noexcept {
  SimplexSet simplices_{};
  SimplexSet result{};
  for (const auto &simplex : simplices) {
    if (simplex->hasCausallyAvailableFacet()) result.insert(simplex);
  }
  return result;
}

SimplexSet Spacetime::getSimplicesWithOrientation(std::tuple<uint8_t, uint8_t> orientation) {
  SimplexOrientation o{std::get<0>(orientation), std::get<1>(orientation)};
  SimplexSet result{};
  for (const auto &simplex : simplices) {
    if (simplex->getOrientation() == o) result.insert(simplex);
  }
  return result;
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
  auto it = simplices.find(simplex);
  if (it == simplices.end()) {
    return nullptr;
  }
  return *it;
}

SimplexPtr Spacetime::getSimplex(std::uint64_t fingerprint) const {
  auto it = simplices.find(fingerprint);
  if (it == simplices.end()) {
    return nullptr;
  }
  return *it;
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

  // Collect simplices containing EXACTLY ONE of v1, v2.
  // Simplices containing both are unaffected (XOR fingerprint is symmetric).
  struct AffectedSimplex { SimplexPtr ptr; std::uint64_t oldFp; };
  std::vector<AffectedSimplex> affected;

  for (const auto &s : v1->getSimplices()) {
    if (!s->hasVertex(v2))
      affected.push_back({s, s->fingerprint.fingerprint()});
  }
  for (const auto &s : v2->getSimplices()) {
    if (!s->hasVertex(v1))
      affected.push_back({s, s->fingerprint.fingerprint()});
  }

  // Phase 1: Remove affected simplices from hash tables (fingerprints still valid)
  for (auto &[s, oldFp] : affected) {
    simplices.erase(s);
    simplexVecIndex.erase(oldFp);
    topSimplexVecIndex.erase(oldFp);
  }

  // Record which affected simplices contained v1 vs v2 BEFORE swapping IDs
  // (hasVertex uses internal ID maps that become stale after the swap)
  std::unordered_set<SimplexPtr> containsV1;
  for (const auto &s : v1->getSimplices()) {
    if (!s->hasVertex(v2)) containsV1.insert(s);
  }

  // Phase 2: Swap vertex IDs, rekey vertex list
  v1->setId(id2);
  v2->setId(id1);
  vertexList->swapKeys(id1, id2);

  // Update internal ID maps on ALL simplices containing either vertex
  // (including sub-simplices like facets and edges, not just top-simplices)
  for (const auto &s : v1->getSimplices()) s->updateVertexId(id1, id2);
  for (const auto &s : v2->getSimplices()) s->updateVertexId(id2, id1);

  // Update fingerprints on affected simplices (those in hash tables)
  for (auto &[s, oldFp] : affected) {
    if (containsV1.count(s)) {
      s->fingerprint.removeId(id1);
      s->fingerprint.addId(id2);
    } else {
      s->fingerprint.removeId(id2);
      s->fingerprint.addId(id1);
    }
    s->fingerprint.refresh();
  }

  // Phase 3: Re-insert into hash tables with new fingerprints
  for (auto &[s, oldFp] : affected) {
    auto newFp = s->fingerprint.fingerprint();
    simplices.insert(s);

    // Rekey simplexOwner (owns the Simplex allocation)
    auto nh = simplexOwner.extract(oldFp);
    if (!nh.empty()) {
      nh.key() = newFp;
      simplexOwner.insert(std::move(nh));
    }

    // Restore vec index mappings (vec position unchanged, just the key)
    for (std::size_t i = 0; i < simplicesVec.size(); ++i) {
      if (simplicesVec[i] == s) { simplexVecIndex[newFp] = i; break; }
    }
    for (std::size_t i = 0; i < topSimplicesVec.size(); ++i) {
      if (topSimplicesVec[i] == s) { topSimplexVecIndex[newFp] = i; break; }
    }
  }
}

bool Spacetime::removeIfIsolated(const VertexPtr &vertex) const noexcept {
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
#ifdef CASET_ASSERTIONS
  std::unordered_set<std::uint64_t> seen{};
  for (const auto &simp : simplices) {
    if (seen.contains(simp->fingerprint.fingerprint())) {
      CLOG(CRITICAL_LEVEL, "Duplicate simplex!");
      throw std::runtime_error("Duplicate simplex!");
    }
    seen.insert(simp->fingerprint.fingerprint());
  }
#endif
  auto fp = simplex->fingerprint.fingerprint();
  const auto &[it, inserted] = simplices.emplace(simplex);
  if (inserted) {
    simplexOwner.emplace(fp, std::unique_ptr<Simplex>(simplex));
    simplexVecIndex[fp] = simplicesVec.size();
    simplicesVec.push_back(simplex);
    // Track top-dimensional simplices separately for efficient random access
    auto d = metric->getSignature()->getDimensions();
    if (simplex->size() == static_cast<std::size_t>(d + 1)) {
      topSimplexVecIndex[fp] = topSimplicesVec.size();
      topSimplicesVec.push_back(simplex);
    }
    updateOrientationCounters(simplex, +1);
  }
  return *it;
}

void Spacetime::unregisterSimplex(const SimplexPtr &simplex) {
  if (!simplices.contains(simplex)) {
#ifdef CASET_ASSERTIONS
    CLOG(CRITICAL_LEVEL, "You attempted to unregister a simplex that does not exist! ", simplex->toString(), " existing simplices are: ");
    for (const auto &s : simplices) {
      CLOG(CRITICAL_LEVEL, "    - ", s->toString());
    }
    for (const auto &s : simplices) {
      if (s->fingerprint.fingerprint() == simplex->fingerprint.fingerprint()) {
        CLOG(CRITICAL_LEVEL, "Hash table said a simplex was not registered, but one was found!");
        throw std::runtime_error("registered simplex unexpectedly found. hash table corrupted.");
      }
    }
#endif
    return;
  }
  updateOrientationCounters(simplex, -1);
  simplices.erase(simplex);
  // Remove from parallel vector via index map (O(1) swap-and-pop)
  auto fp = simplex->fingerprint.fingerprint();
  auto idxIt = simplexVecIndex.find(fp);
  if (idxIt != simplexVecIndex.end()) {
    std::size_t idx = idxIt->second;
    if (idx < simplicesVec.size() - 1) {
      // Swap with last element and update the swapped element's index
      auto backFp = simplicesVec.back()->fingerprint.fingerprint();
      simplicesVec[idx] = simplicesVec.back();
      simplexVecIndex[backFp] = idx;
    }
    simplicesVec.pop_back();
    simplexVecIndex.erase(idxIt);
  }
  // Remove from top-dimensional vector too
  auto topIdxIt = topSimplexVecIndex.find(fp);
  if (topIdxIt != topSimplexVecIndex.end()) {
    std::size_t idx = topIdxIt->second;
    if (idx < topSimplicesVec.size() - 1) {
      auto backFp = topSimplicesVec.back()->fingerprint.fingerprint();
      topSimplicesVec[idx] = topSimplicesVec.back();
      topSimplexVecIndex[backFp] = idx;
    }
    topSimplicesVec.pop_back();
    topSimplexVecIndex.erase(topIdxIt);
  }
  // Free the Simplex allocation
  simplexOwner.erase(fp);
}

void Spacetime::reserve(int nSimplices) {
  simplices.reserve(nSimplices);
  simplexOwner.reserve(nSimplices);
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

const SimplexSet& Spacetime::getSimplices() const noexcept {
  return simplices;
}

VertexPtr Spacetime::getRandomVertex() {
  auto verts = vertexList->toVector();
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
  for (const auto &s : simplices) {
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

} // caset
