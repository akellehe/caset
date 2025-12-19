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

#include "Vertex.h"
#include "Simplex.h"
#include "ForwardDeclarations.h"
#include "SimplexOrientation.h"
#include "spacetime/Spacetime.h"
#include "Logger.h"
#include "utils.h"

#include <algorithm>
#include <ATen/core/interned_strings.h>
#include <c10/util/ThreadLocalDebugInfo.h>

namespace caset {
bool Simplex::hasFacets() const {
  return !facets.empty();
}

#ifdef CASET_ASSERTIONS
class SimplexCorruptionDetector : public CorruptionDetector<SimplexPtr, SimplexPtrHash, SimplexPtrEq> {
};
#endif

const std::vector<SimplexPtr> &Simplex::getFacets() {
#if CASET_ASSERTIONS
  if (getVertices().empty()) throw std::runtime_error("Simplex is empty");
#endif
  if (getVertices().size() == 1) {
#if CASET_ASSERTIONS
    validate();
#endif
    return facets;
  }

  if (facets.empty()) {
    facets.reserve(getVertices().size());
    CLOG(DEBUG_LEVEL, "Computing new facets for ", toString(), "!!");
    facets.reserve(vertices.size());
    auto verts = getVertices();
    for (int skip = 0; skip < verts.size(); skip++) {
      const auto &skipVertex = verts[skip]->getId();
      VertexPtrs faceVertices{};
      Edges faceEdges{};
      faceEdges.reserve(verts.size());
      faceVertices.reserve(verts.size());
      faceVertices.insert(faceVertices.end(), verts.begin(), verts.begin() + skip);
      faceVertices.insert(faceVertices.end(), verts.begin() + skip + 1, verts.end());
      for (const auto &e : getEdges()) {
        if (!e->hasVertex(skipVertex)) faceEdges.push_back(e);
      }
      const auto &[facet, inserted] = spacetime->createSimplex(faceVertices, faceEdges); // Gets or creates!
      if (inserted) {
        SimplexPtr coface = spacetime->getSimplex(this->fingerprint.fingerprint());
        if (coface != nullptr) facet->addCoface(coface);
      }
      facets.push_back(facet);
    }
  }
#if CASET_ASSERTIONS
  for (const auto &f : facets) {
    if (!isCofaceTo(f)) {
      CLOG(DEBUG_LEVEL, toString(), " is not a coface to ", f->toString());
      std::abort();
    }
  }
  validate();
#endif
  return facets;
}

///
/// @param vertices_
Simplex::Simplex(
  Spacetime *spacetime_,
  const VertexPtrs &vertices_,
  Edges edges_
) : spacetime(spacetime_), orientation(SimplexOrientation::orientationOf(vertices_)), vertices(vertices_),
    edges(edges_.begin(), edges_.end()),
    fingerprint({0}) {
#if CASET_ASSERTIONS
  if (vertices_.empty()) throw std::runtime_error("Simplex is empty");
#endif
}

Simplex::Simplex(
  Spacetime *spacetime_,
  const VertexPtrs &vertices_,
  Edges edges_,
  const SimplexOrientation &orientation_
) : spacetime(spacetime_), orientation(orientation_), vertices(vertices_), edges(edges_.begin(), edges_.end()),
    fingerprint() {
  for (const auto &v : vertices_) {
    fingerprint.addId(v->getId());
  }
  fingerprint.refresh();
#if CASET_ASSERTIONS
  if (vertices_.empty()) throw std::runtime_error("Simplex is empty");
#endif
}

SimplexPtr Simplex::create(Spacetime *spacetime_, const VertexPtrs &vertices_, const Edges &edges_) {
#if CASET_ASSERTIONS
  if (vertices_.empty()) throw std::runtime_error("Simplex is empty");
#endif
  SimplexPtr simplex = std::make_shared<Simplex>(spacetime_, vertices_, edges_);
  if (!simplex->initialized) {
    simplex->initialize(simplex);
  }
  // TODO: Here for some reason we add the simplex to it's vertices multiple times.
  return simplex;
}

SimplexPtr Simplex::create(Spacetime *spacetime_,
                           const VertexPtrs &vertices_,
                           const Edges &edges_,
                           const SimplexOrientation &orientation_) {
#if CASET_ASSERTIONS
  if (vertices_.empty()) throw std::runtime_error("Simplex is empty");
#endif
  SimplexPtr simplex = std::make_shared<Simplex>(spacetime_, vertices_, edges_, orientation_);
  simplex->initialize(simplex);
  return simplex;
}

void Simplex::initialize(const SimplexPtr &simplex) {
#ifdef CASET_ASSERTIONS
  if (initialized) {
    CLOG(DEBUG_LEVEL, "You attempted to re-initialize a simplex! Behavior is undefined.");
    std::abort();
  }
#endif
  std::vector<IdType> ids = {};
  ids.reserve(vertices.size());
  vertexIdToIndex.reserve(vertices.size());
  vertexIndexToId.reserve(vertices.size());
  for (const auto &v : vertices) {
    vertexIdToIndex.emplace(v->getId(), ids.size());
    vertexIndexToId.emplace(ids.size(), v->getId());
    ids.push_back(v->getId());
  }
  fingerprint.setIds(ids);

  // We have to register AFTER the fingerprint is set:
  Simplex::registerToVertices(simplex);
  initialized = true;
}

void Simplex::removeCoface(const SimplexPtr &coface) {
#if CASET_ASSERTIONS
  CLOG(DEBUG_LEVEL, "Removing coface ", coface->toString(), " from simplex ", toString());
  if (coface == nullptr || coface.get() == nullptr) {
    CLOG(DEBUG_LEVEL, "Coface was null");
    std::abort();
  }
  if (SimplexCorruptionDetector::isCorrupted(cofaces)) {
    CLOG(DEBUG_LEVEL, "Corruption detected");;
    std::abort();
  }
#endif
  ownershipManager.erase(coface->toString(), toString() + "::cofaces", cofaces, coface);
  // cofaces.erase(coface);
#if CASET_ASSERTIONS
  if (coface == nullptr || coface.get() == nullptr) {
    CLOG(DEBUG_LEVEL, "Coface was null");
    std::abort();
  }
  if (SimplexCorruptionDetector::isCorrupted(cofaces)) {
    CLOG(DEBUG_LEVEL, "Corruption detected");;
    std::abort();
  }
  if (SimplexCorruptionDetector::wouldDuplicate(cofaces, coface)) {
    CLOG(DEBUG_LEVEL, "Failed to remove coface!");
    CLOG(DEBUG_LEVEL, "All cofaces: ");
    for (const auto &c : cofaces) {
      CLOG(DEBUG_LEVEL, "    - ", c->toString());
    }
    std::abort();
  }
#endif
}

Simplices Simplex::unregisterFromFacets(const SimplexPtr &coface) {
  if (!coface->hasFacets()) return {}; // They just haven't been computed.
  const auto &facets_ = coface->getFacets();
  for (const auto &f : facets_) {
#ifdef CASET_ASSERTIONS
    if (!f->hasCoface(coface)) {
      CLOG(DEBUG_LEVEL,
           f->toString(),
           " did not contain coface!",
           coface->toString());
      throw std::runtime_error("Simplex did not contain coface!");
    }
#endif
    f->removeCoface(coface);
#ifdef CASET_ASSERTIONS
    if (f->hasCoface(coface)) {
      CLOG(DEBUG_LEVEL, "Failed to remove coface!");
      throw std::runtime_error("Failed to remove coface!");
    }
#endif
  }
  return facets_;
}

void Simplex::registerToFacets(const SimplexPtr &coface) {
  if (!coface->hasFacets()) {
    CLOG(DEBUG_LEVEL, "Coface had no facets, not registering to facets.");
    return; // Facets not yet computed.
  }
  for (const auto &f : coface->getFacets()) {
#ifdef CASET_ASSERTIONS
    if (f == nullptr || f.get() == nullptr) {
      CLOG(DEBUG_LEVEL, "Facet was null!");
      std::abort();
    }
    if (coface == nullptr || coface.get() == nullptr) {
      CLOG(DEBUG_LEVEL, "Simplex was null!");
      std::abort();
    }
    if (!coface->isCofaceTo(f)) {
      CLOG(DEBUG_LEVEL, coface->toString(), " is not a coface of ", f->toString());
      std::abort();
    }
    if (f->hasCoface(coface)) {
      CLOG(DEBUG_LEVEL, "Facet already contains coface!");
      std::abort();
    }
#endif
    f->addCoface(coface);
  }
}

void Simplex::unregisterFromVertices(const SimplexPtr &simplex) {
  for (const auto &owner : simplex->getVertices()) {
    owner->removeSimplex(simplex);
  }
}

void Simplex::registerToVertices(const SimplexPtr &simplex) {
  for (const auto &owner : simplex->getVertices()) {
    owner->addSimplex(simplex);
  }
}

#ifdef CASET_VERBOSE
std::string Simplex::toString() const noexcept {
  std::stringstream sigmaLabel;
  sigmaLabel << std::to_string(getOrientation().getK()) << "-";
  sigmaLabel << "\\sigma";

  std::stringstream orientationStr;
  orientationStr << "^{(" << std::to_string(std::get<0>(getOrientation().numeric())) << "/";
  orientationStr << std::to_string(std::get<1>(getOrientation().numeric())) << ")}";

  std::string fp = std::to_string(fingerprint.fingerprint());
  std::string fpShort = fp.substr(0, 3) + fp.substr(fp.size() - 3, 3);
  std::stringstream fpStr;
  fpStr << "_{" << fpShort << "}";

  std::stringstream vertexStr;
  std::vector<IdType> vids{};
  for (const auto &v : vertices) vids.push_back(v->getId());
  std::sort(vids.begin(), vids.end());
  for (const auto &v : vids) {
    vertexStr << std::to_string(v);
    if (v != vids[vids.size() - 1]) {
      vertexStr << "|";
    }
  }

  std::stringstream ss;
  ss << "<" << sigmaLabel.str() << orientationStr.str() << fpStr.str() << " " << vertexStr.str() << ">";
  return latexToUtf8(ss.str());
}
#endif

[[nodiscard]] SimplexOrientation Simplex::getOrientation() const noexcept {
  return orientation;
}

[[nodiscard]] VertexPtrs Simplex::getVertices() const noexcept { return vertices; };

[[nodiscard]] bool Simplex::isTimelike() const {
  for (const auto &edge : getEdges()) {
#ifdef CASET_ASSERTIONS
    if (!vertexIdToIndex.contains(edge->getSource()->getId())) {
      CLOG(ERROR_LEVEL,
           "vertexIdLookup was missing source ID ",
           edge->toString(),
           " in simplex ",
           toString(),
           ". edges should all be internal");
      throw std::runtime_error("vertexIdLookup was missing source ID");
    }
    if (!vertexIdToIndex.contains(edge->getTarget()->getId())) {
      CLOG(ERROR_LEVEL,
           "vertexIdLookup was missing target ID ",
           edge->toString(),
           " in simplex ",
           toString(),
           ". edges should all be internal");
      throw std::runtime_error("vertexIdLookup was missing target ID");
    }
#endif
    const auto srcIndex = vertexIdToIndex.find(edge->getSource()->getId())->second;
    const auto tgtIndex = vertexIdToIndex.find(edge->getTarget()->getId())->second;

#ifdef CASET_ASSERTIONS
    if (srcIndex >= vertices.size()) {
      throw std::runtime_error("You requested a src vertex with an index outside the vertex list size.");
    }
    if (tgtIndex >= vertices.size()) {
      throw std::runtime_error("You requested a tgt vertex with an index outside the vertex list size.");
    }
#endif
    const auto &src = vertices[srcIndex];
    const auto &tgt = vertices[tgtIndex];
    if (src->getTime() != tgt->getTime()) return false;
  }
  return true;
}

[[nodiscard]] std::size_t Simplex::computeNumberOfEdges(std::size_t k) {
  if (k == 4) return 6;
  if (k == 3) return 3;
  if (k == 2) return 1;
  if (k == 0 || k == 1) return 0;

  int n = 0;
  for (int i = 0; i < k; i++) {
    n = n + i;
  }
  return n;
}

template<typename T>
T Simplex::binomial(unsigned n, unsigned k) const {
  if (k > n) return 0;
  k = std::min(k, n - k);

  T result = 1;
  for (unsigned i = 1; i <= k; ++i) {
    result = result * (n - (k - i));
    result /= i;
  }

  return result;
}

std::size_t Simplex::getNumberOfFaces(std::size_t j) const {
  auto k = getOrientation().getK();
  return binomial<std::size_t>(k + 1, j + 1);
}

std::size_t Simplex::getNumberOfEdges() const {
  auto k = getOrientation().getK();
  return (k + 1) * k / 2;
}

void Simplex::addCoface(const SimplexPtr &coface) {
#if CASET_ASSERTIONS
  if (coface == nullptr || coface.get() == nullptr) {
    CLOG(DEBUG_LEVEL, "Coface was null");
    std::abort();
  }
  if (!coface->isCofaceTo(shared_from_this())) {
    CLOG(DEBUG_LEVEL, coface->toString(), " is not a coface of ", toString());
    throw std::runtime_error("You attempted to add a coface to a facet for which it is not a coface!");
  }
  if (SimplexCorruptionDetector::isCorrupted(cofaces)) {
    CLOG(DEBUG_LEVEL, "Corruption detected");;
    std::abort();
  }
  if (SimplexCorruptionDetector::wouldDuplicate(cofaces, coface)) {
    CLOG(DEBUG_LEVEL, "You attempted to add a duplicate coface: ", coface->toString(), " to simplex ", toString());
    CLOG(DEBUG_LEVEL, "All cofaces: ");
    for (const auto &c : cofaces) {
      CLOG(DEBUG_LEVEL, "    - ", c->toString());
    }
    std::abort();
  }
  CLOG(INFO_LEVEL, "Adding ", coface->toString(), " as coface to ", toString());
  const auto &[it, inserted] = ownershipManager.insert(coface->toString(), toString() + "::cofaces", cofaces, coface);
  if ((*it)->toString() != coface->toString()) {
    CLOG(DEBUG_LEVEL, "Iterator ", (*it)->toString(), " did not match coface ", coface->toString());
    std::abort();
  }
  if (coface == nullptr || coface.get() == nullptr) {
    CLOG(DEBUG_LEVEL, "Coface was null");
    std::abort();
  }
  if (SimplexCorruptionDetector::isCorrupted(cofaces)) {
    CLOG(DEBUG_LEVEL, "Corruption detected");;
    std::abort();
  }
  if (inserted) {
    CLOG(DEBUG_LEVEL, "Added ", coface->toString(), " to ", toString(), "!");
  } else {
    CLOG(DEBUG_LEVEL, "Failed to add ", coface->toString(), " to ", toString(), "!");
  }
#else
  cofaces.insert(coface);
#endif
}

[[nodiscard]] bool Simplex::hasCoface(const SimplexPtr &coface) const {
#ifdef CASET_ASSERTIONS
  if (SimplexCorruptionDetector::isCorrupted(cofaces)) {
    CLOG(DEBUG_LEVEL, "Corruption detected!");
    std::abort();
  }
#endif
  for (const auto &c : cofaces) {
    if (c == coface) return true;
  }
  return false;
  // Unsafe for corrupted tables:
  return cofaces.contains(coface);
}

[[nodiscard]] bool Simplex::hasVertex(const VertexPtr &vertex) const {
  return vertexIdToIndex.contains(vertex->getId());
}

[[nodiscard]] bool Simplex::hasEdgeContaining(const IdType vertexId) const {
  for (const auto &e : getEdges()) {
    if (e->getSource()->getId() == vertexId) return true;
    if (e->getTarget()->getId() == vertexId) return true;
  }
  return false;
}

void Simplex::validate() const {
  for (const auto &e : getEdges()) {
    if (!hasVertex(e->getSource())) {
      CLOG(ERROR_LEVEL, "Missing source for one of its edges: ", e->toString());
      throw std::runtime_error("Missing source for one of its edges.");
    }
    if (!hasVertex(e->getTarget())) {
      CLOG(ERROR_LEVEL, "Missing target for one of it's edges: ", e->toString());
      throw std::runtime_error("Missing target for one of its edges.");
    }
    if (getVertices().size() == 1) return; // A 0-simplex will have no edges.
    for (const auto &v : getVertices()) {
      if (!hasEdgeContaining(v->getId())) {
        CLOG(ERROR_LEVEL, "Missing an edge for vertex: ", v->toString(), " on simplex ", toString(), " with edges:");
        for (const auto &e2 : getEdges()) {
          CLOG(ERROR_LEVEL, "    - ", e2->toString());
        }
        throw std::runtime_error("Missing an edge for a vertex.");
      }
    }
  }
}

/// TODO: Optimize this method by tracking state on the `edges` member
[[nodiscard]] EdgePtrSet Simplex::getEdges() const {
  EdgePtrSet edges_{};
  edges_.reserve(getNumberOfEdges());
  for (const auto &vertex : getVertices()) {
    for (const auto &edge : vertex->getEdges()) {
      if (hasVertex(edge->getSource()) && hasVertex(edge->getTarget())) {
        edges_.insert(edge);
      }
    }
  }
  return edges_;
}

[[nodiscard]]
std::optional<VertexPtrs>
Simplex::getVerticesWithParityTo(const SimplexPtr &other) const {
  const auto &mine = vertices;
  const auto &theirs = other->getVertices();

  const std::size_t n = mine.size();
  if (n != theirs.size()) {
    throw std::runtime_error("You can only compare simplices of the same size!");
  }
  if (isTimelike() && !other->isTimelike() || !isTimelike() && other->isTimelike()) {
    throw std::runtime_error("Can't establish parity when one face is timelike and the other is not!");
  }
  if (n == 0) return std::nullopt;
  if (n == 1) {
    if (mine[0]->getTime() != theirs[0]->getTime()) return std::nullopt;
    return mine; // already aligned
  }

  auto try_alignment =
      [&](std::size_t start,
          bool reversed)
    -> std::optional<VertexPtrs> {
    VertexPtrs result{};
    result.reserve(n);

    for (std::size_t k = 0; k < n; ++k) {
      std::size_t idx;
      if (!reversed) {
        // orientation-preserving: walk forward
        idx = (start + k) % n;
      } else {
        // orientation-reversing: walk backward
        // k = 0 -> idx = start
        // k = 1 -> idx = start - 1 (mod n)
        idx = (start + n - k) % n;
      }

      if (mine[idx]->getTime() != theirs[k]->getTime()) {
        return std::nullopt; // mismatch, this alignment fails
      }

      result.push_back(mine[idx]);
    }
    return result; // success
  };

  // Try all starting positions where times match theirs[0]
  for (std::size_t i = 0; i < n; ++i) {
    if (mine[i]->getTime() != theirs[0]->getTime()) continue;

    // 1. Try same orientation
    if (auto aligned = try_alignment(i, /*reversed=*/false)) return aligned;

    // 2. Try reversed orientation
    if (auto aligned_rev = try_alignment(i, /*reversed=*/true)) return aligned_rev;
  }

  // No alignment found
  return std::nullopt;
}

[[nodiscard]] bool Simplex::hasEdge(const EdgePtr &edge) const {
  if (!hasVertex(edge->getSource())) {
    return false;
  }
  if (!hasVertex(edge->getTarget())) {
    return false;
  }
  for (const auto &e : getEdges()) {
    if (e->getSource()->getId() == edge->getSource()->getId() && e->getTarget()->getId() == edge->getTarget()->
      getId()) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] bool Simplex::hasEdge(const VertexPtr &vertexA, const VertexPtr &vertexB) const {
  const EdgePtr edge = std::make_shared<Edge>(vertexA, vertexB);
  return hasEdge(edge);
}

int8_t Simplex::checkParity(const SimplexPtr &other) const {
  std::size_t K = vertices.size();

  // Build vertex -> position map for 'a'
  // For small K (≤4,5) you could linear search; this is generic.
  std::unordered_map<IdType, int> positionByVertexIdInA{};
  positionByVertexIdInA.reserve(K);
  for (int i = 0; i < K; ++i) {
    positionByVertexIdInA[vertices[i]->getId()] = i;
  }

  VertexPtrs otherVertices = other->getVertices();
  std::vector<IdType> otherIds{};
  otherIds.reserve(K);
  for (int i = 0; i < K; ++i) {
    otherIds[i] = otherVertices[i]->getId();
  }

  std::vector<int> perm{};
  perm.reserve(K);
  for (int i = 0; i < K; ++i) {
    IdType otherId = otherIds[i];
    if (!positionByVertexIdInA.contains(otherId)) {
      return 0;
    }
    perm[i] = positionByVertexIdInA[otherId];
  }

  // Count cycles of perm on {0..K-1}
  std::vector<bool> visited{};
  visited.reserve(K);
  for (int i = 0; i < K; i++) {
    visited[i] = false;
  }
  int cycles = 0;
  for (int i = 0; i < K; ++i) {
    if (visited[i]) continue;
    ++cycles;
    int j = i;
    while (!visited[j]) {
      visited[j] = true;
      j = perm[j];
    }
  }

  // This might be wrong now that we fixed the k+1 vs k bug
  int N = K;
  int transpositionsMod2 = (N - cycles) & 1;
  return transpositionsMod2 ? -1 : +1;
}

[[nodiscard]] SimplexPtrSet
Simplex::getCofaces() const noexcept {
  return cofaces;
}

bool Simplex::isCofaceTo(const SimplexPtr &facet, bool shallow) const {
  if (shallow) {
    if (getOrientation().getK() != facet->getOrientation().getK() + 1) {
      return false;
    }
  }
  for (const auto &v : facet->getVertices()) {
    if (!hasVertex(v)) return false;
  }
  return true;
}

bool Simplex::operator==(const Simplex &other) const noexcept {
  return fingerprint.fingerprint() == other.fingerprint.fingerprint();
}

bool Simplex::operator==(const SimplexPtr &other) const noexcept {
  return fingerprint.fingerprint() == other->fingerprint.fingerprint();
}

std::uint64_t Simplex::hash() const noexcept {
  return fingerprint.fingerprint();
}

bool Simplex::isCausallyAvailable() const noexcept {
  return getCofaces().size() < 2;
}

bool Simplex::hasCausallyAvailableFacet() {
  for (const auto &face : getFacets()) {
    if (face->getCofaces().size() < 2) return true;
  }

  return false;
}

bool Simplex::isInternal() const noexcept {
  return getCofaces().size() == 2;
}

std::size_t Simplex::maxKPlusOneCofaces() const {
  return getNumberOfFaces(getOrientation().getK());
}

SimplexOrientationSet Simplex::getGluableFaceOrientations() {
  SimplexOrientationSet allowedOrientations{};
  for (const auto &face : getFacets()) {
    if (face->getCofaces().size() < 2) {
      allowedOrientations.insert(face->getOrientation());
    }
  }
  return allowedOrientations;
}

Simplices Simplex::clearFacets() {
  if (facets.empty()) return {};
  Simplices simplices{facets.begin(), facets.end()};
  for (const auto &facet : simplices) {
    removeFacet(facet);
  }
  facets.clear();
  return simplices;
}

Simplices Simplex::clearCofaces() {
  if (cofaces.empty()) return {};
  Simplices simplices{cofaces.begin(), cofaces.end()};
  for (const auto &coface : simplices) {
    removeCoface(coface);
  }
  cofaces.clear();
  return simplices;
}

std::tuple<SimplexPtr, Simplices, Simplices> Simplex::breakReferences(const SimplexPtr &simplex) {
  simplex->spacetime->unregisterSimplex(simplex);
  for (const auto &v : simplex->getVertices()) {
    v->removeSimplex(simplex);
  }

  // Need to clear cofaces of the facets (the facets may not include the vertex being replaced, but they DO reference the
  // coface (simplex in this case), which DOES contain the vertex being replaced.

  const auto cofaces_ = simplex->clearCofaces();
  const auto facets_ = simplex->clearFacets();

  // TODO: Move this to detect the coface via the vertex being replaced.
  for (const auto &facet : facets_) {
    if (facet->hasCoface(simplex)) {
      facet->removeCoface(simplex);
    }
  }
  // for (const auto &coface : cofaces_) {
    // if (coface->hasFacets()) {
      // coface->removeFacet(simplex);
    // }
  // }

#ifdef CASET_ASSERTIONS
  if (simplex->hasFacets()) {
    CLOG(DEBUG_LEVEL, "Simplex still has facets after clearing them!");
    std::abort();
  }
  if (!simplex->getCofaces().empty()) {
    CLOG(DEBUG_LEVEL, "Simplex still has cofaces!");
    std::abort();
  }
#endif
  return {simplex, cofaces_, facets_};
}

bool Simplex::addFacet(const SimplexPtr &simplex) {
  facets.push_back(simplex);
  return true;
}

bool Simplex::removeFacet(const SimplexPtr &facet) {
  auto it = std::find(facets.begin(), facets.end(), facet);
  if (it != facets.end()) {
    facets.erase(it);
    return true;
  }
  return false;
}

void Simplex::restoreReferences(SimplexPtr &simplex, const Simplices &cofaces_, const Simplices &facets_) {
  for (const auto &v : simplex->getVertices()) {
    v->addSimplex(simplex);
  }
  for (const auto c : cofaces_) {
    simplex->addCoface(c);
  }
  for (const auto f : facets_) {
    simplex->addFacet(f);
  }
}

std::uint64_t Simplex::size() const noexcept {
  return vertices.size();
}

/// This simplex is the unattached simplex.
void Simplex::attach(const VertexPtr &unattached,
                     const VertexPtr &attached) {
  CLOG(DEBUG_LEVEL, "================================================================================================");
  CLOG(DEBUG_LEVEL,
       "Attaching unattached vertex, ",
       unattached->toString(),
       " to attached vertex ",
       attached->toString());
  CLOG(DEBUG_LEVEL, "================================================================================================");
  // I think we might need to sort the simplices by dimension
  SimplexPtrSet simplicesToProcessAsSet = unattached->getSimplices();  // Get all simplices that reference this vertex.
  // TODO: The unattached vertex belongs to a facet on a coface. We need to ensure those facets/cofaces are all replaced
  //  by those on the spacetime rather than accidentally duplicating them by replacing vertices such that they collide
  //  with existing simplices.
  Simplices simplicesToProcess{simplicesToProcessAsSet.begin(), simplicesToProcessAsSet.end()};

  // Sort simplices to Process by dimension(k), descending.
  std::sort(simplicesToProcess.begin(),
            simplicesToProcess.end(),
            [](const SimplexPtr &a, const SimplexPtr &b) {
              return a->getOrientation().getK() < b->getOrientation().getK();
            });

  // Need to dereference everything, and store what was dereferenced.
  std::vector<std::tuple<SimplexPtr, Simplices, Simplices>> brokenReferences{}; // simplex, cofaces, facets
  brokenReferences.reserve(simplicesToProcess.size());

  for (const auto &simplex : simplicesToProcess) {
    CLOG(DEBUG_LEVEL, "Unregistering ", simplex->toString(), "...");
    brokenReferences.push_back(Simplex::breakReferences(simplex));
#ifdef CASET_ASSERTIONS
    if (spacetime->getSimplex(simplex)) {
      CLOG(DEBUG_LEVEL, "Spacetime still has simplex ", simplex->toString(), "!!");
    }
#endif
  }

#ifdef CASET_ASSERTIONS
  for (const auto &[
    simplex,
    brokenCofaces,
    brokenFacets
    ] : brokenReferences) {
    for (const auto &bcf : brokenCofaces) {
      if (bcf->referencesSimplex(simplex)) {
        CLOG(DEBUG_LEVEL, bcf->toString(), " still references ", simplex->toString());
        std::abort();
      }
    }
    for (const auto &f : brokenFacets) {
      if (f->referencesSimplex(simplex)) {
        CLOG(DEBUG_LEVEL, f->toString(), " still references ", simplex->toString());
        std::abort();
      }
    }
  }
#endif

  // Now simplex belongs nowhere (except facets, a vector). Free to modify without corrupting hash tables.
  auto [oldEdges, newEdges] = unattached->moveEdgesTo(attached, spacetime);
  for (const auto &simplex : simplicesToProcess) {
    simplex->replaceVertex(unattached, attached);
    // TODO: Note that if we replaced a vertex such that e.g. a facet now collides with an existing Simplex in the
    //  spacetime; THAT FACET MUST BE REPLACED EVERYWHERE.
  }

#ifdef CASET_ASSERTIONS
  for (const auto &[simplex, brokenCofaces, brokenFacets] : brokenReferences) {
    for (const auto &bcf : brokenCofaces) {
      if (bcf->referencesSimplex(simplex)) {
        CLOG(DEBUG_LEVEL, bcf->toString(), " still references ", simplex->toString());
        std::abort();
      }
    }
    for (const auto &f : brokenFacets) {
      if (f->referencesSimplex(simplex)) {
        CLOG(DEBUG_LEVEL, f->toString(), " still references ", simplex->toString());
        std::abort();
      }
    }
  }
#endif

  for (auto [simplex, brokenCofaces, brokenFacets] : brokenReferences) {
    // TODO: May need to check here whether or not the simplex is internal or external. I'm pretty sure this will always
    //  be internal as long as attach() is only used to attach previously unattached simplexes.
    auto registeredSimplex = spacetime->registerSimplex(simplex, !simplex->isCausallyAvailable());
    CLOG(DEBUG_LEVEL, "RE-Registering ", simplex->toString(), "to vertices and facets...");
    // May also need to check for brokenCofaces and brokenFacets in the already registered simplices!
    Simplices registeredBrokenCofaces{};
    registeredBrokenCofaces.reserve(brokenCofaces.size());
    for (const auto &bcf : brokenCofaces) {
      auto registeredCoface = spacetime->registerSimplex(bcf, bcf->isCausallyAvailable());
      registeredBrokenCofaces.push_back(registeredCoface);
    }
    Simplices registeredBrokenFacets{};
    registeredBrokenFacets.reserve(brokenFacets.size());
    for (const auto &bf : brokenFacets) {
      auto registeredFacet = spacetime->registerSimplex(bf, bf->isCausallyAvailable());
      registeredBrokenFacets.push_back(registeredFacet);
    }

    Simplex::restoreReferences(registeredSimplex, registeredBrokenCofaces, registeredBrokenFacets);
  }
  CLOG(DEBUG_LEVEL, "Done attaching.");
  if (unattached->degree() == 0) spacetime->getVertexList()->remove(unattached);
#if CASET_ASSERTIONS
  validate();
#endif
  CLOG(DEBUG_LEVEL, "------------------------------------------------------------------------------------------------");
}

bool Simplex::replaceVertex(const VertexPtr &oldVertex, const VertexPtr &newVertex) {
  // TODO: Probably make this cascade, but we should just go to the Vertex for things to cascade to.
  if (hasVertex(newVertex)) {
#if CASET_ASSERTIONS
    validate();
#endif
    return false;
  }
  auto oldId = oldVertex->getId();
  auto oldIndexIt = vertexIdToIndex.find(oldId);
  if (oldIndexIt == vertexIdToIndex.end()) {
    return false;
  }
  auto oldIndex = oldIndexIt->second;
#ifdef CASET_ASSERTIONS
  if (oldIndex >= vertices.size()) {
    CLOG(DEBUG_LEVEL,
         "You requested an index: ",
         std::to_string(oldIndex),
         " larger than the number of vertices in the simplex: ",
         vertices.size());
    throw std::runtime_error("out of range.");
  }
  if (vertices.size() != vertexIdToIndex.size()) {
    throw std::runtime_error(
      "Vertices not keeping up with id to index mapping: " + std::to_string(vertices.size()) + " != " + std::to_string(
        vertexIdToIndex.size()));
  }
  if (vertices.size() != vertexIndexToId.size()) {
    throw std::runtime_error(
      "Vertices not keeping up with id to index mapping: " + std::to_string(vertices.size()) + " != " + std::to_string(
        vertexIndexToId.size()));
  }
#endif
  vertices[oldIndex] = newVertex;

  vertexIdToIndex.erase(oldId);
  vertexIdToIndex.emplace(newVertex->getId(), oldIndex);

  vertexIndexToId.erase(oldIndex);
  vertexIndexToId.emplace(oldIndex, newVertex->getId());

  fingerprint.removeId(oldId);
  fingerprint.addId(newVertex->getId());

  // CRITICAL: Clear the facets cache because the facets are computed based on vertices.
  // If we don't clear this, getFacets() will return stale facets with the old vertices.
  // facets.clear();
  // cofaces.clear();

  return true;
}

VertexIdMap Simplex::getVertexIdLookup() const noexcept {
  VertexIdMap lookup{};
  for (const auto [vertexId, index] : vertexIdToIndex) {
    lookup.emplace(vertexId, vertices[index]);
  }
  return lookup;
}

template<typename Method, typename... Args>
bool Simplex::cascade(Method method, bool up, bool down, Args &&... args) {
  std::deque<SimplexPtr> simplicesToUpdate;
  SimplexSet seen;
  auto enqueueIfNew = [&](const SimplexPtr &s) {
    if (!seen.contains(s)) simplicesToUpdate.push_back(s);
  };

  // --- Cascade to siblings --- //
  for (const auto &coface : getCofaces()) {
    for (const auto &sibling : coface->getFacets()) {
      if (sibling->fingerprint.fingerprint() == fingerprint.fingerprint()) continue;
      (sibling.get()->*method)(std::forward<Args>(args)...);
    }
  }

  // --- Cascading to cofaces ---
  if (up && !cofaces.empty()) {
    simplicesToUpdate.insert(simplicesToUpdate.end(),
                             cofaces.begin(),
                             cofaces.end());
    while (!simplicesToUpdate.empty()) {
      const auto coface = simplicesToUpdate.front(); // copy the shared_ptr
      simplicesToUpdate.pop_front();

      if (!seen.insert(coface).second) {
        continue;
      }

      // Call the member function on this coface
      if ((coface.get()->*method)(std::forward<Args>(args)...)) {
        for (const auto &nextCoface : coface->getCofaces()) {
          enqueueIfNew(nextCoface);
        }
      }
    }
  }

  // --- Cascading to facets ---
  auto facets_ = getFacets();
  if (down && !facets_.empty()) {
    simplicesToUpdate.clear();
    simplicesToUpdate.insert(simplicesToUpdate.end(),
                             facets_.begin(),
                             facets_.end());
    while (!simplicesToUpdate.empty()) {
      const auto facet = simplicesToUpdate.front(); // copy, NOT reference
      simplicesToUpdate.pop_front();

      if (!seen.insert(facet).second) continue;

      if ((facet.get()->*method)(std::forward<Args>(args)...)) {
        for (const auto &nextFacet : facet->getFacets()) {
          enqueueIfNew(nextFacet);
        }
      }
    }
  }
  return true;
};

bool Simplex::removeEdge(const EdgePtr &edge) {
  return edges.erase(edge) > 0;
}

bool Simplex::addEdge(const EdgePtr &edge) {
  const auto [it, inserted] = edges.emplace(edge);
  return inserted;
}

bool Simplex::hasStoredFacet(const SimplexPtr &facet) {
  if (facets.empty()) return false;
  for (const auto &f : facets) {
    if (f == facet) return true;
  }
  return false;
}

bool Simplex::referencesSimplex(const SimplexPtr &simplex) {
  for (const auto &f : facets) {
    if (f == simplex) {
      CLOG(DEBUG_LEVEL, "A facet of ", toString(), " was equal to the simplex; ", f->toString(), "==", simplex->toString());
      return true;
    }
    if (f->hasCoface(simplex)) {
      CLOG(DEBUG_LEVEL, "A facet of ", toString(), " referenced the simplex as a coface; ", f->toString(), "->", simplex->toString());
      return true;
    }
    if (f->hasStoredFacet(simplex)) {
      CLOG(DEBUG_LEVEL, "A facet ", toString(), " referenced the simplex as a facet: ", f->toString(), "->", simplex->toString());
      return true;
    }
  }
  for (const auto &c : cofaces) {
    if (c == simplex) {
      CLOG(DEBUG_LEVEL, "A coface of ", toString(), " was equal to the simplex; ", c->toString(), "==", simplex->toString());
      return true;
    }
    if (c->hasCoface(simplex)) {
      CLOG(DEBUG_LEVEL, "A coface of ", toString(), " references the simplex as a coface: ", c->toString(), "->", simplex->toString());
      return true;
    }
    if (c->hasStoredFacet(simplex)) {
      CLOG(DEBUG_LEVEL, "A coface of ", toString(), " references the simplex as a facet: ", c->toString(), "->", simplex->toString());
      return true;
    }
  }
  return false;
}
}
