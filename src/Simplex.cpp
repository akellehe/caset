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
// #include <ATen/core/interned_strings.h>
// #include <c10/util/ThreadLocalDebugInfo.h>

namespace caset {

template<int D>
bool Simplex<D>::hasFacets() const {
  return !facets.empty();
}

#ifdef CASET_ASSERTIONS
class SimplexCorruptionDetector : public CorruptionDetector<SimplexPtr, SimplexPtrHash, SimplexPtrEq> {
};
#endif

template<int D>
const std::vector<SimplexPtr> &Simplex<D>::getFacets() {
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
    const auto &verts = vertices;  // Use member directly, avoid copy
    const std::size_t n = verts.size();
    const std::size_t facetSize = n - 1;

    facets.reserve(n);

    // CRITICAL OPTIMIZATION: Cache edges once before loop
    const auto &allEdges = getEdges();

    // Pre-compute coface once
    SimplexPtr coface = spacetime->getSimplex(this->fingerprint.fingerprint());

    for (std::size_t skip = 0; skip < n; ++skip) {
      const auto skipVertexId = verts[skip]->getId();

      // Build faceVertices efficiently in one pass
      VertexPtrs faceVertices{};
      faceVertices.reserve(facetSize);
      for (std::size_t i = 0; i < n; ++i) {
        if (i != skip) faceVertices.push_back(verts[i]);
      }

      // Filter edges without the skipped vertex
      Edges faceEdges{};
      faceEdges.reserve(facetSize);  // Approximate size
      for (const auto &e : allEdges) {
        if (!e->hasVertex(skipVertexId)) faceEdges.push_back(e);
      }

      const auto &[facet, inserted] = spacetime->createSimplex(faceVertices, faceEdges); // Gets or creates!
      if (inserted && coface != nullptr) {
        facet->addCoface(coface);
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
template<int D>
Simplex<D>::Simplex(
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

template<int D>
Simplex<D>::Simplex(
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

template<int D>
SimplexPtr Simplex<D>::create(Spacetime *spacetime_, const VertexPtrs &vertices_, const Edges &edges_) {
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

template<int D>
bool Simplex<D>::isInitialized() const noexcept { return initialized; }

template<int D>
SimplexPtr Simplex<D>::create(Spacetime *spacetime_,
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

template<int D>
void Simplex<D>::initialize(const SimplexPtr &simplex) {
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
    ti = std::min(ti, v->getTime());
    tf = std::max(tf, v->getTime());
    vertexIdToIndex.emplace(v->getId(), ids.size());
    vertexIndexToId.emplace(ids.size(), v->getId());
    ids.push_back(v->getId());
  }
  fingerprint.setIds(ids);
  _isTimelike = ti == tf;

  // We have to register AFTER the fingerprint is set:
  registerToVertices(simplex);
  initialized = true;

  if (ti != tf) {
    CLOG(INFO_LEVEL, "ti != tf: ", std::to_string(ti), " != ", std::to_string(tf), " for ", toString());
  } else {
    CLOG(INFO_LEVEL, "ti == tf: ", std::to_string(ti), " != ", std::to_string(tf), " for ", toString());
  }
}

template<int D>
double Simplex<D>::getTi() const noexcept {
  return ti;
}

template<int D>
double Simplex<D>::getTf() const noexcept {
  return tf;
}

template<int D>
void Simplex<D>::registerToVertices(const SimplexPtr &simplex) {
  for (const auto &owner : simplex->getVertices()) {
    owner->addSimplex(simplex);
  }
}

#ifdef CASET_VERBOSE
template<int D>
std::string Simplex<D>::toString() const noexcept {
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

template<int D>
[[nodiscard]] SimplexOrientation Simplex<D>::getOrientation() const noexcept {
  return orientation;
}

template<int D>
[[nodiscard]] VertexPtrs Simplex<D>::getVertices() const noexcept { return vertices; };

template<int D>
[[nodiscard]] bool Simplex<D>::isTimelike() const {
  CLOG(INFO_LEVEL, "===============================", toString(), "========================================");
  CLOG(INFO_LEVEL, "ti: ", std::to_string(ti));
  CLOG(INFO_LEVEL, "tf: ", std::to_string(tf));
  for (const auto &v : vertices) {
    CLOG(INFO_LEVEL, "Checking vertex ", v->toString(), " with time ", std::to_string(v->getTime()));
  }
  CLOG(INFO_LEVEL, "---------------------------------------------------------------------------------------------------");
  return ti == tf;
}

template<int D>
[[nodiscard]] std::size_t Simplex<D>::computeNumberOfEdges(std::size_t k) {
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

template<typename T, int D>
T Simplex<D>::binomial(unsigned n, unsigned k) const {
  if (k > n) return 0;
  k = std::min(k, n - k);

  T result = 1;
  for (unsigned i = 1; i <= k; ++i) {
    result = result * (n - (k - i));
    result /= i;
  }

  return result;
}

template<int D>
std::size_t Simplex<D>::getNumberOfFaces(std::size_t j) const {
  auto k = getOrientation().getK();
  return binomial<std::size_t>(k + 1, j + 1);
}

template<int D>
std::size_t Simplex<D>::getNumberOfEdges() const {
  auto k = getOrientation().getK();
  return (k + 1) * k / 2;
}

template<int D>
void Simplex<D>::addCoface(const SimplexPtr &coface) {
#if CASET_ASSERTIONS
  if (coface == nullptr || coface.get() == nullptr) {
    CLOG(DEBUG_LEVEL, "Coface was null");
    std::abort();
  }
  if (!coface->isCofaceTo(std::make_shared<Simplex>(*this))) { // Don't allow this to happen in production.
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

template<int D>
[[nodiscard]] bool Simplex<D>::hasCoface(const SimplexPtr &coface) const {
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

template<int D>
[[nodiscard]] bool Simplex<D>::hasVertex(const VertexPtr &vertex) const {
  return vertexIdToIndex.contains(vertex->getId());
}

template<int D>
[[nodiscard]] bool Simplex<D>::hasEdgeContaining(const IdType vertexId) const {
  for (const auto &e : getEdges()) {
    if (e->getSource()->getId() == vertexId) return true;
    if (e->getTarget()->getId() == vertexId) return true;
  }
  return false;
}

template<int D>
void Simplex<D>::validate() const {
#ifdef CASET_ASSERTIONS
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
#endif
}

template<int D>
[[nodiscard]] EdgePtrSet Simplex<D>::getEdges() const {
  return edges;
}

template<int D>
[[nodiscard]] bool Simplex<D>::hasEdge(const EdgePtr &edge) const {
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

template<int D>
[[nodiscard]] bool Simplex<D>::hasEdge(const VertexPtr &vertexA, const VertexPtr &vertexB) const {
  const EdgePtr edge = std::make_shared<Edge>(vertexA, vertexB);
  return hasEdge(edge);
}

template<int D>
[[nodiscard]] SimplexPtrSet
Simplex<D>::getCofaces() const noexcept {
  return cofaces;
}

template<int D>
bool Simplex<D>::isCofaceTo(const SimplexPtr &facet, bool shallow) const {
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

template<int D>
bool Simplex<D>::operator==(const Simplex &other) const noexcept {
  return fingerprint.fingerprint() == other.fingerprint.fingerprint();
}

template<int D>
bool Simplex<D>::operator==(const SimplexPtr &other) const noexcept {
  return fingerprint.fingerprint() == other->fingerprint.fingerprint();
}

template<int D>
std::uint64_t Simplex<D>::hash() const noexcept {
  return fingerprint.fingerprint();
}

template<int D>
bool Simplex<D>::isCausallyAvailable() const noexcept {
  return getCofaces().size() < 2;
}

template<int D>
bool Simplex<D>::hasCausallyAvailableFacet() {
  for (const auto &face : getFacets()) {
    if (face->isTimelike()) continue;
    if (face->getCofaces().size() < 2) return true;
  }
  return false;
}

template<int D>
bool Simplex<D>::isInternal() const noexcept {
  return getCofaces().size() == 2;
}

template<int D>
std::size_t Simplex<D>::maxKPlusOneCofaces() const {
  return getNumberOfFaces(getOrientation().getK());
}

template<int D>
std::uint64_t Simplex<D>::size() const noexcept {
  return vertices.size();
}

template<int D>
bool Simplex<D>::replaceVertex(const VertexPtr &oldVertex, const VertexPtr &newVertex) {
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

template<int D>
VertexIdMap Simplex<D>::getVertexIdLookup() const noexcept {
  VertexIdMap lookup{};
  for (const auto [vertexId, index] : vertexIdToIndex) {
    lookup.emplace(vertexId, vertices[index]);
  }
  return lookup;
}

template<int D>
bool Simplex<D>::removeEdge(const EdgePtr &edge) {
  return edges.erase(edge) > 0;
}

template<int D>
bool Simplex<D>::addEdge(const EdgePtr &edge) {
  const auto [it, inserted] = edges.emplace(edge);
  return inserted;
}

template<int D>
bool Simplex<D>::hasStoredFacet(const SimplexPtr &facet) {
  if (facets.empty()) return false;
  for (const auto &f : facets) {
    if (f == facet) return true;
  }
  return false;
}

std::pair<SimplexPtr, Simplices> Simplex<D>::cone(VertexPtr &vertex) {
  auto signature = spacetime->getMetric()->getSignature();
  auto foliation = spacetime->getFoliation();
  if (signature->getSignatureType() == SignatureType::Lorentzian) {
    // We have to preserve causality. That means if we cone to e.g. a (1, 3) facet (one vertex at \f$ t \f$, 3 at
    // \f$ t+1 \f$) with a (1, 4) coface; then the new simplex has to be a (2, 3) simplex with (2, 3) - (1, 3) = (1, 0)
    // so we have to create a new vertex at time \f$ t \f$ rather than \f$ t+1 \f$ (which would have been the second
    // slot)
    // In general given a \f$ (n, m) \f$ simplex with a \f$ (n-1, m) \f$ or \f$ (n, m-1) \f$ facet; we have to match the
    // facet, but then what happens next depends on the foliation (preferred or not). If the foliation is preferred;
    // then we need a layer of timelike edges between every layer of spacelike edges. In order to ensure we only pair
    // compatible simplices; we just have to ensure the vertices stay balanced on either end of the spacelike sheet.
    //
    // If we have a e.g. a (3, 1) simplex with a (2, 1) facet, then we have (3, 1) - (2, 1) = (1, 0) = 1 extra vertex
    // at \f$ t \f$ . So we need to add the vertex with which we cone at \f$ t = t+1 \f$ to make the new coface a (2, 2)
    // simplex.
    if (foliation == Foliation::PREFERRED) {
      auto [facet_ti, facet_tf] = getOrientation().numeric();
      auto [coface_ti, coface_tf] = (*cofaces.begin())->getOrientation().numeric();
      if (coface_ti > facet_ti) {
        // Need an extra tf vertex.
        vertex->setTime(getTf());
      } else if (coface_tf > facet_tf) {
        // Need an extra ti vertex.
        vertex->setTime(getTi());
      }
    }
  }
  VertexPtrs kPlusOneVertices{vertices.begin(), vertices.end()};
  Edges newEdges{edges.begin(), edges.end()};
  for (auto &existing : kPlusOneVertices) {
    if (existing->getTime() == vertex->getTime()) {
      newEdges.push_back(spacetime->createEdge(existing, vertex, -(spacetime->getAlpha() * spacetime->getA())));
    } else {
      newEdges.push_back(spacetime->createEdge(existing, vertex, spacetime->getA()));
    }
  }
  kPlusOneVertices.push_back(vertex);
  auto [kSimplex, created] = spacetime->createSimplex(kPlusOneVertices, newEdges);
  Simplices newFacets{};
  auto myFingerprint = fingerprint.fingerprint();
  for (const auto &f : kSimplex->getFacets()) {
    if (f->fingerprint.fingerprint() != myFingerprint) {
      facets.push_back(f);
    }
  }
  return {kSimplex, facets};
}

}
