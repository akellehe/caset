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

#include <ranges>
#include <ATen/core/interned_strings.h>
#include <c10/util/ThreadLocalDebugInfo.h>

namespace caset {
std::vector<SimplexPtr > Simplex::getFacets() {
#if CASET_DEBUG
  if (getVertices().empty()) throw std::runtime_error("Simplex is empty");
#endif
  if (getVertices().size() == 1) {
#if CASET_DEBUG
    validate();
#endif
    return {};
  }
  if (!facets.empty()) return facets;
  auto verts = getVertices();
  facets.reserve(verts.size());
  for (int skip = 0; skip < verts.size(); skip++) {
    const auto &skipVertex = verts[skip]->getId();
    Vertices faceVertices{};
    Edges faceEdges{};
    faceEdges.reserve(verts.size());
    faceVertices.reserve(verts.size());
    faceVertices.insert(faceVertices.end(), verts.begin(), verts.begin() + skip);
    faceVertices.insert(faceVertices.end(), verts.begin() + skip + 1, verts.end());
    for (const auto &e : getEdges()) {
      if (!e->hasVertex(skipVertex)) faceEdges.push_back(e);
    }
    SimplexPtr facet = Simplex::create(faceVertices, faceEdges);
    facet->addCoface(shared_from_this());
    facets.push_back(facet);
    if (!facet->getOrientation()->isDegenerate()) availableFacetsByOrientation[facet->getOrientation()].insert(facet);
  }
#if CASET_DEBUG
  validate();
#endif
  return facets;
}

///
/// @param vertices_
Simplex::Simplex(
  const Vertices &vertices_, Edges edges_
) : orientation(std::make_shared<SimplexOrientation>(0, 0)), vertices(vertices_), edges(std::move(edges_)), fingerprint({}) {
#if CASET_DEBUG
  if (vertices_.empty()) throw std::runtime_error("Simplex is empty");
#endif

  orientation = SimplexOrientation::orientationOf(vertices_);
}

Simplex::Simplex(
  const Vertices &vertices_, Edges edges_,
  const SimplexOrientationPtr &orientation_
) : orientation(orientation_), vertices(vertices_), edges(std::move(edges_)), fingerprint({}) {
#if CASET_DEBUG
  if (vertices_.empty()) throw std::runtime_error("Simplex is empty");
#endif
}

SimplexPtr Simplex::create(const Vertices &vertices_, const Edges &edges_) {
#if CASET_DEBUG
  if (vertices_.empty()) throw std::runtime_error("Simplex is empty");
#endif
  SimplexPtr simplex = std::make_shared<Simplex>(vertices_, edges_);
  simplex->initialize(simplex);

  return simplex;
}

SimplexPtr Simplex::create(const Vertices &vertices_, const Edges &edges_, const SimplexOrientationPtr &orientation_) {
#if CASET_DEBUG
  if (vertices_.empty()) throw std::runtime_error("Simplex is empty");
#endif
  SimplexPtr simplex = std::make_shared<Simplex>(vertices_, edges_, orientation_);
  simplex->initialize(simplex);
  return simplex;
}

void Simplex::initialize(const SimplexPtr &simplex) {
  std::vector<IdType> ids = {};
  ids.reserve(vertices.size());
  vertexIdLookup.reserve(vertices.size());
  for (const auto &v : vertices) {
    ids.push_back(v->getId());
    vertexIdLookup.insert({v->getId(), v});
    v->addSimplex(simplex);
  }
  fingerprint = Fingerprint(ids);
#if CASET_DEBUG
  if (getVertexIdLookup().empty()) throw std::runtime_error("Simplex is empty");
#endif

#if CASET_DEBUG
  validate();
#endif
}

std::string Simplex::toString() const {
  std::stringstream ss;
  ss << "<";
  ss << "(" << std::to_string(std::get<0>(getOrientation()->numeric()));
  ss << ", " << std::to_string(std::get<1>(getOrientation()->numeric()));
  ss << ")-" << std::to_string(getOrientation()->getK());
  ss << "-Simplex (";
  for (const auto &v : vertices) {
    ss << v->toString() << "→";
  }
  if (!vertices.empty()) {
    ss << vertices[0]->toString() << ")>";
  } else {
    ss << ")>";
  }
  return ss.str();
}

[[nodiscard]] SimplexOrientationPtr Simplex::getOrientation() const noexcept {
  return orientation;
}

[[nodiscard]] Vertices Simplex::getVertices() const noexcept { return vertices; };

[[nodiscard]] std::size_t Simplex::size() const noexcept {
  return vertices.size();
}

[[nodiscard]] bool Simplex::isTimelike() const {
  for (const auto &edge : getEdges()) {
    if (!vertexIdLookup.contains(edge->getSourceId())) {
      CLOG(ERROR_LEVEL,
           "vertexIdLookup was missing source ID ",
           edge->toString(),
           " in simplex ",
           toString(),
           ". edges should all be internal");
      throw std::runtime_error("vertexIdLookup was missing source ID");
    }
    if (!vertexIdLookup.contains(edge->getTargetId())) {
      CLOG(ERROR_LEVEL,
           "vertexIdLookup was missing target ID ",
           edge->toString(),
           " in simplex ",
           toString(),
           ". edges should all be internal");
      throw std::runtime_error("vertexIdLookup was missing target ID");
    }
    const auto &src = vertexIdLookup.find(edge->getSourceId())->second;
    const auto &tgt = vertexIdLookup.find(edge->getTargetId())->second;
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
  auto k = getOrientation()->getK();
  return binomial<std::size_t>(k + 1, j + 1);
}

std::size_t Simplex::getNumberOfEdges() const {
  auto k = getOrientation()->getK();
  return (k + 1) * k / 2;
}

void Simplex::addCoface(const SimplexPtr &simplex) {
  cofaces.insert(simplex);
#if CASET_DEBUG
  simplex->validate();
  validate();
#endif
}

[[nodiscard]] bool Simplex::hasCoface(const SimplexPtr &simplex) const {
  for (const auto &s : cofaces) {
    if (s->fingerprint.fingerprint() == simplex->fingerprint.fingerprint()) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] bool Simplex::hasVertex(const IdType vertexId) const {
  return vertexIdLookup.contains(vertexId);
}

[[nodiscard]] bool Simplex::hasEdgeContaining(const IdType vertexId) const {
  for (const auto &e : getEdges()) {
    if (e->getSourceId() == vertexId) return true;
    if (e->getTargetId() == vertexId) return true;
  }
  return false;
}

void Simplex::validate() const {
  for (const auto &e : getEdges()) {
    if (!hasVertex(e->getSourceId())) {
      CLOG(ERROR_LEVEL, "Missing source for one of its edges: ", e->toString());
      throw std::runtime_error("Missing source for one of its edges.");
    }
    if (!hasVertex(e->getTargetId())) {
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

/// @returns Edges in traversal order (the order of input vertices).
[[nodiscard]] Edges Simplex::getEdges() const {
  return edges;
}

using RemoveEdgeByPtr = bool (Simplex::*)(const EdgePtr &);

[[nodiscard]]
std::optional<Vertices>
Simplex::getVerticesWithParityTo(const SimplexPtr &other) const {
  CLOG(DEBUG_LEVEL, "Simplex::getVerticesWithParityTo. Simplex 1: ", toString(), "\nSimplex 2: ", other->toString());
  const auto &mine = vertices;
  const auto &theirs = other->getVertices();

  const std::size_t n = mine.size();
  CLOG(INFO_LEVEL, "Attempting to align ", std::to_string(n), " vertices.");
  if (n != theirs.size()) {
    throw std::runtime_error("You can only compare simplices of the same size!");
  }
  if (isTimelike() != other->isTimelike()) {
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
    -> std::optional<Vertices> {
        Vertices result{};
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
    // 1. Try same orientation
    if (auto aligned = try_alignment(i, /*reversed=*/false)) return aligned;

    // 2. Try reversed orientation
    if (auto aligned_rev = try_alignment(i, /*reversed=*/true)) return aligned_rev;
  }

  // No alignment found
  return std::nullopt;
}

[[nodiscard]] bool Simplex::hasEdge(const EdgePtr &edge) const {
  if (!hasVertex(edge->getSourceId())) {
    return false;
  }
  if (!hasVertex(edge->getTargetId())) {
    return false;
  }
  for (const auto &e : getEdges()) {
    if (e->getSourceId() == edge->getSourceId() && e->getTargetId() == edge->getTargetId()) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] bool Simplex::hasEdge(const IdType vertexAId, const IdType vertexBId) const {
  const EdgePtr edge = std::make_shared<Edge>(vertexAId, vertexBId);
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

  Vertices otherVertices = other->getVertices();
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

[[nodiscard]] SimplexSet
Simplex::getCofaces() const noexcept {
  return cofaces;
}

bool Simplex::operator==(const Simplex &other) const noexcept {
  return fingerprint.fingerprint() == other.fingerprint.fingerprint();
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
  return getNumberOfFaces(getOrientation()->getK());
}

SimplexOrientations Simplex::getGluableFaceOrientations() {
  SimplexOrientations allowedOrientations{};
  if (facets.empty()) {
    CLOG(WARN_LEVEL, "Simplex::getGluableFaceOrientations(): facets empty" );
    getFacets();
    CLOG(WARN_LEVEL, "No we have ", facets.size(), " facets and ", availableFacetsByOrientation.size(), " orientations ");
  }
  for (const auto &o : availableFacetsByOrientation | std::views::keys) {
    allowedOrientations.push_back(o);
  }
  return allowedOrientations;
}

SimplexSet Simplex::getAvailableFacetsByOrientation(const SimplexOrientationPtr &orientation) {
  if (facets.empty()) {
    for (const auto &facet : getFacets()) {
      // Facets that have never been computed are always causally available, no need to check here.
      // if (facet->getOrientation()->isDegenerate()) continue;
      if (facet->isTimelike()) continue; // TODO: At some point I think we want to connect the next time slot, but not sure.
      availableFacetsByOrientation[facet->getOrientation()].insert(facet);
    }
  }
  if (availableFacetsByOrientation.contains(orientation)) {
    return availableFacetsByOrientation.at(orientation);
  }
  return {};
}

bool Simplex::operator==(const SimplexPtr &other) const noexcept {
  return fingerprint.fingerprint() == other->fingerprint.fingerprint();
}

/// This simplex is the unattached simplex.
void Simplex::attach(const VertexPtr &unattached, const VertexPtr &attached, const std::shared_ptr<EdgeList> &edgeList, const std::shared_ptr<VertexList> &vertexList) {
  const auto [oldEdges, newEdges] = unattached->moveEdgesTo(attached, edgeList, vertexList);
  for (const auto &simplex : unattached->getSimplices()) {
    simplex->replaceVertex(unattached, attached);
  }
  for (const auto &edgeKey : newEdges) {
    edgeList->get(edgeKey)->addSimplex(shared_from_this()); // TODO: Remove the old simplex!
  }
  if (unattached->degree() == 0) vertexList->remove(unattached);
#if CASET_DEBUG
  validate();
#endif
}

void Simplex::markAsUnavailable() {
#if CASET_DEBUG
  if (isCausallyAvailable()) CLOG(ERROR_LEVEL, "Facet is still available!");
#endif
  for (const auto &coface : getCofaces()) {
    coface->markFacetAsUnavailable(shared_from_this());
  }
}

bool Simplex::replaceVertex(const VertexPtr &oldVertex, const VertexPtr &newVertex) {
  if (!hasVertex(oldVertex->getId())) {
    return false;
  }
  if (hasVertex(newVertex->getId())) {
#if CASET_DEBUG
    validate();
#endif
    return false;
  }
  std::vector<IdType> vertexIds = {};
  vertexIds.reserve(vertices.size());
  for (int i = 0; i < vertices.size(); i++) {
    if (vertices[i]->getId() == oldVertex->getId()) {
      vertices[i] = newVertex;
      vertexIdLookup.erase(oldVertex->getId());
      vertexIdLookup.insert({newVertex->getId(), newVertex});
    }
    vertexIds.push_back(vertices[i]->getId());
  }
  oldVertex->removeSimplex(shared_from_this());
  newVertex->addSimplex(shared_from_this());
  fingerprint.refreshFingerprint(vertexIds);
  for (const auto &e : getEdges()) {
    if (e->hasVertex(oldVertex->getId())) {
      if (e->getSourceId() == oldVertex->getId()) {
        e->replaceSourceVertex(newVertex->getId());
      } else {
        e->replaceTargetVertex(newVertex->getId());
      }
    }
  }
#if CASET_DEBUG
  validate();
#endif
  return true;
}

VertexIdMap Simplex::getVertexIdLookup() const noexcept {
  return vertexIdLookup;
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
}
}
