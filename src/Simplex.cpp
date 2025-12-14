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
#include "utils.h"
#include "Fingerprint.h"
#include "Edge.h"


namespace caset {
std::vector<SimplexRawPtr> Simplex::getFacets() {
#if CASET_DEBUG
  if (getVertices().empty()) throw std::runtime_error("Simplex is empty");
#endif
  if (getVertices().size() == 1) {
#if CASET_DEBUG
    validate();
#endif
    return {};
  }
  if (facets.empty()) {
    auto verts = getVertices();
    facets.reserve(verts.size());
    for (int skip = 0; skip < verts.size(); skip++) {
      const auto &skipVertex = verts[skip];
      VertexPtrs faceVertices{};
      Edges faceEdges{};
      faceEdges.reserve(verts.size());
      faceVertices.reserve(verts.size());
      faceVertices.insert(faceVertices.end(), verts.begin(), verts.begin() + skip);
      faceVertices.insert(faceVertices.end(), verts.begin() + skip + 1, verts.end());
      for (const auto e : getEdges()) {
        if (!e->hasVertex(skipVertex.get())) faceEdges.push_back(e);
      }
      // TODO: Simplex::create should probably only be called by Spacetime, which holds canonical simplices.
      std::unique_ptr<Simplex> facet = Simplex::create(faceVertices, faceEdges);
      facet->addCoface(this);
      facets.push_back(std::move(facet));
    }
#if CASET_DEBUG
    validate();
#endif
  }
  std::vector<SimplexRawPtr> rawFacets{};
  rawFacets.reserve(getVertices().size());
  for (const auto &f : facets) {
    rawFacets.push_back(f.get());
  }
  return rawFacets;
}

///
/// @param vertices_
Simplex::Simplex(
  const VertexPtrs &vertices_,
  Edges edges_
) : orientation(std::make_shared<SimplexOrientation>(0, 0)), vertices(vertices_), edges(edges_.begin(), edges_.end()),
    fingerprint({}) {
#if CASET_DEBUG
  if (vertices_.empty()) throw std::runtime_error("Simplex is empty");
#endif

  orientation = SimplexOrientation::orientationOf(vertices_);
}

Simplex::Simplex(
  const VertexPtrs &vertices_,
  Edges edges_,
  const SimplexOrientationPtr &orientation_
) : orientation(orientation_), vertices(vertices_), edges(edges_.begin(), edges_.end()), fingerprint({}) {
#if CASET_DEBUG
  if (vertices_.empty()) throw std::runtime_error("Simplex is empty");
#endif
}

std::unique_ptr<Simplex> Simplex::create(const VertexPtrs &vertices_, Edges edges_) {
#if CASET_DEBUG
  if (vertices_.empty()) throw std::runtime_error("Simplex is empty");
#endif
  std::unique_ptr<Simplex> simplex = std::make_unique<Simplex>(vertices_, edges_);
  simplex->initialize(simplex.get());
  return simplex;
}

std::unique_ptr<Simplex> Simplex::create(const VertexPtrs &vertices_,
                                         Edges edges_,
                                         const SimplexOrientationPtr &orientation_) {
#if CASET_DEBUG
  if (vertices_.empty()) throw std::runtime_error("Simplex is empty");
#endif
  std::unique_ptr<Simplex> simplex = std::make_unique<Simplex>(vertices_, edges_, orientation_);
  simplex->initialize(simplex.get());
  return simplex;
}

void Simplex::initialize(SimplexRawPtr simplex) {
  std::vector<IdType> ids = {};
  ids.reserve(vertices.size());
  for (const auto &v : vertices) {
    ids.push_back(v->getId());
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
  ss << std::to_string(getOrientation()->getK());
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

[[nodiscard]] VertexPtrs Simplex::getVertices() const noexcept { return vertices; };

[[nodiscard]] std::size_t Simplex::size() const noexcept {
  return vertices.size();
}

[[nodiscard]] bool Simplex::isTimelike() const {
  const auto startTime = vertices[0]->getTime();
  for (const auto &v : vertices) {
    if (v->getTime() != startTime) return false;
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

void Simplex::addCoface(SimplexRawPtr simplex) {
  cofaces.insert(simplex);
#if CASET_DEBUG
  simplex->validate();
  validate();
#endif
}

[[nodiscard]] bool Simplex::hasCoface(SimplexRawPtr simplex) const {
  for (const auto s : cofaces) {
    if (s->fingerprint.fingerprint() == simplex->fingerprint.fingerprint()) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] bool Simplex::hasVertex(const IdType vertexId) {
  for (const auto &v : vertices) {
    if (v->getId() == vertexId) return true;
  }
  return false;
}

[[nodiscard]] bool Simplex::hasVertex(const Vertex *vertex) {
  for (const auto &v : vertices)
    if (v.get() == vertex) return true;
  return false;
}

[[nodiscard]] bool Simplex::hasEdge(const Edge *edge) const {
  for (const auto &e : edges) {
    if (e == edge) return true;
  }
  return false;
}

[[nodiscard]] bool Simplex::hasEdgeContaining(const IdType vertexId) const {
  for (auto e : edges) {
    if (e->getSource()->getId() == vertexId) return true;
    if (e->getTarget()->getId() == vertexId) return true;
  }
  return false;
}

void Simplex::validate() const {
  // for (auto e : getEdges()) {
    // CLOG(INFO_LEVEL, "Validating edge ", e->toString());
    // if (!hasVertex(e->getSourceId())) {
      // CLOG(ERROR_LEVEL, "Missing source for one of its edges: ", e->toString());
      // throw std::runtime_error("Missing source for one of its edges.");
    // }
    // if (!hasVertex(e->getTargetId())) {
      // CLOG(ERROR_LEVEL, "Missing target for one of it's edges: ", e->toString());
      // throw std::runtime_error("Missing target for one of its edges.");
    // }
    // if (getVertices().size() == 1) return; // A 0-simplex will have no edges.
  // }
  // for (const auto &v : getVertices()) {
    // if (!hasEdgeContaining(v->getId())) {
      // CLOG(ERROR_LEVEL,
           // "There was no edge containing ",
           // v->getId(),
           // " for vertex: ",
           // v->toString(),
           // " on simplex ",
           // toString(),
           // ". Existing edges are:");
      // for (auto e2 : getEdges()) {
        // CLOG(ERROR_LEVEL, "    - ", e2->toString());
      // }
      // throw std::runtime_error("Missing an edge for a vertex.");
    // }
  // }
}

/// @returns Edges in traversal order (the order of input vertices).
[[nodiscard]] const EdgeSet &Simplex::getEdges() const noexcept {
  return edges;
}

[[nodiscard]] std::vector<std::shared_ptr<Edge>> Simplex::getEdgesForPython() const noexcept {
  std::vector<std::shared_ptr<Edge>> result{};
  for (const auto e : getEdges()) {
    result.push_back(std::make_shared<Edge>(e->getSource(), e->getTarget(), e->getSquaredLength()));
  }
  return result;
}

bool Simplex::removeEdge(Edge *edge) {
  bool removed = edges.erase(edge) > 0;
  if (removed) {
    edge->removeSimplex(this);
  }
  return removed;
}

std::pair<Edge *, bool> Simplex::addEdge(Edge *edge) {
  auto [it, inserted] = edges.insert(edge);
  if (inserted) edge->addSimplex(this);
  return {*it, inserted};
}

[[nodiscard]]
std::optional<VertexPtrs>
Simplex::getVerticesWithParityTo(SimplexRawPtr other) const {
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

int8_t Simplex::checkParity(SimplexRawPtr other) const {
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

[[nodiscard]] std::unordered_set<SimplexRawPtr>
Simplex::getCofaces() const noexcept {
  return cofaces;
}

[[nodiscard]] py::list
Simplex::getCofacesForPython() {
  py::list cofacesForPython{};
  for (auto cof : getCofaces()) {
    cofacesForPython.append(wrap_non_owning(cof));
  }
  return cofacesForPython;
}

[[nodiscard]]
py::list
Simplex::getFacetsForPython() {
  py::list facetsForPython{};
  for (auto facet : getFacets()) {
    facetsForPython.append(wrap_non_owning(facet));
  }
  return facetsForPython;
}

bool Simplex::operator==(const Simplex &other) const noexcept {
  if (vertices.size() != other.vertices.size()) return false;
  for (int i = 0; i < vertices.size(); ++i) {
    if (vertices[i] != other.vertices[i]) return false;
  }
  return true;
}

bool Simplex::isCausallyAvailable() const noexcept {
  return getCofaces().size() < 2;
}

bool Simplex::hasCausallyAvailableFacet() {
  for (const auto face : getFacets()) {
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

std::unordered_set<SimplexOrientationPtr>
Simplex::getGluableFaceOrientations() {
  auto allowedOrientations = std::unordered_set<SimplexOrientationPtr>{};
  for (const auto face : getFacets()) {
    if (face->getCofaces().size() < 2) {
      allowedOrientations.insert(face->getOrientation());
    }
  }
  return allowedOrientations;
}

bool Simplex::operator==(SimplexRawPtr other) const noexcept {
  return fingerprint.fingerprint() == other->fingerprint.fingerprint();
}

bool Simplex::replaceVertex(const VertexPtr &oldVertex, const VertexPtr &newVertex) {
  bool replaced = false;
  std::vector<IdType> vertexIds = {};
  vertexIds.reserve(vertices.size());
  for (int i = 0; i < vertices.size(); i++) {
    if (vertices[i]->getId() == oldVertex->getId()) {
      vertices[i] = newVertex;
      replaced = true;
    }
    vertexIds.push_back(vertices[i]->getId());
  }
  if (!replaced) return false;
  oldVertex->removeSimplex(this);
  newVertex->addSimplex(this);
  fingerprint.refreshFingerprint(vertexIds);

#if CASET_DEBUG
  validate();
#endif
  return true;
}

VertexIdMap
Simplex::getVertexIdLookup() const noexcept {
  VertexIdMap vertexIdMap{};
  for (const auto &v : vertices) {
    vertexIdMap.insert({v->getId(), v});
  }
  return vertexIdMap;
}
}
