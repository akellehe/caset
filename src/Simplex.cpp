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

#include "Simplex.h"

namespace caset {
std::vector<std::shared_ptr<Simplex> > Simplex::getFacets() noexcept {
  if (!facets.empty()) return facets;
  auto verts = getVertices();
  facets.reserve(verts.size());
  for (int skip = 0; skip < verts.size(); skip++) {
    std::vector<std::shared_ptr<Vertex> > faceVertices;
    faceVertices.reserve(verts.size());
    faceVertices.insert(faceVertices.end(), verts.begin(), verts.begin() + skip);
    faceVertices.insert(faceVertices.end(), verts.begin() + skip + 1, verts.end());
    std::shared_ptr<Simplex> facet = std::make_shared<Simplex>(faceVertices);
    facet->addCoface(shared_from_this());
    facets.push_back(facet);
  }
  return facets;
}

///
/// @param vertices_
Simplex::Simplex(
  Vertices vertices_
) : orientation(std::make_shared<SimplexOrientation>(0, 0)), vertices(vertices_), fingerprint({}) {
  orientation = SimplexOrientation::orientationOf(vertices_);
  initialize(vertices_);
}

Simplex::Simplex(
  Vertices vertices_,
  SimplexOrientationPtr orientation_
) : orientation(std::move(orientation_)), vertices(vertices_), fingerprint({}) {
  initialize(vertices_);
}

void Simplex::initialize(Vertices &vertices_) {
  std::vector<IdType> ids = {};
  ids.reserve(vertices_.size());
  vertexIdLookup.reserve(vertices_.size());
  for (int i = 0; i < vertices_.size(); i++) {
    ids.push_back(vertices_[i]->getId());
    vertexIdLookup.insert({vertices_[i]->getId(), vertices_[i]});
  }
  fingerprint = Fingerprint(ids);
  computeEdges();
}

std::string Simplex::toString() const {
  std::stringstream ss;
  ss << "<";
  ss << std::to_string(getOrientation()->getK());
  ss << "-Simplex (";
  for (const auto &v : vertices) {
    ss << v->toString() << "→";
  }
  ss << vertices[0]->toString() << ")>";
  return ss.str();
}

/// Computes the volume of the simplex, \f$ V_{\sigma} \f$
///
double Simplex::getVolume() const {
  return 0;
}

const std::vector<std::shared_ptr<Simplex> > Simplex::getHinges() const {
  return {};
}

const double Simplex::getDeficitAngle() const {
  return 0;
}

const double Simplex::computeDihedralAngles() const {
  return 0.;
};

[[nodiscard]] SimplexOrientationPtr Simplex::getOrientation() const noexcept {
  return orientation;
}

[[nodiscard]] Vertices Simplex::getVertices() const noexcept { return vertices; };

[[nodiscard]] std::size_t Simplex::size() const noexcept {
  return vertices.size();
}

[[nodiscard]] bool Simplex::isTimelike() const {
  for (const auto &edge : edges) {
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

void Simplex::addCoface(const std::shared_ptr<Simplex> &simplex) {
  cofaces.insert(simplex);
}

[[nodiscard]] bool Simplex::hasCoface(const std::shared_ptr<Simplex> &simplex) const {
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

/// @returns Edges in traversal order (the order of input vertices).
[[nodiscard]] Edges Simplex::getEdges() const {
  return edges;
}

std::vector<std::tuple<IdType, IdType> > Simplex::moveEdges(const VertexPtr &from, const VertexPtr &to) {
  std::vector<std::tuple<IdType, IdType> > oldEdges = {};
  oldEdges.reserve(edges.size());
  for (const auto &edge : edges) {
    if (edge->getSourceId() == from->getId()) {
      oldEdges.push_back({edge->getSourceId(), edge->getTargetId()});
      edge->replaceSourceVertex(to->getId());
    }
    if (edge->getTargetId() == from->getId()) {
      oldEdges.push_back({edge->getSourceId(), edge->getTargetId()});
      edge->replaceTargetVertex(to->getId());
    }
  }
  return oldEdges;
}

void Simplex::addEdge(const EdgePtr &edge, bool addToCofaces) {
  edges.push_back(edge);
  if (addToCofaces) {
    for (const auto &coface : cofaces) {
      coface->addEdge(edge);
    }
  }
}

using RemoveEdgeByPtr = bool (Simplex::*)(const EdgePtr &);

bool Simplex::removeEdge(const EdgeKey &key) {
  const auto tempEdge = std::make_shared<Edge>(key.first, key.second);
  return removeEdge(tempEdge);
}

bool Simplex::removeEdge(const EdgePtr &edge) {
  for (int i = 0; i < edges.size(); ++i) {
    if (edges[i] == edge) {
      edges.erase(edges.begin() + i);
      CLOG(DEBUG_LEVEL, "Cascading removeEdge(", edge->toString(), ")");
      return cascade(static_cast<RemoveEdgeByPtr>(&Simplex::removeEdge), true, edge);
    }
  }
  return false;
}

EdgeIdSet Simplex::getEdgesExternalTo(const VertexPtr &vertex) {
  EdgeIdSet externalEdges{};
  for (const auto &edge : vertex->getEdges()) {
    if (!vertexIdLookup.contains(edge->getSourceId()) ||
      !vertexIdLookup.contains(edge->getTargetId())) {
      externalEdges.insert(edge->getKey());
    }
  }
  return externalEdges;
}

bool Simplex::removeVertex(const VertexPtr &vertex) {
  std::vector<IdType> vertexIds{};
  vertexIds.reserve(vertices.size() - 1);
  bool erased = false;
  for (int i = 0; i < vertices.size(); ++i) {
    if (vertices[i] == vertex) {
      vertices.erase(vertices.begin() + i);
      erased = true;
    } else {
      vertexIds.push_back(vertices[i]->getId());
    }
  }
  if (!erased) return false;
  fingerprint.refreshFingerprint(vertexIds);
  CLOG(DEBUG_LEVEL, "Cascading removeVertex(", vertex->toString(), ")");
  return cascade(&Simplex::removeVertex, false, vertex);
}

[[nodiscard]]
std::optional<Vertices>
Simplex::getVerticesWithParityTo(const std::shared_ptr<Simplex> &other) const {
  const auto &mine = vertices;
  const auto &theirs = other->getVertices();

  const std::size_t n = mine.size();
  if (n != theirs.size()) {
    throw std::runtime_error("You can only compare simplices of the same size!");
  }
  CLOG(DEBUG_LEVEL, "isTimelike");
  if (isTimelike() && !other->isTimelike() || !isTimelike() && other->isTimelike()) {
    throw std::runtime_error("Can't establish parity when one face is timelike and the other is not!");
  }
  if (n == 0) return std::nullopt;
  CLOG(DEBUG_LEVEL, "isTimelike");
  if (n == 1) {
    if (mine[0]->getTime() != theirs[0]->getTime()) return std::nullopt;
    return mine; // already aligned
  }

  auto try_alignment =
      [&](std::size_t start,
          bool reversed)
    -> std::optional<Vertices> {
    CLOG(DEBUG_LEVEL, "Trying alignment...");
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
        CLOG(DEBUG_LEVEL, "Alignment failed.");
        return std::nullopt; // mismatch, this alignment fails
      }

      CLOG(DEBUG_LEVEL, "result.push_back()...");
      result.push_back(mine[idx]);
    }
    CLOG(DEBUG_LEVEL, "Alignment finished.");
    return result; // success
  };

  // Try all starting positions where times match theirs[0]
  CLOG(DEBUG_LEVEL, "comparing times...");
  for (std::size_t i = 0; i < n; ++i) {
    if (mine[i]->getTime() != theirs[0]->getTime()) continue;

    // 1. Try same orientation
    CLOG(DEBUG_LEVEL, "Trying alignment");
    if (auto aligned = try_alignment(i, /*reversed=*/false)) return aligned;

    // 2. Try reversed orientation
    CLOG(DEBUG_LEVEL, "Trying reversed alignment");
    if (auto aligned_rev = try_alignment(i, /*reversed=*/true)) return aligned_rev;
  }

  // No alignment found
  CLOG(DEBUG_LEVEL, "Alignment finished.");
  return std::nullopt;
}

void Simplex::computeEdges() {
  auto nEdges = computeNumberOfEdges(getOrientation()->getK());

  if (!edges.empty()) {
    edges = Edges{};
  }

  edges.reserve(nEdges);

  VertexPtr origin = nullptr;

  // The direction of the edges can be either way; source -> target or target -> source. Just ensure we move across
  // the vertices in the correct order
  for (const auto &v : getVertices()) {
    for (const auto &e : v->getInEdges()) {
      if (hasVertex(e->getSourceId()) && hasVertex(e->getTargetId())) {
        edges.push_back(e);
      }
    }
  }
}

[[nodiscard]] bool Simplex::hasEdge(const EdgePtr &edge) {
  return hasEdge(edge->getSourceId(), edge->getTargetId());
}

[[nodiscard]] bool Simplex::hasEdge(const IdType vertexAId, const IdType vertexBId) {
  for (const auto &edge : edges) {
    if (edge->getSourceId() == vertexAId && edge->getTargetId() == vertexBId) {
      return true;
    }
    if (edge->getTargetId() == vertexAId && edge->getSourceId() == vertexBId) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] bool Simplex::hasVertex(const VertexPtr &vertex) const {
  return vertexIdLookup.contains(vertex->getId());
}

int8_t Simplex::checkParity(std::shared_ptr<Simplex> &other) {
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
      CLOG(WARN_LEVEL, "Other face contains ", otherId, " but this face does not!");
      for (auto &v : otherIds) {
        CLOG(WARN_LEVEL, "Other face contains ", otherId);
      }
      for (auto &v : vertices) {
        CLOG(WARN_LEVEL, "This face contains ", v->getId());
      }
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

  int N = K;
  int transpositionsMod2 = (N - cycles) & 1;
  return transpositionsMod2 ? -1 : +1;
}

[[nodiscard]] std::unordered_set<std::shared_ptr<Simplex>, SimplexHash, SimplexEq>
Simplex::getCofaces() const noexcept {
  return cofaces;
}

bool Simplex::operator==(const Simplex &other) const noexcept {
  return fingerprint.fingerprint() == other.fingerprint.fingerprint();
}

bool Simplex::isCausallyAvailable() noexcept {
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

std::unordered_set<SimplexOrientationPtr> Simplex::getGluableFaceOrientations() {
  auto allowedOrientations = std::unordered_set<SimplexOrientationPtr>{};
  for (const auto &face : getFacets()) {
    if (face->getCofaces().size() < 2) {
      allowedOrientations.insert(face->getOrientation());
    }
  }
  CLOG(DEBUG_LEVEL,
       "Found ",
       std::to_string(allowedOrientations.size()),
       " allowed orientations for simplex ",
       toString());
  return allowedOrientations;
}

bool Simplex::operator==(const std::shared_ptr<Simplex> &other) const noexcept {
  return fingerprint.fingerprint() == other->fingerprint.fingerprint();
}

bool Simplex::replaceEdge(const EdgePtr &oldEdge, const EdgePtr &newEdge) {
  for (int i = 0; i < edges.size(); ++i) {
    if (edges[i] == oldEdge) {
      edges[i] = newEdge;
      CLOG(DEBUG_LEVEL, "Cascading replaceEdge");
      return cascade(&Simplex::replaceEdge, true, oldEdge, newEdge);
    }
  }
  return false;
}

bool Simplex::replaceVertex(const VertexPtr &oldVertex, const VertexPtr &newVertex) {
  if (!hasVertex(oldVertex)) {
    CLOG(WARN_LEVEL, "Not replacing ", oldVertex->getId(), " with ", newVertex->getId(), " because the simplex does not have the old vertex!");
    return false;
  }
  if (hasVertex(newVertex)) {
    CLOG(WARN_LEVEL, "Not replacing ", oldVertex->getId(), " with ", newVertex->getId(), " because the simplex already has the new vertex!");
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
  fingerprint.refreshFingerprint(vertexIds);
  CLOG(DEBUG_LEVEL, "Cascading replaceVertex(", std::to_string(oldVertex->getId()), ", ", std::to_string(newVertex->getId()), ")");
  return cascade(&Simplex::replaceVertex, false, oldVertex, newVertex);
  throw std::runtime_error("This simplex does not contain the vertex you're trying to replace!");
}

VertexIdMap Simplex::getVertexIdLookup() const noexcept {
  return vertexIdLookup;
}

template<typename Method, typename... Args>
bool Simplex::cascade(Method method, bool shallow, Args &&... args) {
  std::deque<std::shared_ptr<Simplex> > simplicesToUpdate;
  std::unordered_set<std::shared_ptr<Simplex>, SimplexHash, SimplexEq> seen;

  auto enqueue_if_new = [&](const std::shared_ptr<Simplex> &s) {
    if (!seen.contains(s)) {
      simplicesToUpdate.push_back(s);
    }
  };

  // --- Cascading to cofaces ---
  if (!cofaces.empty()) {
    simplicesToUpdate.insert(simplicesToUpdate.end(),
                             cofaces.begin(),
                             cofaces.end());
    while (!simplicesToUpdate.empty()) {
      const auto coface = simplicesToUpdate.front(); // copy the shared_ptr
      simplicesToUpdate.pop_front();

      if (!seen.insert(coface).second) continue;

      // Call the member function on this coface
      if ((coface.get()->*method)(std::forward<Args>(args)...)) {
        for (const auto &nextCoface : coface->getCofaces()) {
          // TODO: need exclusively local methods, this is recursing.
          if (!shallow) enqueue_if_new(nextCoface);
        }
      }
    }
  }

  // --- Cascading to facets ---
  auto facets_ = getFacets();
  if (!facets_.empty()) {
    simplicesToUpdate.clear();
    simplicesToUpdate.insert(simplicesToUpdate.end(),
                             facets_.begin(),
                             facets_.end());
    while (!simplicesToUpdate.empty()) {
      const auto facet = simplicesToUpdate.front(); // copy, NOT reference
      simplicesToUpdate.pop_front();

      if (!seen.insert(facet).second) continue;

      // TODO: need exclusively local methods, this is recursing.
      if ((facet.get()->*method)(std::forward<Args>(args)...)) {
        for (const auto &nextFacet : facet->getFacets()) {
          if (!shallow) enqueue_if_new(nextFacet);
        }
      }
    }
  }
  return true;
}
}
