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

#include "mesh/Vertex.h"
#include "mesh/Simplex.h"
#include "mesh/ForwardDeclarations.h"
#include "mesh/SimplexOrientation.h"
#include "spacetime/Spacetime.h"
#include "Logger.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>
#include <stdexcept>
#include <string>
#include <unordered_map>

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::mesh {
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;

// Tripwire: catch dereferences of stale Simplex pointers.  Storage is
// stable, so reads from a logically-removed simplex won't fault — they'd
// just see empty child vectors and proceed silently.  This macro turns
// that silent failure mode into a loud abort under TESSERA_ASSERTIONS,
// while costing nothing in release builds.  Used at the top of the
// hot getters that callers might invoke through a cached SimplexPtr.
#ifdef TESSERA_ASSERTIONS
  #define TESSERA_TRIPWIRE_LIVE(method_name)                              \
    do {                                                                  \
      if (isStale()) {                                                    \
        CLOG(CRITICAL_LEVEL, "Stale Simplex* dereferenced via " method_name \
                             " — caller is holding a pointer to a "      \
                             "simplex that was already removed.");        \
        std::abort();                                                     \
      }                                                                   \
    } while (0)
#else
  #define TESSERA_TRIPWIRE_LIVE(method_name) ((void)0)
#endif

bool Simplex::hasFacets() const {
  TESSERA_TRIPWIRE_LIVE("hasFacets");
  return !facets.empty();
}

#ifdef TESSERA_ASSERTIONS
class SimplexCorruptionDetector : public CorruptionDetector<SimplexPtr, SimplexPtrHash, SimplexPtrEq> {
};
#endif

const std::vector<SimplexPtr> &Simplex::getFacets() {
  TESSERA_TRIPWIRE_LIVE("getFacets");
#if TESSERA_ASSERTIONS
  if (getVertices().empty()) throw std::runtime_error("Simplex is empty");
#endif
  if (getVertices().size() == 1) {
#if TESSERA_ASSERTIONS
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

    // Use this directly — no hash lookup needed.
    SimplexPtr coface = this;

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
      if (coface != nullptr && !facet->hasCoface(coface)) {
        facet->addCoface(coface);
      }
      facets.push_back(facet);
    }
  }
#if TESSERA_ASSERTIONS
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
    edges(std::move(edges_)),
    fingerprint({0}) {
#if TESSERA_ASSERTIONS
  if (vertices_.empty()) throw std::runtime_error("Simplex is empty");
#endif
}

Simplex::Simplex(
  Spacetime *spacetime_,
  const VertexPtrs &vertices_,
  Edges edges_,
  const SimplexOrientation &orientation_
) : spacetime(spacetime_), orientation(orientation_), vertices(vertices_), edges(std::move(edges_)),
    fingerprint() {
  for (const auto &v : vertices_) {
    fingerprint.addId(v->getId());
  }
  fingerprint.refresh();
#if TESSERA_ASSERTIONS
  if (vertices_.empty()) throw std::runtime_error("Simplex is empty");
#endif
}

Simplex* Simplex::create(Spacetime *spacetime_, const VertexPtrs &vertices_, const Edges &edges_) {
#if TESSERA_ASSERTIONS
  if (vertices_.empty()) throw std::runtime_error("Simplex is empty");
#endif
  Simplex* simplex = new Simplex(spacetime_, vertices_, edges_);
  if (!simplex->initialized) {
    simplex->initialize(simplex);
  }
  // registerToVertices() is called during initialize(); addSimplex() deduplicates.
  return simplex;
}

bool Simplex::isInitialized() const noexcept { return initialized; }

void Simplex::releaseChildren() noexcept {
#ifdef TESSERA_ASSERTIONS
  if (!isStale()) {
    CLOG(CRITICAL_LEVEL, "releaseChildren() called on a still-registered "
                         "simplex; caller must mark it stale first.");
    std::abort();
  }
#endif
  // swap-with-empty deallocates the underlying buffer (clear() alone would
  // keep capacity).  Order doesn't matter — none of these refer to each
  // other.
  VertexPtrs().swap(vertices);
  Edges().swap(edges);
  Simplices().swap(facets);
  Simplices().swap(cofaces);
}

Simplex* Simplex::create(Spacetime *spacetime_,
                           const VertexPtrs &vertices_,
                           const Edges &edges_,
                           const SimplexOrientation &orientation_) {
#if TESSERA_ASSERTIONS
  if (vertices_.empty()) throw std::runtime_error("Simplex is empty");
#endif
  Simplex* simplex = new Simplex(spacetime_, vertices_, edges_, orientation_);
  simplex->initialize(simplex);
  return simplex;
}

void Simplex::initialize(Simplex* simplex) {
#ifdef TESSERA_ASSERTIONS
  if (simplex->initialized) {
    CLOG(DEBUG_LEVEL, "You attempted to re-initialize a simplex! Behavior is undefined.");
    std::abort();
  }
#endif
  std::vector<IdType> ids = {};
  ids.reserve(vertices.size());
  for (const auto &v : vertices) {
    ti = std::min(ti, v->getTime());
    tf = std::max(tf, v->getTime());
    ids.push_back(v->getId());
  }
  fingerprint.setIds(ids);
  _isSpatial = ti == tf;

  // We have to register AFTER the fingerprint is set:
  registerToVertices(simplex);
  initialized = true;

  if (ti != tf) {
    CLOG(INFO_LEVEL, "ti != tf: ", std::to_string(ti), " != ", std::to_string(tf), " for ", toString());
  } else {
    CLOG(INFO_LEVEL, "ti == tf: ", std::to_string(ti), " == ", std::to_string(tf), " for ", toString());
  }
}

// getTi(), getTf() inlined in Simplex.h

void Simplex::registerToVertices(Simplex* simplex) {
  for (const auto &owner : simplex->getVertices()) {
    owner->addSimplex(simplex);
  }
}

#ifdef TESSERA_VERBOSE
std::string Simplex::toString() const noexcept {
  std::stringstream sigmaLabel;
  sigmaLabel << std::to_string(getOrientation().getK()) << "-";
  sigmaLabel << "\\sigma";

  std::stringstream orientationStr;
  orientationStr << "^{(" << std::to_string(std::get<0>(getOrientation().numeric())) << "/";
  orientationStr << std::to_string(std::get<1>(getOrientation().numeric())) << ")}";

  std::string fp = std::to_string(fingerprint.fingerprint());
  std::string fpShort = fp.size() >= 6
      ? fp.substr(0, 3) + fp.substr(fp.size() - 3)
      : fp;
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

// getOrientation() inlined in Simplex.h

[[nodiscard]] const VertexPtrs &Simplex::getVertices() const noexcept {
  TESSERA_TRIPWIRE_LIVE("getVertices");
  return vertices;
}

// isSpatial() / isTimelike() inlined in Simplex.h (uses cached _isSpatial)

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

void Simplex::addCoface(SimplexPtr coface) {
#if TESSERA_ASSERTIONS
  if (coface == nullptr) {
    CLOG(DEBUG_LEVEL, "Coface was null");
    std::abort();
  }
  if (!coface->isCofaceTo(this)) {
    CLOG(DEBUG_LEVEL, coface->toString(), " is not a coface of ", toString());
    throw std::runtime_error("You attempted to add a coface to a facet for which it is not a coface!");
  }
  if (hasCoface(coface)) {
    CLOG(DEBUG_LEVEL, "You attempted to add a duplicate coface: ", coface->toString(), " to simplex ", toString());
    std::abort();
  }
#endif
  // Cofaces are 0-2 elements; linear duplicate check is faster than hash table
  if (!hasCoface(coface)) {
    cofaces.push_back(coface);
  }
}

void Simplex::removeCoface(SimplexPtr coface) {
  auto fp = coface->fingerprint.fingerprint();
  for (auto it = cofaces.begin(); it != cofaces.end(); ++it) {
    if ((*it)->fingerprint.fingerprint() == fp) {
      // Swap-and-pop for O(1) removal
      *it = cofaces.back();
      cofaces.pop_back();
      return;
    }
  }
}

[[nodiscard]] bool Simplex::hasCoface(SimplexPtr coface) const {
  TESSERA_TRIPWIRE_LIVE("hasCoface");
  auto fp = coface->fingerprint.fingerprint();
  for (const auto &c : cofaces) {
    if (c->fingerprint.fingerprint() == fp) return true;
  }
  return false;
}

[[nodiscard]] bool Simplex::hasVertex(const VertexPtr &vertex) const {
  TESSERA_TRIPWIRE_LIVE("hasVertex");
  const auto id = vertex->getId();
  for (const auto &v : vertices)
    if (v->getId() == id) return true;
  return false;
}

[[nodiscard]] bool Simplex::hasEdgeContaining(const IdType vertexId) const {
  for (const auto &e : getEdges()) {
    if (e->getSource()->getId() == vertexId) return true;
    if (e->getTarget()->getId() == vertexId) return true;
  }
  return false;
}

void Simplex::validate() const {
#ifdef TESSERA_ASSERTIONS
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

[[nodiscard]] const Edges &Simplex::getEdges() const {
  TESSERA_TRIPWIRE_LIVE("getEdges");
  return edges;
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
  if (!hasVertex(vertexA) || !hasVertex(vertexB)) return false;
  auto aId = vertexA->getId();
  auto bId = vertexB->getId();
  for (const auto &e : edges) {
    if ((e->getSource()->getId() == aId && e->getTarget()->getId() == bId) ||
        (e->getSource()->getId() == bId && e->getTarget()->getId() == aId))
      return true;
  }
  return false;
}

[[nodiscard]] const Simplices &
Simplex::getCofaces() const noexcept {
  TESSERA_TRIPWIRE_LIVE("getCofaces");
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

bool Simplex::operator==(const Simplex* other) const noexcept {
  return fingerprint.fingerprint() == other->fingerprint.fingerprint();
}

std::uint64_t Simplex::hash() const noexcept {
  return fingerprint.fingerprint();
}

bool Simplex::isBoundary() const noexcept {
  return cofaces.size() < 2;
}

bool Simplex::hasBoundaryFacet() {
  for (const auto &face : getFacets()) {
    if (face->isBoundary()) return true;
  }
  return false;
}

std::size_t Simplex::maxKPlusOneCofaces() const {
  return getNumberOfFaces(getOrientation().getK());
}

// size() inlined in Simplex.h

bool Simplex::replaceVertex(const VertexPtr &oldVertex, const VertexPtr &newVertex) {
  // TODO: Probably make this cascade, but we should just go to the Vertex for things to cascade to.
  if (hasVertex(newVertex)) {
#if TESSERA_ASSERTIONS
    validate();
#endif
    return false;
  }
  auto oldId = oldVertex->getId();
  std::size_t oldIndex = vertices.size(); // sentinel
  for (std::size_t i = 0; i < vertices.size(); ++i) {
    if (vertices[i]->getId() == oldId) { oldIndex = i; break; }
  }
  if (oldIndex == vertices.size()) return false;

  vertices[oldIndex] = newVertex;

  fingerprint.removeId(oldId);
  fingerprint.addId(newVertex->getId());

  // Clear cached facets/cofaces — they depend on the vertex set which just changed.
  facets.clear();
  cofaces.clear();

  return true;
}

VertexIdMap Simplex::getVertexIdLookup() const noexcept {
  VertexIdMap lookup{};
  for (const auto &v : vertices) {
    lookup.emplace(v->getId(), v);
  }
  return lookup;
}

bool Simplex::removeEdge(const EdgePtr &edge) {
  auto fp = edge->fingerprint.fingerprint();
  for (auto it = edges.begin(); it != edges.end(); ++it) {
    if ((*it)->fingerprint.fingerprint() == fp) {
      *it = edges.back();
      edges.pop_back();
      // Keep the Edge → Simplex index in sync. Without this the
      // edge's `simplices_` still claims this simplex as a member and
      // the next `Vertex::removeOutEdge` would dispatch into a
      // simplex that no longer contains the edge.
      edge->unregisterSimplex(this);
      return true;
    }
  }
  return false;
}

bool Simplex::addEdge(const EdgePtr &edge) {
  auto fp = edge->fingerprint.fingerprint();
  for (const auto &e : edges) {
    if (e->fingerprint.fingerprint() == fp) return false;
  }
  edges.push_back(edge);
  edge->registerSimplex(this);
  return true;
}

// updateVertexId / swapVertexIds are intentional no-ops (inlined in
// Simplex.h). The Simplex stores VertexPtrs and reads IDs through
// them; when Spacetime::swapVertexLabels rewrites a Vertex's ID, the
// Simplex sees the new ID automatically on its next ``getId()``.

bool Simplex::hasStoredFacet(const SimplexPtr &facet) const {
  if (facets.empty()) return false;
  for (const auto &f : facets) {
    if (f == facet) return true;
  }
  return false;
}

std::pair<SimplexPtr, Simplices> Simplex::cone(VertexPtr vertex) {
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
    if (foliation == Foliation::PREFERRED && !cofaces.empty()) {
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
      // Spacelike edge (same time slice): ℓ² = a
      newEdges.push_back(spacetime->createEdge(existing, vertex, spacetime->getA()));
    } else {
      // Timelike edge (different time slices): ℓ² = -α·a
      newEdges.push_back(spacetime->createEdge(existing, vertex, -(spacetime->getAlpha() * spacetime->getA())));
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

// =====================================================================
// Geometry
// =====================================================================

double Simplex::determinant(const std::vector<double> &M, int n) {
    if (n == 1) return M[0];
    if (n == 2) return M[0] * M[3] - M[1] * M[2];
    std::vector<double> A(M);
    double det = 1.0;
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        double maxVal = std::abs(A[col * n + col]);
        for (int row = col + 1; row < n; ++row) {
            double val = std::abs(A[row * n + col]);
            if (val > maxVal) { maxVal = val; pivot = row; }
        }
        if (maxVal < 1e-15) return 0.0;
        if (pivot != col) {
            for (int j = 0; j < n; ++j)
                std::swap(A[col * n + j], A[pivot * n + j]);
            det = -det;
        }
        det *= A[col * n + col];
        for (int row = col + 1; row < n; ++row) {
            double factor = A[row * n + col] / A[col * n + col];
            for (int j = col + 1; j < n; ++j)
                A[row * n + j] -= factor * A[col * n + j];
        }
    }
    return det;
}

std::vector<double> Simplex::cofactorMatrix(
    const std::vector<double> &M, int n) {
    std::vector<double> C(n * n, 0.0);
    if (n == 1) { C[0] = 1.0; return C; }
    std::vector<double> sub((n - 1) * (n - 1));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int si = 0;
            for (int r = 0; r < n; ++r) {
                if (r == i) continue;
                int sj = 0;
                for (int c = 0; c < n; ++c) {
                    if (c == j) continue;
                    sub[si * (n - 1) + sj] = M[r * n + c];
                    sj++;
                }
                si++;
            }
            double sign = ((i + j) % 2 == 0) ? 1.0 : -1.0;
            C[i * n + j] = sign * determinant(sub, n - 1);
        }
    }
    return C;
}

std::vector<double> Simplex::gramMatrix(bool wickRotate) const {
    int dPlus1 = static_cast<int>(vertices.size());
    int d = dPlus1 - 1;
    if (d < 1) return {};

    // Squared-distance lookup. Honor the signed l^2 so the Lorentzian sign of
    // timelike edges survives into G; wickRotate takes |l^2| (Euclidean/CDT).
    std::unordered_map<std::uint64_t, double> sqMap;
    for (const auto &e : edges) {
        auto fp = Fingerprint::mix64(e->getSource()->getId()) ^
                  Fingerprint::mix64(e->getTarget()->getId());
        sqMap[fp] = wickRotate ? std::abs(e->getSquaredLength())
                               : e->getSquaredLength();
    }
    auto getSq = [&](int i, int j) -> double {
        if (i == j) return 0.0;
        auto fp = Fingerprint::mix64(vertices[i]->getId()) ^
                  Fingerprint::mix64(vertices[j]->getId());
        auto it = sqMap.find(fp);
        return it != sqMap.end() ? it->second : 0.0;
    };

    std::vector<double> G(d * d, 0.0);
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j)
            G[i * d + j] = 0.5 * (getSq(0, i + 1) + getSq(0, j + 1)
                                   - getSq(i + 1, j + 1));
    return G;
}

std::vector<double> Simplex::cayleyMengerMatrix(bool wickRotate) const {
    int dPlus1 = static_cast<int>(vertices.size());
    if (dPlus1 < 1) return {};

    // Squared-distance lookup; signed by default, |l^2| under wickRotate.
    std::unordered_map<std::uint64_t, double> sqMap;
    for (const auto &e : edges) {
        auto fp = Fingerprint::mix64(e->getSource()->getId()) ^
                  Fingerprint::mix64(e->getTarget()->getId());
        sqMap[fp] = wickRotate ? std::abs(e->getSquaredLength())
                               : e->getSquaredLength();
    }
    auto getSq = [&](int i, int j) -> double {
        if (i == j) return 0.0;
        auto fp = Fingerprint::mix64(vertices[i]->getId()) ^
                  Fingerprint::mix64(vertices[j]->getId());
        auto it = sqMap.find(fp);
        return it != sqMap.end() ? it->second : 0.0;
    };

    // Bordered matrix: zero corner, a border of ones, squared distances inside.
    int n = dPlus1 + 1;
    std::vector<double> B(n * n, 0.0);
    for (int k = 1; k < n; ++k) { B[k] = 1.0; B[k * n] = 1.0; }
    for (int i = 0; i < dPlus1; ++i)
        for (int j = 0; j < dPlus1; ++j)
            B[(i + 1) * n + (j + 1)] = getSq(i, j);
    return B;
}

double Simplex::dihedralAngle(SimplexPtr hinge, bool wickRotate) const {
    int dPlus1 = static_cast<int>(vertices.size());

    // Find the two vertices in this simplex but not in the hinge.
    auto hingeVerts = hinge->getVertices();
    std::vector<int> opposite;
    for (int k = 0; k < dPlus1; ++k) {
        bool inHinge = false;
        for (const auto &hv : hingeVerts)
            if (hv->getId() == vertices[k]->getId()) { inHinge = true; break; }
        if (!inHinge) opposite.push_back(k);
    }
    if (opposite.size() != 2) return 0.0;
    int vi = opposite[0], vj = opposite[1];

    // Cayley-Menger bordered matrix; its cofactors give the dihedral angle.
    int n = dPlus1 + 1;
    auto B = cayleyMengerMatrix(wickRotate);
    auto cof = cofactorMatrix(B, n);
    int bi = vi + 1, bj = vj + 1;
    double Cij = cof[bi * n + bj];
    double Cii = cof[bi * n + bi];
    double Cjj = cof[bj * n + bj];

    double denom = std::sqrt(std::abs(Cii * Cjj));
    if (denom < 1e-15) return 0.0;
    // The diagonal Cayley–Menger cofactors C_ii, C_jj share the
    // dimension-parity sign (-1)^d: positive in even dimension, negative in odd
    // (e.g. -3 for a unit tetrahedron). Taking |C_ii·C_jj| under the sqrt drops
    // that sign, so the normalization must reapply it — otherwise
    // cos θ = -C_ij / sqrt(C_ii·C_jj) collapses to its supplement (π - θ) for
    // odd-dimensional simplices. In even dimension C_ii > 0 and this is a no-op.
    if (Cii < 0.0) denom = -denom;
    double cosTheta = std::clamp(-Cij / denom, -1.0, 1.0);
    return std::acos(cosTheta);
}

double Simplex::deficitAngle() const {
    if (!spacetime) return 2.0 * std::numbers::pi;
    int d = spacetime->getMetric()->getSignature()->getDimensions();
    int topSize = d + 1;

    double sum = 0.0;
    if (vertices.empty()) return 2.0 * std::numbers::pi;

    for (const auto &sigma : vertices[0]->getSimplices()) {
        if (static_cast<int>(sigma->size()) != topSize) continue;
        bool containsAll = true;
        for (std::size_t i = 1; i < vertices.size(); ++i)
            if (!sigma->hasVertex(vertices[i])) { containsAll = false; break; }
        if (containsAll)
            // Deficit angles drive the CDT/Regge action, which is defined on
            // the Wick-rotated (Euclidean) geometry — request |l^2| explicitly.
            sum += sigma->dihedralAngle(const_cast<Simplex*>(this), /*wickRotate=*/true);
    }
    return 2.0 * std::numbers::pi - sum;
}

std::complex<double> Simplex::lorentzianDihedralAngle(SimplexPtr hinge) const {
    const int dPlus1 = static_cast<int>(vertices.size());
    // The two vertices of this simplex not in the hinge.
    const auto hingeVerts = hinge->getVertices();
    std::vector<int> opposite;
    for (int k = 0; k < dPlus1; ++k) {
        bool inHinge = false;
        for (const auto &hv : hingeVerts)
            if (hv->getId() == vertices[k]->getId()) { inHinge = true; break; }
        if (!inHinge) opposite.push_back(k);
    }
    if (opposite.size() != 2) return {0.0, 0.0};
    const int vi = opposite[0], vj = opposite[1];

    // Signed (non-Wick) Cayley-Menger cofactors → the dihedral cosine ratio r,
    // UN-clamped. std::acos on its complex extension returns the ordinary angle
    // for |r| <= 1 and a boost (complex) for |r| > 1 — see the header.
    const int n = dPlus1 + 1;
    const auto B = cayleyMengerMatrix(/*wickRotate=*/false);
    const auto cof = cofactorMatrix(B, n);
    const int bi = vi + 1, bj = vj + 1;
    const double Cij = cof[static_cast<std::size_t>(bi) * n + bj];
    const double Cii = cof[static_cast<std::size_t>(bi) * n + bi];
    const double Cjj = cof[static_cast<std::size_t>(bj) * n + bj];
    double denom = std::sqrt(std::abs(Cii * Cjj));
    if (denom < 1e-15) return {0.0, 0.0};
    if (Cii < 0.0) denom = -denom;  // (-1)^d diagonal-sign fix (see dihedralAngle)
    const double r = -Cij / denom;
    return std::acos(std::complex<double>(r, 0.0));
}

std::complex<double> Simplex::lorentzianDeficitAngle() const {
    using cd = std::complex<double>;
    const cd twoPi(2.0 * std::numbers::pi, 0.0);
    if (!spacetime || vertices.empty()) return twoPi;
    const int topSize =
        spacetime->getMetric()->getSignature()->getDimensions() + 1;
    cd sum(0.0, 0.0);
    for (const auto &sigma : vertices[0]->getSimplices()) {
        if (static_cast<int>(sigma->size()) != topSize) continue;
        bool containsAll = true;
        for (std::size_t i = 1; i < vertices.size(); ++i)
            if (!sigma->hasVertex(vertices[i])) { containsAll = false; break; }
        if (containsAll)
            sum += sigma->lorentzianDihedralAngle(const_cast<Simplex *>(this));
    }
    return twoPi - sum;
}

double Simplex::area(bool wickRotate) const {
    if (edges.size() < 3) return 0.0;
    auto sq = [&](std::size_t k) -> double {
        return wickRotate ? std::abs(edges[k]->getSquaredLength())
                          : edges[k]->getSquaredLength();
    };
    double a2 = sq(0);
    double b2 = sq(1);
    double c2 = sq(2);
    double val = 2.0 * (a2 * b2 + b2 * c2 + c2 * a2)
                 - (a2 * a2 + b2 * b2 + c2 * c2);
    if (val <= 0.0) return 0.0;
    return std::sqrt(val) / 4.0;
}

double Simplex::volume() const {
    int d = static_cast<int>(vertices.size()) - 1;
    if (d < 1) return 0.0;

    // Honest, signature-respecting Gram matrix: timelike edges keep l^2 < 0,
    // so det(G) can be negative for a Lorentzian cell.
    const std::vector<double> G = gramMatrix(/*wickRotate=*/false);
    if (static_cast<int>(G.size()) != d * d) return 0.0;

    const double detG = determinant(G, d);
    double factorial = 1.0;
    for (int i = 2; i <= d; ++i) factorial *= static_cast<double>(i);

    // Signed d-content: sqrt(det G)/d! with the sign of det(G) carried out so
    // the result stays real and records the signature instead of |det G|.
    const double sign = (detG < 0.0) ? -1.0 : 1.0;
    return sign * std::sqrt(std::abs(detG)) / factorial;
}

void Simplex::assertSpacelikeAdmissible(double tol) const {
    const int n = static_cast<int>(size());  // vertices = d + 1
    if (n < 2) return;                        // trivially admissible
    const int d = n - 1;

    // Skip simplices that contain any null/timelike (worldline) edge: their
    // admissibility is Lorentzian, not the spacelike triangle inequalities. The
    // Cayley-Menger bordered matrix carries the squared edge lengths in its
    // lower-right (d+1)x(d+1) block, offset (1, 1) of the (d+2)x(d+2) array.
    const std::vector<double> cm = cayleyMengerMatrix(/*wickRotate=*/false);
    const int mm = n + 1;  // CM is (d+2) x (d+2)
    for (int a = 0; a < n; ++a) {
        for (int b = a + 1; b < n; ++b) {
            const double s = cm[static_cast<std::size_t>(1 + a) * mm + (1 + b)];
            if (s <= tol) return;  // null/timelike edge → not a spacelike cell
        }
    }

    // All edges spacelike: the Gram matrix must be positive-definite. Check via
    // Sylvester's criterion (every leading principal minor > 0) so the test
    // stays Eigen-free, reusing the existing determinant helper.
    const std::vector<double> g = gramMatrix(/*wickRotate=*/false);
    for (int k = 1; k <= d; ++k) {
        std::vector<double> sub(static_cast<std::size_t>(k) * k);
        for (int i = 0; i < k; ++i)
            for (int j = 0; j < k; ++j)
                sub[static_cast<std::size_t>(i) * k + j] =
                    g[static_cast<std::size_t>(i) * d + j];
        const double minor = determinant(sub, k);
        if (!(minor > tol)) {
            throw std::runtime_error(
                "Simplex::assertSpacelikeAdmissible: inadmissible spacelike "
                "simplex — Gram matrix is not positive-definite (leading minor "
                + std::to_string(k) + " = " + std::to_string(minor) +
                "); the spacelike triangle inequalities are violated. The metric "
                "is not silently repaired.");
        }
    }
}

namespace {

// Signed real square root: sign(x)·sqrt(|x|). The signed-content convention
// (matching Simplex::volume), so a timelike circumcentric height contributes
// negative content rather than throwing on a negative radicand.
double signedSqrt(double x) {
    return (x < 0.0) ? -std::sqrt(-x) : std::sqrt(x);
}

// Circumcenter (barycentric) + signed R² from the Gram matrix G (flat d×d,
// relative to vertex 0). Solves G β = ½·diag(G) Eigen-free via the adjugate
// (cofactorᵀ/det); λ_0 = 1−Σβ, λ_i = β_i; R² = Σ_i β_i·(½ G_ii).
void circumFromGram(const std::vector<double>& G, int d,
                    std::vector<double>& bary, double& r2) {
    bary.assign(static_cast<std::size_t>(d) + 1, 0.0);
    if (d <= 0) { bary[0] = 1.0; r2 = 0.0; return; }  // single vertex
    std::vector<double> halfDiag(d);
    for (int i = 0; i < d; ++i)
        halfDiag[i] = 0.5 * G[static_cast<std::size_t>(i) * d + i];
    const double detG = ::tessera::mesh::Simplex::determinant(G, d);
    const std::vector<double> cof =
        ::tessera::mesh::Simplex::cofactorMatrix(G, d);  // cof[r*d+c] = C_rc
    // β_i = Σ_j (G⁻¹)_ij·halfDiag_j, with (G⁻¹)_ij = adj_ij/det = C_ji/det.
    std::vector<double> beta(d, 0.0);
    double sum = 0.0;
    for (int i = 0; i < d; ++i) {
        double acc = 0.0;
        for (int j = 0; j < d; ++j)
            acc += cof[static_cast<std::size_t>(j) * d + i] * halfDiag[j];
        beta[i] = (detG != 0.0) ? acc / detG : 0.0;
        bary[static_cast<std::size_t>(i) + 1] = beta[i];
        sum += beta[i];
    }
    bary[0] = 1.0 - sum;
    r2 = 0.0;
    for (int i = 0; i < d; ++i) r2 += beta[i] * halfDiag[i];
}

// Sign (±1) of the circumcenter of coface `cf` at the vertex of `cf` not in its
// facet `s` — the side of the facet's hull on which c(cf) sits.
double oppositeVertexSign(const ::tessera::mesh::Simplex* cf,
                          const ::tessera::mesh::Simplex* s) {
    const auto& cfv = cf->getVertices();
    const auto& sv = s->getVertices();
    int oppIdx = -1;
    for (std::size_t i = 0; i < cfv.size(); ++i) {
        bool inS = false;
        for (const auto* w : sv)
            if (w->getId() == cfv[i]->getId()) { inS = true; break; }
        if (!inS) { oppIdx = static_cast<int>(i); break; }
    }
    if (oppIdx < 0) return 1.0;
    const std::vector<double> bary = cf->circumcenterBarycentric();
    return (bary[static_cast<std::size_t>(oppIdx)] < 0.0) ? -1.0 : 1.0;
}

// Recursive signed circumcentric dual content of `s` in an n-complex.
double dualVolRec(const ::tessera::mesh::Simplex* s, int n) {
    const int k = static_cast<int>(s->size()) - 1;
    if (k >= n) return 1.0;  // top cell: dual is a point (content 1)
    const double rk2 = s->circumradiusSquared();
    double acc = 0.0;
    for (const auto& cf : s->getCofaces()) {
        const double h =
            oppositeVertexSign(cf, s) * signedSqrt(cf->circumradiusSquared() - rk2);
        acc += h * dualVolRec(cf, n);
    }
    return acc / static_cast<double>(n - k);
}

}  // namespace

std::vector<double> Simplex::circumcenterBarycentric() const {
    const int d = static_cast<int>(size()) - 1;
    std::vector<double> bary;
    double r2 = 0.0;
    circumFromGram(gramMatrix(/*wickRotate=*/false), d, bary, r2);
    return bary;
}

double Simplex::circumradiusSquared() const {
    const int d = static_cast<int>(size()) - 1;
    std::vector<double> bary;
    double r2 = 0.0;
    circumFromGram(gramMatrix(/*wickRotate=*/false), d, bary, r2);
    return r2;
}

double Simplex::dualVolume() const {
    // Ambient top dimension: walk up cofaces to a top simplex (empty cofaces).
    const Simplex* top = this;
    while (!top->getCofaces().empty()) top = top->getCofaces()[0];
    const int n = static_cast<int>(top->size()) - 1;
    return dualVolRec(this, n);
}

double Simplex::hodgeStar() const {
    const double v = volume();
    if (v == 0.0) {
        throw std::runtime_error(
            "Simplex::hodgeStar: primal volume is zero (degenerate simplex)");
    }
    return dualVolume() / v;
}

}
