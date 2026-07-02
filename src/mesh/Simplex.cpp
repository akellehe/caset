// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "mesh/Vertex.h"
#include "mesh/Simplex.h"
#include "mesh/ForwardDeclarations.h"
#include "mesh/TemporalOrientation.h"
#include "spacetime/Spacetime.h"
#include "Logger.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

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
) : spacetime(spacetime_), orientation(TemporalOrientation::orientationOf(vertices_)), vertices(vertices_),
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
  const TemporalOrientation &orientation_
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
                           const TemporalOrientation &orientation_) {
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
                               : e->getSquaredLength().real();
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
                               : e->getSquaredLength().real();
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

std::vector<double> Simplex::cayleyMengerCanonical(
    bool wickRotate, std::unordered_map<std::uint64_t, int> &pos1) const {
    const int dPlus1 = static_cast<int>(vertices.size());
    pos1.clear();
    if (dPlus1 < 1) return {};

    // Canonical order: vertices sorted by ascending id (the reference orientation).
    std::vector<VertexPtr> sorted(vertices.begin(), vertices.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const VertexPtr a, const VertexPtr b) {
                  return a->getId() < b->getId();
              });
    for (int i = 0; i < dPlus1; ++i)
        pos1[sorted[static_cast<std::size_t>(i)]->getId()] = i + 1;  // border-offset

    std::unordered_map<std::uint64_t, double> sqMap;
    for (const auto &e : edges) {
        auto fp = Fingerprint::mix64(e->getSource()->getId()) ^
                  Fingerprint::mix64(e->getTarget()->getId());
        sqMap[fp] = wickRotate ? std::abs(e->getSquaredLength())
                               : e->getSquaredLength().real();
    }
    auto getSq = [&](int i, int j) -> double {
        if (i == j) return 0.0;
        auto fp = Fingerprint::mix64(sorted[static_cast<std::size_t>(i)]->getId()) ^
                  Fingerprint::mix64(sorted[static_cast<std::size_t>(j)]->getId());
        auto it = sqMap.find(fp);
        return it != sqMap.end() ? it->second : 0.0;
    };

    const int n = dPlus1 + 1;
    std::vector<double> B(static_cast<std::size_t>(n) * n, 0.0);
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
    //
    // Evaluate in the canonical (sorted-by-id) frame: the signed cofactor sign fix
    // below (Cii<0) is sensitive to the order the cell's vertices are stored in, so
    // a cell a Pachner move stored in causal order would otherwise yield a
    // different deficit than the same geometry built sorted — making the action
    // depend on build history. The canonical frame makes it a true invariant.
    const int n = dPlus1 + 1;
    std::unordered_map<std::uint64_t, int> pos1;
    const auto B = cayleyMengerCanonical(/*wickRotate=*/false, pos1);
    const auto cof = cofactorMatrix(B, n);
    int bi = pos1[vertices[vi]->getId()];
    int bj = pos1[vertices[vj]->getId()];
    // The Cii<0 sign fix below is asymmetric in (i,j); anchor it on the lower
    // canonical position so the result does not depend on which opposite vertex
    // the (stored) ordering happened to present first.
    if (bi > bj) std::swap(bi, bj);
    const double Cij = cof[static_cast<std::size_t>(bi) * n + bj];
    const double Cii = cof[static_cast<std::size_t>(bi) * n + bi];
    const double Cjj = cof[static_cast<std::size_t>(bj) * n + bj];
    const double D = std::sqrt(std::abs(Cii * Cjj));
    if (D < 1e-15) return {0.0, 0.0};
    if (Cii * Cjj >= 0.0) {
        // Same-sign cofactors: the wedge stays on one side of the light cone
        // (the m=0 real angle for |r| <= 1 and the boost regime for |r| > 1).
        double denom = D;
        if (Cii < 0.0) denom = -denom;  // (-1)^d diagonal-sign fix (see dihedralAngle)
        const double r = -Cij / denom;
        return std::acos(std::complex<double>(r, 0.0));
    }
    // Opposite-sign cofactors: the wedge CROSSES the light cone (one facet
    // direction spacelike, one timelike -- the m=1 case, #581). The true
    // denominator sqrt(Cii)*sqrt(Cjj) (principal branches) is then purely
    // imaginary (+i*D), the cosine ratio -Cij/(i*D) = i*(Cij/D) is purely
    // imaginary, and the principal acos is
    //     theta = pi/2 - i*asinh(Cij/D).
    // Each crossing contributes exactly pi/2 to Re(theta) (Sorkin's quarter
    // turn) plus a signed boost; around a flat one-ray-per-quadrant vertex
    // star the boosts telescope to zero, which pins this sign convention.
    const double y = Cij / D;
    return {std::numbers::pi / 2.0, -std::asinh(y)};
}

std::complex<double> Simplex::lorentzianDeficitAngle() const {
    using cd = std::complex<double>;
    const cd twoPi(2.0 * std::numbers::pi, 0.0);
    if (!spacetime || vertices.empty()) return twoPi;
    const int topSize =
        spacetime->getMetric()->getSignature()->getDimensions() + 1;
    (void)topSize;
    cd sum(0.0, 0.0);
    for (auto *sigma : incidentTopCells())
        sum += sigma->lorentzianDihedralAngle(const_cast<Simplex *>(this));
    return twoPi - sum;
}

std::map<std::pair<std::uint64_t, std::uint64_t>, std::complex<double>>
Simplex::lorentzianDeficitAngleGradient() const {
    using cd = std::complex<double>;
    std::map<std::pair<std::uint64_t, std::uint64_t>, cd> grad;
    if (!spacetime || vertices.empty()) return grad;
    const int topSize =
        spacetime->getMetric()->getSignature()->getDimensions() + 1;

    // The top cells containing this hinge -- the same set lorentzianDeficitAngle
    // sums over. d(eps)/dl^2 = -sum_tau d(theta_tau)/dl^2.
    (void)topSize;
    for (auto *tau : incidentTopCells()) {
        const auto &tv = tau->getVertices();
        const int m = static_cast<int>(tv.size());          // d + 1
        // local indices of the two vertices NOT in the hinge
        std::vector<int> opp;
        for (int k = 0; k < m; ++k) {
            bool inHinge = false;
            for (const auto &hv : vertices)
                if (hv->getId() == tv[k]->getId()) { inHinge = true; break; }
            if (!inHinge) opp.push_back(k);
        }
        if (opp.size() != 2) continue;
        const int bi = opp[0] + 1, bj = opp[1] + 1;          // CM border offset

        const int n = m + 1;                                 // CM is (d+2)x(d+2)
        const std::vector<double> B = tau->cayleyMengerMatrix(/*wickRotate=*/false);
        const double detB = determinant(B, n);
        if (std::abs(detB) < 1e-300) continue;
        const std::vector<double> C = cofactorMatrix(B, n);
        // B^-1 = adj(B)/det = cof^T/det ; B symmetric => Binv symmetric.
        std::vector<double> Binv(static_cast<std::size_t>(n) * n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                Binv[i * n + j] = C[j * n + i] / detB;

        const double Cij = C[bi * n + bj];
        const double Cii = C[bi * n + bi];
        const double Cjj = C[bj * n + bj];
        const double sC = (Cii >= 0.0) ? 1.0 : -1.0;
        const double sP = (Cii * Cjj >= 0.0) ? 1.0 : -1.0;
        const double D = std::sqrt(std::abs(Cii * Cjj));
        if (D < 1e-300) continue;
        // Opposite-sign cofactors = the m=1 light-cone-crossing wedge (#581):
        // theta = pi/2 - i*asinh(y), y = Cij/D, so d theta/dy = -i/sqrt(1+y^2)
        // (never singular: |sin theta| = sqrt(1+y^2) >= 1). Same-sign wedges
        // keep the acos branch bit-for-bit.
        const bool crossing = sP < 0.0;
        const double denom = sC * D;
        const double y = Cij / D;
        cd dthetaDr;
        if (crossing) {
            dthetaDr = cd(0.0, -1.0) / std::sqrt(1.0 + y * y);
        } else {
            const double r = -Cij / denom;
            const cd theta = std::acos(cd(r, 0.0));
            const cd sinTheta = std::sin(theta);
            if (std::abs(sinTheta) < 1e-300) continue;       // flat/folded: skip
            dthetaDr = cd(-1.0, 0.0) / sinTheta;             // boost-safe branch
        }

        // dC_pq for the edge (a,b): dB is the indicator at (a+1,b+1)&(b+1,a+1);
        // dC = det[ tr(B^-1 dB) B^-1 - B^-1 dB B^-1 ], extracted entrywise.
        auto dCof = [&](int p, int q, int a, int b) -> double {
            const double bab = Binv[(a + 1) * n + (b + 1)];
            return detB * (2.0 * bab * Binv[p * n + q]
                           - (Binv[p * n + (a + 1)] * Binv[(b + 1) * n + q]
                              + Binv[p * n + (b + 1)] * Binv[(a + 1) * n + q]));
        };
        for (int a = 0; a < m; ++a) {
            for (int b = a + 1; b < m; ++b) {
                const double dCij = dCof(bi, bj, a, b);
                const double dCii = dCof(bi, bi, a, b);
                const double dCjj = dCof(bj, bj, a, b);
                // d sqrt(|Cii*Cjj|) — the sP factor makes this valid on both
                // sides of the crossing.
                const double dD = sP * (dCii * Cjj + Cii * dCjj) / (2.0 * D);
                double dr;
                if (crossing) {
                    dr = (dCij * D - Cij * dD) / (D * D);    // dy
                } else {
                    const double ddenom = sC * dD;
                    dr = -(dCij * denom - Cij * ddenom) / (denom * denom);
                }
                const std::uint64_t va = tv[a]->getId(), vb = tv[b]->getId();
                grad[{std::min(va, vb), std::max(va, vb)}] -= dthetaDr * dr;
            }
        }
    }
    return grad;
}

std::map<std::pair<std::pair<std::uint64_t, std::uint64_t>,
                   std::pair<std::uint64_t, std::uint64_t>>,
         std::complex<double>>
Simplex::lorentzianDeficitAngleHessian() const {
    using cd = std::complex<double>;
    using EK = std::pair<std::uint64_t, std::uint64_t>;
    std::map<std::pair<EK, EK>, cd> hess;
    if (!spacetime || vertices.empty()) return hess;
    const int topSize =
        spacetime->getMetric()->getSignature()->getDimensions() + 1;

    // d^2(eps)/dl^2_e dl^2_f = -sum_tau d^2(theta_tau). Same top-cell set and
    // cofactor machinery as lorentzianDeficitAngleGradient, carried one more
    // derivative: d^2 theta = (d2theta/dr^2) dr_e dr_f + (dtheta/dr) d2r.
    for (const auto &tau : vertices[0]->getSimplices()) {
        if (static_cast<int>(tau->size()) != topSize) continue;
        bool containsAll = true;
        for (std::size_t i = 1; i < vertices.size(); ++i)
            if (!tau->hasVertex(vertices[i])) { containsAll = false; break; }
        if (!containsAll) continue;

        const auto &tv = tau->getVertices();
        const int m = static_cast<int>(tv.size());
        std::vector<int> opp;
        for (int k = 0; k < m; ++k) {
            bool inHinge = false;
            for (const auto &hv : vertices)
                if (hv->getId() == tv[k]->getId()) { inHinge = true; break; }
            if (!inHinge) opp.push_back(k);
        }
        if (opp.size() != 2) continue;
        const int bi = opp[0] + 1, bj = opp[1] + 1;

        const int n = m + 1;
        const std::vector<double> B = tau->cayleyMengerMatrix(/*wickRotate=*/false);
        const double detB = determinant(B, n);
        if (std::abs(detB) < 1e-300) continue;
        const std::vector<double> C = cofactorMatrix(B, n);
        std::vector<double> Binv(static_cast<std::size_t>(n) * n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                Binv[i * n + j] = C[j * n + i] / detB;

        const double Cij = C[bi * n + bj];
        const double Cii = C[bi * n + bi];
        const double Cjj = C[bj * n + bj];
        const double sC = (Cii >= 0.0) ? 1.0 : -1.0;
        const double sP = (Cii * Cjj >= 0.0) ? 1.0 : -1.0;
        const double D = std::sqrt(std::abs(Cii * Cjj));
        if (D < 1e-300) continue;
        // Same m=1 crossing branch as the gradient (#581): on opposite-sign
        // cofactors theta = pi/2 - i*asinh(y) with y = Cij/D, so
        // d theta/dy = -i/(1+y^2)^{1/2} and d^2 theta/dy^2 = +i*y/(1+y^2)^{3/2}
        // (never singular). Same-sign wedges keep the acos machinery.
        const bool crossing = sP < 0.0;
        const double denom = sC * D;
        const double y = Cij / D;
        cd dthetaDr, d2thetaDr2;
        if (crossing) {
            const double onePlus = 1.0 + y * y;
            const double sq = std::sqrt(onePlus);
            dthetaDr = cd(0.0, -1.0) / sq;
            d2thetaDr2 = cd(0.0, y) / (onePlus * sq);
        } else {
            const double r = -Cij / denom;
            const cd theta = std::acos(cd(r, 0.0));
            const cd sinT = std::sin(theta);
            if (std::abs(sinT) < 1e-300) continue;
            dthetaDr = cd(-1.0, 0.0) / sinT;
            d2thetaDr2 = cd(-r, 0.0) / (sinT * sinT * sinT);
        }

        auto bb = [&](int x, int y) -> double { return Binv[x * n + y]; };
        // dC_pq/dl^2_(a,b): a,b local vertex indices (CM border = +1).
        auto dCof = [&](int p, int q, int a, int b) -> double {
            const int A = a + 1, Bn = b + 1;
            return detB * (2.0 * bb(A, Bn) * bb(p, q)
                           - bb(p, A) * bb(Bn, q) - bb(p, Bn) * bb(A, q));
        };
        // dBinv_xy/dl^2_(c,d) = -(Binv_xC Binv_Dy + Binv_xD Binv_Cy).
        auto dBi = [&](int x, int y, int c, int d) -> double {
            const int Cn = c + 1, Dn = d + 1;
            return -(bb(x, Cn) * bb(Dn, y) + bb(x, Dn) * bb(Cn, y));
        };
        // d^2 C_pq/dl^2_(a,b) dl^2_(c,d) = ddetB*T + detB*dT.
        auto d2Cof = [&](int p, int q, int a, int b, int c, int d) -> double {
            const int A = a + 1, Bn = b + 1, Cn = c + 1, Dn = d + 1;
            const double T = 2.0 * bb(A, Bn) * bb(p, q)
                             - bb(p, A) * bb(Bn, q) - bb(p, Bn) * bb(A, q);
            const double ddetB = detB * 2.0 * bb(Cn, Dn);
            const double dT =
                2.0 * (dBi(A, Bn, c, d) * bb(p, q) + bb(A, Bn) * dBi(p, q, c, d))
                - (dBi(p, A, c, d) * bb(Bn, q) + bb(p, A) * dBi(Bn, q, c, d))
                - (dBi(p, Bn, c, d) * bb(A, q) + bb(p, Bn) * dBi(A, q, c, d));
            return ddetB * T + detB * dT;
        };

        struct Loc { int a, b; std::uint64_t va, vb; double dr; };
        std::vector<Loc> es;
        for (int a = 0; a < m; ++a)
            for (int b = a + 1; b < m; ++b) {
                const double dCij = dCof(bi, bj, a, b);
                const double dCii = dCof(bi, bi, a, b);
                const double dCjj = dCof(bj, bj, a, b);
                const double dD = sP * (dCii * Cjj + Cii * dCjj) / (2.0 * D);
                double dr;
                if (crossing) {
                    dr = (dCij * D - Cij * dD) / (D * D);    // dy
                } else {
                    const double ddenom = sC * dD;
                    dr = -(dCij * denom - Cij * ddenom) / (denom * denom);
                }
                es.push_back({a, b, tv[a]->getId(), tv[b]->getId(), dr});
            }

        for (const auto &e : es) {
            const double dCij_e = dCof(bi, bj, e.a, e.b);
            const double dCii_e = dCof(bi, bi, e.a, e.b);
            const double dCjj_e = dCof(bj, bj, e.a, e.b);
            const double dP_e = dCii_e * Cjj + Cii * dCjj_e;
            const double dD_e = sP * dP_e / (2.0 * D);
            const double ddenom_e = sC * dD_e;
            for (const auto &f : es) {
                const double dCij_f = dCof(bi, bj, f.a, f.b);
                const double dCii_f = dCof(bi, bi, f.a, f.b);
                const double dCjj_f = dCof(bj, bj, f.a, f.b);
                const double dP_f = dCii_f * Cjj + Cii * dCjj_f;
                const double dD_f = sP * dP_f / (2.0 * D);
                const double ddenom_f = sC * dD_f;

                const double d2Cij = d2Cof(bi, bj, e.a, e.b, f.a, f.b);
                const double d2Cii = d2Cof(bi, bi, e.a, e.b, f.a, f.b);
                const double d2Cjj = d2Cof(bj, bj, e.a, e.b, f.a, f.b);
                const double d2P = d2Cii * Cjj + dCii_e * dCjj_f
                                   + dCii_f * dCjj_e + Cii * d2Cjj;
                const double d2D = sP * d2P / (2.0 * D)
                                   - dP_e * dP_f / (4.0 * D * D * D);

                // Quotient rule, second order. Same-sign: r = N/Den with
                // N = -Cij, Den = denom. Crossing: y = Cij/D (#581).
                double N, Ne, Nf, Nef, Den, De, Df, Def;
                if (crossing) {
                    N = Cij; Ne = dCij_e; Nf = dCij_f; Nef = d2Cij;
                    Den = D; De = dD_e; Df = dD_f; Def = d2D;
                } else {
                    N = -Cij; Ne = -dCij_e; Nf = -dCij_f; Nef = -d2Cij;
                    Den = denom; De = ddenom_e; Df = ddenom_f;
                    Def = sC * d2D;
                }
                const double d2r =
                    ((Nef * Den + Ne * Df - Nf * De - N * Def) * Den
                     - 2.0 * (Ne * Den - N * De) * Df) / (Den * Den * Den);

                const cd d2theta = d2thetaDr2 * cd(e.dr, 0.0) * cd(f.dr, 0.0)
                                   + dthetaDr * cd(d2r, 0.0);
                const EK ke{std::min(e.va, e.vb), std::max(e.va, e.vb)};
                const EK kf{std::min(f.va, f.vb), std::max(f.va, f.vb)};
                hess[{ke, kf}] -= d2theta;      // eps = 2pi - sum theta
            }
        }
    }
    return hess;
}

double Simplex::area(bool wickRotate) const {
    if (edges.size() < 3) return 0.0;
    auto sq = [&](std::size_t k) -> double {
        return wickRotate ? std::abs(edges[k]->getSquaredLength())
                          : edges[k]->getSquaredLength().real();
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

// Exact d(R^2)/d(l^2_e) for simplex `s` w.r.t. edge (ea,eb): R^2 = h^T G^-1 h
// (h = 1/2 diag G), so dR^2 = 2(dh)^T beta - beta^T (dG) beta, beta = G^-1 h.
// The Gram matrix is linear in l^2, so dG/dh are indicator matrices.
double dCircumR2(const ::tessera::mesh::Simplex* s,
                 std::uint64_t ea, std::uint64_t eb) {
    const int d = static_cast<int>(s->size()) - 1;
    if (d <= 0) return 0.0;
    const auto& sv = s->getVertices();
    const std::vector<double> G = s->gramMatrix(/*wickRotate=*/false);
    const double detG = ::tessera::mesh::Simplex::determinant(G, d);
    if (std::abs(detG) < 1e-300) return 0.0;
    const std::vector<double> cofG =
        ::tessera::mesh::Simplex::cofactorMatrix(G, d);
    std::vector<double> h(d), beta(d, 0.0);
    for (int i = 0; i < d; ++i) h[i] = 0.5 * G[i * d + i];
    for (int i = 0; i < d; ++i) {              // beta = G^-1 h, (G^-1)_ij=cof_ji/det
        double a = 0.0;
        for (int j = 0; j < d; ++j) a += cofG[j * d + i] * h[j];
        beta[i] = a / detG;
    }
    const std::uint64_t lo = std::min(ea, eb), hi = std::max(ea, eb);
    auto ind = [&](int p, int q) -> double {
        if (p == q) return 0.0;
        const std::uint64_t a = sv[p]->getId(), b = sv[q]->getId();
        return (std::min(a, b) == lo && std::max(a, b) == hi) ? 1.0 : 0.0;
    };
    std::vector<double> dG(static_cast<std::size_t>(d) * d), dh(d);
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j)
            dG[i * d + j] = 0.5 * (ind(0, i + 1) + ind(0, j + 1) - ind(i + 1, j + 1));
    for (int i = 0; i < d; ++i) dh[i] = 0.5 * dG[i * d + i];
    double r = 0.0;
    for (int i = 0; i < d; ++i) r += 2.0 * dh[i] * beta[i];
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j) r -= beta[i] * dG[i * d + j] * beta[j];
    return r;
}

// Exact d^2(R^2)/d(l^2_e)d(l^2_f). Since dR^2 = 2(dh)^T beta - beta^T dG beta and
// G is linear in l^2 (dG, dh constant indicators), the second derivative is
// 2(dh_e)^T (d_f beta) - 2 beta^T dG_e (d_f beta), with
// d_f beta = G^-1 (dh_f - dG_f beta). Symmetric in (e,f).
double d2CircumR2(const ::tessera::mesh::Simplex* s,
                  std::uint64_t ea, std::uint64_t eb,
                  std::uint64_t fa, std::uint64_t fb) {
    const int d = static_cast<int>(s->size()) - 1;
    if (d <= 0) return 0.0;
    const auto& sv = s->getVertices();
    const std::vector<double> G = s->gramMatrix(/*wickRotate=*/false);
    const double detG = ::tessera::mesh::Simplex::determinant(G, d);
    if (std::abs(detG) < 1e-300) return 0.0;
    const std::vector<double> cofG =
        ::tessera::mesh::Simplex::cofactorMatrix(G, d);
    std::vector<double> Ginv(static_cast<std::size_t>(d) * d);
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j)
            Ginv[i * d + j] = cofG[j * d + i] / detG;   // (G^-1)_ij = C_ji/det
    std::vector<double> h(d), beta(d, 0.0);
    for (int i = 0; i < d; ++i) h[i] = 0.5 * G[i * d + i];
    for (int i = 0; i < d; ++i) {
        double a = 0.0;
        for (int j = 0; j < d; ++j) a += Ginv[i * d + j] * h[j];
        beta[i] = a;
    }
    auto indMat = [&](std::uint64_t e0, std::uint64_t e1,
                      std::vector<double>& dG, std::vector<double>& dh) {
        const std::uint64_t lo = std::min(e0, e1), hi = std::max(e0, e1);
        auto ind = [&](int p, int q) -> double {
            if (p == q) return 0.0;
            const std::uint64_t a = sv[p]->getId(), b = sv[q]->getId();
            return (std::min(a, b) == lo && std::max(a, b) == hi) ? 1.0 : 0.0;
        };
        dG.assign(static_cast<std::size_t>(d) * d, 0.0);
        dh.assign(d, 0.0);
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j)
                dG[i * d + j] =
                    0.5 * (ind(0, i + 1) + ind(0, j + 1) - ind(i + 1, j + 1));
        for (int i = 0; i < d; ++i) dh[i] = 0.5 * dG[i * d + i];
    };
    std::vector<double> dG_e, dh_e, dG_f, dh_f;
    indMat(ea, eb, dG_e, dh_e);
    indMat(fa, fb, dG_f, dh_f);
    // d_f beta = G^-1 (dh_f - dG_f beta)
    std::vector<double> tmp(d, 0.0), dbeta_f(d, 0.0);
    for (int i = 0; i < d; ++i) {
        double a = dh_f[i];
        for (int j = 0; j < d; ++j) a -= dG_f[i * d + j] * beta[j];
        tmp[i] = a;
    }
    for (int i = 0; i < d; ++i) {
        double a = 0.0;
        for (int j = 0; j < d; ++j) a += Ginv[i * d + j] * tmp[j];
        dbeta_f[i] = a;
    }
    double r = 0.0;
    for (int i = 0; i < d; ++i) r += 2.0 * dh_e[i] * dbeta_f[i];
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j)
            r -= 2.0 * beta[i] * dG_e[i * d + j] * dbeta_f[j];
    return r;
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

std::vector<Simplex *> Simplex::incidentTopCells() const {
    std::vector<Simplex *> out;
    if (!spacetime || vertices.empty()) return out;
    const int topSize =
        spacetime->getMetric()->getSignature()->getDimensions() + 1;
    std::unordered_set<std::uint64_t> seen;
    for (const auto &anchor : vertices) {
        for (const auto &sigma : anchor->getSimplices()) {
            if (static_cast<int>(sigma->size()) != topSize) continue;
            bool containsAll = true;
            for (const auto &hv : vertices)
                if (!sigma->hasVertex(hv)) { containsAll = false; break; }
            if (!containsAll) continue;
            if (seen.insert(sigma->fingerprint.fingerprint()).second)
                out.push_back(sigma);
        }
    }
    return out;
}

bool Simplex::hasTopCoface() const {
    if (!spacetime || vertices.empty()) return false;
    const int topSize =
        spacetime->getMetric()->getSignature()->getDimensions() + 1;
    if (static_cast<int>(size()) >= topSize) return true;  // already top
    return !incidentTopCells().empty();
}

int Simplex::ambientTopDimension() const {
    // Prefer the metric dimension: it is the genuine ambient n and is immune to
    // orphan cofaces a move may have left dangling in this simplex's coface list.
    if (spacetime) return spacetime->getMetric()->getSignature()->getDimensions();
    // Coordinate-free fixture (no spacetime): fall back to the coface walk.
    const Simplex* top = this;
    while (!top->getCofaces().empty()) top = top->getCofaces()[0];
    return static_cast<int>(top->size()) - 1;
}

double Simplex::dualVolume() const {
    return dualVolRec(this, ambientTopDimension());
}

std::map<std::pair<std::uint64_t, std::uint64_t>, double>
Simplex::volumeGradient() const {
    // dV/dl^2_e = (V/2) tr(G^-1 dG_e), Jacobi's formula on the Gram determinant
    // (V = sgn sqrt(|det G|)/d!, G linear in l^2 so dG_e is an indicator matrix —
    // the same dG the #354 dCircumR2 uses). G^-1 via the adjugate (cofactor^T/det),
    // Eigen-free, matching circumFromGram / volume().
    std::map<std::pair<std::uint64_t, std::uint64_t>, double> grad;
    const int d = static_cast<int>(size()) - 1;
    if (d < 1) return grad;
    const std::vector<double> G = gramMatrix(/*wickRotate=*/false);
    if (static_cast<int>(G.size()) != d * d) return grad;
    const double detG = determinant(G, d);
    if (std::abs(detG) < 1e-300) return grad;
    const std::vector<double> cofG = cofactorMatrix(G, d);  // cof[r*d+c] = C_rc
    const double V = volume();
    const auto &sv = vertices;
    for (std::size_t p = 0; p < sv.size(); ++p)
        for (std::size_t q = p + 1; q < sv.size(); ++q) {
            const std::uint64_t a = sv[p]->getId(), b = sv[q]->getId();
            const std::pair<std::uint64_t, std::uint64_t> ek{std::min(a, b),
                                                             std::max(a, b)};
            auto ind = [&](int i, int j) -> double {
                if (i == j) return 0.0;
                const std::uint64_t x = sv[static_cast<std::size_t>(i)]->getId();
                const std::uint64_t y = sv[static_cast<std::size_t>(j)]->getId();
                return (std::min(x, y) == ek.first && std::max(x, y) == ek.second)
                           ? 1.0 : 0.0;
            };
            // tr(G^-1 dG) = sum_ij (G^-1)_ij dG_ji; dG symmetric, (G^-1)_ij=cof_ji/det.
            double tr = 0.0;
            for (int i = 0; i < d; ++i)
                for (int j = 0; j < d; ++j) {
                    const double dGij =
                        0.5 * (ind(0, i + 1) + ind(0, j + 1) - ind(i + 1, j + 1));
                    const double GinvIJ =
                        cofG[static_cast<std::size_t>(j) * d + i] / detG;
                    tr += GinvIJ * dGij;
                }
            grad[ek] += 0.5 * V * tr;
        }
    return grad;
}

std::map<std::pair<std::uint64_t, std::uint64_t>, double>
Simplex::dualVolumeGradient() const {
    std::map<std::pair<std::uint64_t, std::uint64_t>, double> grad;
    if (vertices.empty()) return grad;
    const int n = ambientTopDimension();
    const int k = static_cast<int>(size()) - 1;
    if (k != n - 2) return grad;          // the (n-2) hinge the Regge action needs

    const double Rh2 = circumradiusSquared();
    struct Facet {
        const Simplex* cf; double sgn; double R1; double inner;
        std::vector<std::pair<const Simplex*, std::pair<double, double>>> tops;
    };
    std::vector<Facet> fs;
    std::set<std::pair<std::uint64_t, std::uint64_t>> edges;
    for (const auto& cf : getCofaces()) {
        Facet f;
        f.cf = cf; f.sgn = oppositeVertexSign(cf, this);
        f.R1 = cf->circumradiusSquared(); f.inner = 0.0;
        for (const auto& tp : cf->getCofaces()) {
            const double sgn2 = oppositeVertexSign(tp, cf);
            const double R2 = tp->circumradiusSquared();
            f.inner += sgn2 * signedSqrt(R2 - f.R1);
            f.tops.push_back({tp, {sgn2, R2}});
            const auto& tv = tp->getVertices();
            for (std::size_t i = 0; i < tv.size(); ++i)
                for (std::size_t j = i + 1; j < tv.size(); ++j) {
                    const std::uint64_t a = tv[i]->getId(), b = tv[j]->getId();
                    edges.insert({std::min(a, b), std::max(a, b)});
                }
        }
        fs.push_back(std::move(f));
    }
    const double inv = 1.0 / (static_cast<double>(n - k) * (n - k - 1));
    for (const auto& e : edges) {
        const double dRh = dCircumR2(this, e.first, e.second);
        double dV = 0.0;
        for (const auto& f : fs) {
            const double dR1 = dCircumR2(f.cf, e.first, e.second);
            const double x1 = f.R1 - Rh2;
            const double ss1 = signedSqrt(x1);
            const double dss1 = 1.0 / (2.0 * std::sqrt(std::abs(x1) + 1e-300));
            double dinner = 0.0;
            for (const auto& t : f.tops) {
                const double R2 = t.second.second, sgn2 = t.second.first;
                const double dR2 = dCircumR2(t.first, e.first, e.second);
                const double x2 = R2 - f.R1;
                dinner += sgn2 * (dR2 - dR1)
                          / (2.0 * std::sqrt(std::abs(x2) + 1e-300));
            }
            dV += f.sgn * (dss1 * (dR1 - dRh) * f.inner + ss1 * dinner);
        }
        grad[e] = dV * inv;
    }
    return grad;
}

std::map<std::pair<std::pair<std::uint64_t, std::uint64_t>,
                   std::pair<std::uint64_t, std::uint64_t>>,
         double>
Simplex::dualVolumeHessian() const {
    using EK = std::pair<std::uint64_t, std::uint64_t>;
    std::map<std::pair<EK, EK>, double> hess;
    if (vertices.empty()) return hess;
    const int n = ambientTopDimension();
    const int k = static_cast<int>(size()) - 1;
    if (k != n - 2) return hess;

    const double Rh2 = circumradiusSquared();
    struct Facet {
        const Simplex* cf; double sgn; double R1;
        std::vector<std::pair<const Simplex*, std::pair<double, double>>> tops;
    };
    std::vector<Facet> fs;
    std::set<EK> edges;
    for (const auto& cf : getCofaces()) {
        Facet f; f.cf = cf; f.sgn = oppositeVertexSign(cf, this);
        f.R1 = cf->circumradiusSquared();
        for (const auto& tp : cf->getCofaces()) {
            const double sgn2 = oppositeVertexSign(tp, cf);
            f.tops.push_back({tp, {sgn2, tp->circumradiusSquared()}});
            const auto& tv = tp->getVertices();
            for (std::size_t i = 0; i < tv.size(); ++i)
                for (std::size_t j = i + 1; j < tv.size(); ++j) {
                    const std::uint64_t a = tv[i]->getId(), b = tv[j]->getId();
                    edges.insert({std::min(a, b), std::max(a, b)});
                }
        }
        fs.push_back(std::move(f));
    }
    const double inv = 1.0 / (static_cast<double>(n - k) * (n - k - 1));
    auto gp = [](double x) -> double {
        return 1.0 / (2.0 * std::sqrt(std::abs(x) + 1e-300));
    };
    auto gpp = [](double x) -> double {
        const double ax = std::abs(x) + 1e-300;
        return -((x < 0.0) ? -1.0 : 1.0) / (4.0 * ax * std::sqrt(ax));
    };
    const std::vector<EK> ev(edges.begin(), edges.end());
    for (const auto& e : ev) {
        const double dRh_e = dCircumR2(this, e.first, e.second);
        for (const auto& f : ev) {
            const double dRh_f = dCircumR2(this, f.first, f.second);
            const double d2Rh =
                d2CircumR2(this, e.first, e.second, f.first, f.second);
            double dV2 = 0.0;
            for (const auto& fac : fs) {
                const double dR1_e = dCircumR2(fac.cf, e.first, e.second);
                const double dR1_f = dCircumR2(fac.cf, f.first, f.second);
                const double d2R1 =
                    d2CircumR2(fac.cf, e.first, e.second, f.first, f.second);
                const double x1 = fac.R1 - Rh2;
                const double ss1 = signedSqrt(x1);
                const double dx1_e = dR1_e - dRh_e, dx1_f = dR1_f - dRh_f;
                const double dss1_e = gp(x1) * dx1_e, dss1_f = gp(x1) * dx1_f;
                const double d2ss1 = gpp(x1) * dx1_e * dx1_f + gp(x1) * (d2R1 - d2Rh);
                double S = 0.0, dS_e = 0.0, dS_f = 0.0, d2S = 0.0;
                for (const auto& t : fac.tops) {
                    const double sgn2 = t.second.first, R2 = t.second.second;
                    const double dR2_e = dCircumR2(t.first, e.first, e.second);
                    const double dR2_f = dCircumR2(t.first, f.first, f.second);
                    const double d2R2 =
                        d2CircumR2(t.first, e.first, e.second, f.first, f.second);
                    const double x2 = R2 - fac.R1;
                    const double dx2_e = dR2_e - dR1_e, dx2_f = dR2_f - dR1_f;
                    S += sgn2 * signedSqrt(x2);
                    dS_e += sgn2 * gp(x2) * dx2_e;
                    dS_f += sgn2 * gp(x2) * dx2_f;
                    d2S += sgn2 * (gpp(x2) * dx2_e * dx2_f + gp(x2) * (d2R2 - d2R1));
                }
                dV2 += fac.sgn
                       * (d2ss1 * S + dss1_e * dS_f + dss1_f * dS_e + ss1 * d2S);
            }
            hess[{e, f}] = dV2 * inv;
        }
    }
    return hess;
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
