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
#include "mesh/EdgeList.h"
#include "mesh/VertexList.h"
#include "mesh/EdgeKey.h"
#include "mesh/Edge.h"
#include "mesh/ForwardDeclarations.h"
#include "mesh/Simplex.h"
#include "mesh/Fingerprint.h"
#include "spacetime/Spacetime.h"
#include "utils.h"
#include <iomanip>
#include <sstream>

namespace caset {
Vertex::Vertex() noexcept : id(0) { }
Vertex::Vertex(const std::uint64_t id_, const std::vector<double> &coords) noexcept : id(id_), coordinates(coords),
  fingerprint({id_}) { }
Vertex::Vertex(const std::uint64_t id_) noexcept : id(id_), fingerprint({id_}) { }

std::uint64_t Vertex::getId() const noexcept { return id; }

void Vertex::setTime(double time) noexcept {
  if (coordinates.empty()) {
    coordinates = std::vector<double>();
    coordinates.push_back(time);
  }
  coordinates[0] = time;
}

[[nodiscard]] double Vertex::getTime() const {
  if (coordinates.empty()) {
    return 0;
  }
  if (coordinates.size() == 1) {
    return std::abs(coordinates[0]);
  }
  if (coordinates.size() >= 4) {
    double sumOfSquares = 0;
    for (const auto c : coordinates) {
      sumOfSquares += c * c;
    }
    return std::sqrt(sumOfSquares);
  }
  const std::string msg = "Invalid coordinate vector of length " + std::to_string(coordinates.size());
  throw std::out_of_range(msg);
}

bool Vertex::operator==(const Vertex &vertex) const noexcept {
  return vertex.getId() == id;
}

const std::vector<double> &
Vertex::getCoordinates() const {
  if (coordinates.empty()) {
    throw std::runtime_error("You requested coordinates for a vertex that is coordinate independent.");
  }
  return coordinates;
}

void
Vertex::setCoordinates(const std::vector<double> &coords) noexcept {
  coordinates = coords;
}

Edges
Vertex::getEdges() const noexcept {
  Edges edges;
  edges.reserve(inEdges.size() + outEdges.size());
  for (const auto &e : inEdges) edges.push_back(e);
  for (const auto &e : outEdges) edges.push_back(e);
  return edges;
}

EdgePtr Vertex::getEdge(const EdgePtr &edge) const {
  auto fp = edge->fingerprint.fingerprint();
  for (const auto &e : inEdges) {
    if (e->fingerprint.fingerprint() == fp) return e;
  }
  for (const auto &e : outEdges) {
    if (e->fingerprint.fingerprint() == fp) return e;
  }
  return nullptr;
}

std::pair<EdgePtrSet, EdgePtrSet>
Vertex::moveEdgesToImpl(
  const VertexPtr &recipient,
  Spacetime *spacetime,
  EdgeDirection direction
) {
#ifdef CASET_ASSERTIONS
  if (spacetime == nullptr) {
    throw std::runtime_error("Spacetime was null in vertex.cpp");
  }
#endif
  EdgePtrSet oldEdges{};
  EdgePtrSet newEdges{};

  Edges &edgesToMove = (direction == EdgeDirection::In) ? inEdges : outEdges;
  const char *directionStr = (direction == EdgeDirection::In) ? "in-edge" : "out-edge";

  // Collect old edges before we modify the list
  for (auto &e : edgesToMove) oldEdges.insert(e);

  for (auto &oldEdge : edgesToMove) {
    const auto &targetVertex = oldEdge->getTarget();
    const auto &sourceVertex = oldEdge->getSource();

    if (direction == EdgeDirection::In) {
#ifdef CASET_ASSERTIONS
      if (sourceVertex == this) throw std::runtime_error("sourceVertex was this");
#endif
      sourceVertex->removeOutEdge(oldEdge);
    } else if (direction == EdgeDirection::Out) {
#ifdef CASET_ASSERTIONS
      if (targetVertex == this) throw std::runtime_error("targetVertex was this");
#endif
      targetVertex->removeInEdge(oldEdge);
    }

    spacetime->getEdgeList()->remove(oldEdge);

    // For inEdges: redirect edge to point TO the new vertex (new source = vertex)
    // For outEdges: redirect edge to point FROM the new vertex (new target = vertex)
    const auto &newEdge = (direction == EdgeDirection::In)
                            ? spacetime->createEdge(sourceVertex, recipient, oldEdge->getSquaredLength())
                            : spacetime->createEdge(recipient, targetVertex, oldEdge->getSquaredLength());

    newEdges.insert(newEdge);
  }
  edgesToMove.clear();
  return {oldEdges, newEdges};
}

std::pair<EdgePtrSet, EdgePtrSet>
Vertex::moveInEdgesTo(
  const VertexPtr &vertex,
  Spacetime *spacetime
) {
  return moveEdgesToImpl(vertex, spacetime, EdgeDirection::In);
}

std::pair<EdgePtrSet, EdgePtrSet>
Vertex::moveEdgesTo(const VertexPtr &vertex, Spacetime *spacetime) {
#ifdef CASET_ASSERTIONS
  if (spacetime == nullptr) {
    throw std::runtime_error("Spacetime was null in vertex.cpp (2)");
  }
#endif
  EdgePtrSet oldEdges{};
  EdgePtrSet newEdges{};
  const auto &[oldInEdges, newInEdges] = moveInEdgesTo(vertex, spacetime);
  const auto &[oldOutEdges, newOutEdges] = moveOutEdgesTo(vertex, spacetime);
  oldEdges.insert(oldInEdges.begin(), oldInEdges.end());
  oldEdges.insert(oldOutEdges.begin(), oldOutEdges.end());
  newEdges.insert(newInEdges.begin(), newInEdges.end());
  newEdges.insert(newOutEdges.begin(), newOutEdges.end());
  return {oldEdges, newEdges};
}

std::pair<EdgePtrSet, EdgePtrSet>
Vertex::moveOutEdgesTo(const VertexPtr &vertex, Spacetime *spacetime) {
  return moveEdgesToImpl(vertex, spacetime, EdgeDirection::Out);
}

void Vertex::checkDuplicates(const std::string &msg) const {
  std::unordered_set<std::uint64_t> seen{};
  for (const auto &simp : simplices) {
    if (seen.contains(simp->fingerprint.fingerprint())) {
      CLOG(CRITICAL_LEVEL, "Simplex was duplicated for vertex!!!! " + msg);
      throw std::runtime_error(msg);
    }
    seen.insert(simp->fingerprint.fingerprint());
  }
}

bool Vertex::addSimplex(const SimplexPtr &simplex) {
#if CASET_ASSERTIONS
  if (simplex == nullptr) {
    CLOG(CRITICAL_LEVEL, "You passed a null simplex!");
    throw std::runtime_error("You passed a null simplex!");
  }
  checkDuplicates("Duplicated before emplacing a new simplex.");
#endif
  auto fp = simplex->fingerprint.fingerprint();
  for (const auto &s : simplices) {
    if (s->fingerprint.fingerprint() == fp) return false;
  }
  simplices.push_back(simplex);
#ifdef CASET_ASSERTIONS
  checkDuplicates("Duplicated after emplacing a new simplex.");
#endif
  return true;
}

bool Vertex::removeSimplex(const SimplexPtr &simplex) {
  CLOG(INFO_LEVEL, "Removing simplex: ", simplex->toString(), " from ", toString());
  auto fp = simplex->fingerprint.fingerprint();
  for (auto it = simplices.begin(); it != simplices.end(); ++it) {
    if ((*it)->fingerprint.fingerprint() == fp) {
      *it = simplices.back();
      simplices.pop_back();
      return true;
    }
  }
  return false;
}

const Simplices &
Vertex::getSimplices() const noexcept {
  return simplices;
}

#ifdef CASET_VERBOSE
std::string Vertex::toString() const noexcept {
  std::stringstream ss;
  ss << "<V" << "_{" << std::to_string(getId()) << "}";
  ss << "^{in=" << std::to_string(inEdges.size()) << "}";
  ss << "_{out=" << std::to_string(outEdges.size()) << "}";
  ss << " (t=" << std::fixed << std::setprecision(1) << getTime() << ")>";
  return latexToUtf8(ss.str());
}
#endif

void Vertex::addInEdge(const EdgePtr &edge) noexcept {
  auto fp = edge->fingerprint.fingerprint();
  for (const auto &e : inEdges) {
    if (e->fingerprint.fingerprint() == fp) return;
  }
  inEdges.push_back(edge);
}

void Vertex::addOutEdge(const EdgePtr &edge) noexcept {
  auto fp = edge->fingerprint.fingerprint();
  for (const auto &e : outEdges) {
    if (e->fingerprint.fingerprint() == fp) return;
  }
  outEdges.push_back(edge);
}

void Vertex::removeInEdge(const EdgePtr &edge) noexcept {
#ifdef CASET_ASSERTIONS
  if (edge == nullptr) {
    CLOG(WARN_LEVEL, "You passed a null pointer to remove an in edge! Refusing.");
    std::abort();
  }
#endif
  for (const auto &simplex : simplices) {
    if (simplex->hasVertex(edge->getSource()) && simplex->hasVertex(edge->getTarget())) {
      simplex->removeEdge(edge);
    }
  }
  auto fp = edge->fingerprint.fingerprint();
  for (auto it = inEdges.begin(); it != inEdges.end(); ++it) {
    if ((*it)->fingerprint.fingerprint() == fp) {
      *it = inEdges.back();
      inEdges.pop_back();
      return;
    }
  }
}

void Vertex::removeOutEdge(const EdgePtr &edge) noexcept {
#ifdef CASET_ASSERTIONS
  if (edge == nullptr) {
    CLOG(WARN_LEVEL, "You passed a null pointer to remove an out edge! Refusing.");
    std::abort();
  }
#endif
  for (const auto &simplex : simplices) {
    if (simplex->hasVertex(edge->getSource()) && simplex->hasVertex(edge->getTarget())) {
      simplex->removeEdge(edge);
    }
  }
  auto fp = edge->fingerprint.fingerprint();
  for (auto it = outEdges.begin(); it != outEdges.end(); ++it) {
    if ((*it)->fingerprint.fingerprint() == fp) {
      *it = outEdges.back();
      outEdges.pop_back();
      return;
    }
  }
}

std::size_t Vertex::degree() const noexcept { return inEdges.size() + outEdges.size(); }

const Edges &
Vertex::getInEdges() const noexcept { return inEdges; }

const Edges &
Vertex::getOutEdges() const noexcept { return outEdges; }
};
