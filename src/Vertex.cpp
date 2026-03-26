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
#include "EdgeList.h"
#include "VertexList.h"
#include "EdgeKey.h"
#include "Edge.h"
#include "ForwardDeclarations.h"
#include "Simplex.h"
#include "Fingerprint.h"
#include "spacetime/Spacetime.h"
#include "utils.h"
#include <iomanip>
#include <sstream>

namespace caset {
Vertex::Vertex() noexcept { id = 0; simplices.reserve(7); }
Vertex::Vertex(const std::uint64_t id_, const std::vector<double> &coords) noexcept : id(id_), coordinates(coords),
  fingerprint({id_}) { simplices.reserve(7); }
Vertex::Vertex(const std::uint64_t id_) noexcept : id(id_), fingerprint({id_}) { simplices.reserve(7);}

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

std::vector<double>
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

EdgePtr Vertex::getEdge(const EdgePtr &edge) {
  auto foundIn = inEdges.find(edge);
  if (foundIn != inEdges.end()) {
    return *foundIn;
  }
  auto foundOut = outEdges.find(edge);
  if (foundOut != outEdges.end()) {
    return *foundOut;
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

  EdgePtrSet &edgesToMove = (direction == EdgeDirection::In) ? inEdges : outEdges;
  const char *directionStr = (direction == EdgeDirection::In) ? "in-edge" : "out-edge";

  for (auto &oldEdge : edgesToMove) {
    const auto &targetVertex = oldEdge->getTarget();
    const auto &sourceVertex = oldEdge->getSource();

    if (direction == EdgeDirection::In) {
#ifdef CASET_ASSERTIONS
      if (sourceVertex.get() == this) throw std::runtime_error("sourceVertex was this");
#endif
      sourceVertex->removeOutEdge(oldEdge);
    } else if (direction == EdgeDirection::Out) {
#ifdef CASET_ASSERTIONS
      if (targetVertex.get() == this) throw std::runtime_error("targetVertex was this");
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

void Vertex::checkDuplicates(std::string msg) const {
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
  // CLOG(INFO_LEVEL, "Adding simplex to vertex", toString());
#if CASET_ASSERTIONS
  if (simplex == nullptr || simplex.get() == nullptr) {
    CLOG(CRITICAL_LEVEL, "You passed a null simplex!");
    throw std::runtime_error("You passed a null simplex!");
  }
  checkDuplicates("Duplicated before emplacing a new simplex.");
#endif
  const auto [it, inserted] = simplices.emplace(simplex);
#ifdef CASET_ASSERTIONS
  checkDuplicates("Duplicated after emplacing a new simplex.");
#endif
  return inserted;
}

bool Vertex::removeSimplex(const SimplexPtr &simplex) {
// #if CASET_ASSERTIONS
  // if (!simplices.contains(simplex)) {
    // throw std::runtime_error("You attempted to remove a simplex that did not exist");
  // }
// #endif
  CLOG(INFO_LEVEL, "Removing simplex: ", simplex->toString(), " from ", toString());
  return simplices.erase(simplex) > 0;
}

const SimplexPtrSet &
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
  inEdges.insert(edge);
}

void Vertex::addOutEdge(const EdgePtr &edge) noexcept {
  outEdges.insert(edge);
}

void Vertex::removeInEdge(const EdgePtr &edge) noexcept {
#ifdef CASET_ASSERTIONS
  if (edge == nullptr) {
    CLOG(WARN_LEVEL, "You passed a null pointer to remove an in edge! Refusing.");
    std::abort();
  }
  if (!inEdges.contains(edge)) {
    CLOG(WARN_LEVEL, "Edge ", edge->toString(), " not found in vertex ", toString());
    std::abort();
  }
#endif
  for (const auto &simplex : simplices) {
    simplex->removeEdge(edge);
  }
  inEdges.erase(edge);
}

void Vertex::removeOutEdge(const EdgePtr &edge) noexcept {
#ifdef CASET_ASSERTIONS
  if (edge == nullptr) {
    CLOG(WARN_LEVEL, "You passed a null pointer to remove an out edge! Refusing.");
    std::abort();
  }
  if (!outEdges.contains(edge)) {
    CLOG(WARN_LEVEL, "Edge ", edge->toString(), " not found in vertex ", toString());
    std::abort();
  }
#endif
  for (const auto &simplex : simplices) {
    simplex->removeEdge(edge);
  }
  outEdges.erase(edge);
}

std::size_t Vertex::degree() const noexcept { return inEdges.size() + outEdges.size(); }

const EdgePtrSet &
Vertex::getInEdges() const noexcept { return inEdges; }

const EdgePtrSet &
Vertex::getOutEdges() const noexcept { return outEdges; }
};
