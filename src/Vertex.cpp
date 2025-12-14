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

#include "EdgeKey.h"
#include "Vertex.h"
#include "Edge.h"
#include "EdgeList.h"
#include "ForwardDeclarations.h"
#include "Simplex.h"

namespace caset {
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

std::unordered_set<Edge *>
Vertex::getEdges() const noexcept {
  std::unordered_set<Edge *> edges{};
  edges.reserve(inEdges.size() + outEdges.size());
  edges.insert(inEdges.begin(), inEdges.end());
  edges.insert(outEdges.begin(), outEdges.end());
  return edges;
}

py::object
Vertex::moveInEdgesToForPython(
  const std::shared_ptr<Vertex> &vertex) {
  auto [old, new_] = moveInEdgesTo(vertex);
  py::object returnValue = py::make_tuple(py::cast(*old), py::cast(*new_));
  return returnValue;
}

py::object
Vertex::moveOutEdgesToForPython(
  const std::shared_ptr<Vertex> &vertex
) {
  auto [old, new_] = moveOutEdgesTo(vertex);
  py::object returnValue = py::make_tuple(py::cast(*old), py::cast(*new_));
  return returnValue;
}

std::pair<std::shared_ptr<EdgeKeySet>, std::shared_ptr<EdgeKeySet> >
Vertex::moveEdgesToImpl(
  const std::shared_ptr<Vertex> &recipient, // Recipient is from the attached simplex.
  EdgeDirection direction
) {
  std::shared_ptr<EdgeKeySet> toDelete = std::make_shared<EdgeKeySet>();
  std::shared_ptr<EdgeKeySet> toUpdate = std::make_shared<EdgeKeySet>();

  // Select which edge set to operate on
  auto &edges = (direction == EdgeDirection::In) ? inEdges : outEdges;

  for (auto donorEdge : edges) {
    if (donorEdge == nullptr) {
      CLOG(ERROR_LEVEL, "Found a nullptr edge in vertex ", toString());
      throw std::runtime_error("Found a nullptr edge in vertex.");
    }

    EdgeKey oldKey = donorEdge->getKey();

    // Get the other vertex and replace the appropriate endpoint
    direction == EdgeDirection::In
      ? donorEdge->replaceTargetVertex(recipient)
      : donorEdge->replaceSourceVertex(recipient);

    // Try to add the edge to the recipient. The edge owned by the recipient is always canonical.
    const auto &[canonicalEdge, wasCanonical] =
        (direction == EdgeDirection::In)
          ? recipient->addInEdge(donorEdge)
          : recipient->addOutEdge(donorEdge);

#ifdef CASET_ASSERTIONS
    if (wasCanonical && canonicalEdge != donorEdge) throw std::runtime_error("Canonical lies!");
    if (!wasCanonical && canonicalEdge == donorEdge) throw std::runtime_error("Canonical lies (2)!");
#endif

    if (!wasCanonical) {
      // donorEdge is a duplicate and should be replaced in all simplices
      CLOG(WARN_LEVEL, "WILL MARK ", oldKey.toString(), " FOR DELETION");
      toDelete->insert(oldKey);
      donorEdge->replaceOnReferents(canonicalEdge);
    } else {
      // donorEdge is a new canonical edge that needs its key updated or to be replaced by its existing newKey's
      // (pre-existing) corresponding Edge in EdgeList. In other words. donorEdge either needs to be added to EdgeList
      // at newKey or it needs to be _replaced_ by what is already sitting at newKey.
#ifdef CASET_ASSERTIONS
      if (oldKey == donorEdge->getKey()) {
        throw std::runtime_error("We expected donorEdge to be a new canonical Edge, but it had the same key as the old edge. Key is by reference?");
      }
#endif
      toUpdate->insert(oldKey);
    }
  }

  std::unordered_set<Edge *> empty{};
  if (direction == EdgeDirection::In) {
    inEdges = {};
    // inEdges.swap(empty);
  } else {
    outEdges = {};
  }

#ifdef CASET_ASSERTIONS
  for (const auto &k : *toUpdate) {
    if (toDelete->contains(k)) {
      throw std::runtime_error("You attempted to update and delete a key at the same time.");
    }
  }
  for (const auto &k : *toDelete) {
    if (toUpdate->contains(k)) {
      throw std::runtime_error("You attempted to update and delete a key at the same time (2).");
    }
  }
#endif

  return {toUpdate, toDelete};
}

std::pair<std::shared_ptr<EdgeKeySet>, std::shared_ptr<EdgeKeySet> >
Vertex::moveInEdgesTo(const std::shared_ptr<Vertex> &recipient) {
  return moveEdgesToImpl(recipient, EdgeDirection::In);
}

std::pair<std::shared_ptr<EdgeKeySet>, std::shared_ptr<EdgeKeySet> >
Vertex::moveOutEdgesTo(const std::shared_ptr<Vertex> &recipient) {
  return moveEdgesToImpl(recipient, EdgeDirection::Out);
}

std::pair<std::shared_ptr<EdgeKeySet>, std::shared_ptr<EdgeKeySet> >
Vertex::absorbInto(const std::shared_ptr<Vertex> &vertex) {
  EdgeKeySet toUpdateSet = EdgeKeySet{};
  EdgeKeySet toDeleteSet = EdgeKeySet{};
  std::shared_ptr<EdgeKeySet> toUpdate = std::make_shared<EdgeKeySet>(toUpdateSet);
  std::shared_ptr<EdgeKeySet> toDelete = std::make_shared<EdgeKeySet>(toDeleteSet);
  const auto &[updateInEdges, deleteInEdges] = moveInEdgesTo(vertex);
  const auto &[updateOutEdges, deleteOutEdges] = moveOutEdgesTo(vertex);
  toUpdate->insert(updateInEdges->begin(), updateInEdges->end());
  toUpdate->insert(updateOutEdges->begin(), updateOutEdges->end());
  toDelete->insert(deleteInEdges->begin(), deleteInEdges->end());
  toDelete->insert(deleteOutEdges->begin(), deleteOutEdges->end());
  return {toUpdate, toDelete};
}

py::object
Vertex::moveEdgesToForPython(const std::shared_ptr<Vertex> &vertex) {
  auto [old, new_] = absorbInto(vertex);
  py::object returnValue = py::make_tuple(py::cast(*old), py::cast(*new_));
  return returnValue;
}

bool const Vertex::hasEdge(const Edge *edge) const {
  for (const auto &inEdge : inEdges) {
    if (edge == inEdge) return true;
  }
  for (const auto &outEdge : outEdges) {
    if (outEdge == edge) return true;
  }
  return false;
}

void Vertex::assertUnused() const {
  for (const auto &simplex : getSimplices()) {
    for (const auto &edge : simplex->getEdges()) {
      assert(!edge->hasVertex(this));
    }
    assert(!simplex->hasVertex(this));
  }
  for (const auto &edge : getEdges()) {
    assert(!edge->hasVertex(this));
  }
}

void Vertex::addSimplex(Simplex *simplex) {
#if CASET_ASSERTIONS
  for (const auto &simp : simplices) {
    if (simp == simplex) {
      CLOG(ERROR_LEVEL, "You tried to add a simplex more than once!");
      throw std::runtime_error("you tried to add a simplex more than once.");
    }
  }
#endif
  simplices.insert(simplex);
}

void Vertex::removeSimplex(Simplex *simplex) {
  simplices.erase(simplex);
}

std::unordered_set<Simplex *>
Vertex::getSimplices() const noexcept {
  return simplices;
}

std::vector<std::shared_ptr<Simplex>> Vertex::getSimplicesForPython() const {
  std::vector<std::shared_ptr<Simplex>> simplicesForPython{};
  simplicesForPython.reserve(simplices.size());
  for (const auto &s : simplices) {
    simplicesForPython.emplace_back(s);
  }
  return simplicesForPython;
}

std::string Vertex::toString() const noexcept {
  std::stringstream ss;
  ss << "<V" << std::to_string(id) << " ";
  ss << "(in=" << std::to_string(inEdges.size());
  ss << ", out=" << std::to_string(outEdges.size());
  ss << ", t=" << std::to_string(getTime()) << ")>";
  return ss.str();
}

void Vertex::removeOutEdge(Edge *edge) noexcept {
  if (!outEdges.contains(edge))
    CLOG(WARN_LEVEL, "Edge ", edge->toString(), " not found in vertex ", toString());
  outEdges.erase(edge);
}

std::pair<EdgeRawPtr, bool> Vertex::addInEdge(Edge *edge) noexcept {
  auto [it, inserted] = inEdges.insert(edge);
  return {*it, inserted};
}

std::pair<EdgeRawPtr, bool> Vertex::addOutEdge(Edge *edge) noexcept {
  auto [it, inserted] = outEdges.insert(edge);
  return {*it, inserted};
}

void Vertex::removeInEdge(Edge *edge) noexcept {
  if (!inEdges.contains(edge))
    CLOG(WARN_LEVEL, "Edge ", edge->toString(), " not found in vertex ", toString());
  inEdges.erase(edge);
}
std::size_t Vertex::degree() const noexcept { return inEdges.size() + outEdges.size(); }

std::unordered_set<Edge *>
Vertex::getInEdges() const noexcept { return inEdges; }

std::unordered_set<Edge *>
Vertex::getOutEdges() const noexcept { return outEdges; }
};
