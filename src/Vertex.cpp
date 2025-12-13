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
Vertex::getCoordinates() const
{
  if (coordinates.empty()) {
    throw std::runtime_error("You requested coordinates for a vertex that is coordinate independent.");
  }
  return coordinates;
}

void
Vertex::setCoordinates(const std::vector<double> &coords) noexcept
{
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
  py::object returnValue = py::make_tuple(py::cast(old), py::cast(new_));
  return returnValue;
}

py::object
Vertex::moveOutEdgesToForPython(
    const std::shared_ptr<Vertex> &vertex
    ) {
  auto [old, new_] = moveOutEdgesTo(vertex);
  py::object returnValue = py::make_tuple(py::cast(old), py::cast(new_));
  return returnValue;
}


std::pair<std::shared_ptr<EdgeKeySet>, std::shared_ptr<EdgeKeySet>>
Vertex::moveInEdgesTo(
  const std::shared_ptr<Vertex> &recipient // Recipient is the new target for the donated in-edges
  ) {
  std::shared_ptr<EdgeKeySet> oldEdges = std::make_shared<EdgeKeySet>();
  std::shared_ptr<EdgeKeySet> newEdges = std::make_shared<EdgeKeySet>();

  for (auto donorInEdge: inEdges) {
    if (donorInEdge == nullptr) {
      CLOG(ERROR_LEVEL, "Found a nullptr (in) edge in vertex ", toString());
      throw std::runtime_error("Found a nullptr (in) edge in vertex.");
    }
    auto oldKey = donorInEdge->getKey();
    auto source = donorInEdge->getSource();
    oldEdges->insert(oldKey);

    // Don't modify donorInEdge! Let updateKey handle it to avoid use-after-free
    // updateKey returns the canonical edge (either the updated edge or an existing merged edge)
    // Edge *canonicalEdge = edgeList->get(oldKey);
    EdgeKey newKey(source->getId(), recipient->getId());

    // Add canonical edge to both source and recipient
    recipient->addInEdge(donorInEdge);

    newEdges->insert(newKey);
  }
  inEdges.clear();
  return {oldEdges, newEdges};
}

std::pair<std::shared_ptr<EdgeKeySet>, std::shared_ptr<EdgeKeySet>>
Vertex::moveOutEdgesTo(const std::shared_ptr<Vertex> &recipient) {
  std::shared_ptr<EdgeKeySet> oldEdges = std::make_shared<EdgeKeySet>();
  std::shared_ptr<EdgeKeySet> newEdges = std::make_shared<EdgeKeySet>();
  std::unordered_set<Edge *> newOutEdges{};
  for (auto donorOutEdge : outEdges) {
    if (donorOutEdge == nullptr) {
      CLOG(ERROR_LEVEL, "Found a nullptr (out) edge in vertex ", toString());
      throw std::runtime_error("Found a nullptr (out) edge in vertex.");
    }
    auto oldKey = donorOutEdge->getKey();
    auto target = donorOutEdge->getTarget();
    oldEdges->insert(oldKey);

    // Don't modify donorOutEdge! Let updateKey handle it to avoid use-after-free
    // updateKey returns the canonical edge (either the updated edge or an existing merged edge)
    // Edge* canonicalEdge = edgeList->updateKey(oldKey, recipient->getId(), targetId);
    EdgeKey newKey(recipient->getId(), target->getId());

    // Add canonical edge to both target and recipient
    recipient->addOutEdge(donorOutEdge);

    newEdges->insert(newKey);
  }
  outEdges.clear();
  return {oldEdges, newEdges};
}

std::pair<std::shared_ptr<EdgeKeySet>, std::shared_ptr<EdgeKeySet>>
Vertex::moveEdgesTo(const std::shared_ptr<Vertex> &vertex) {
  EdgeKeySet oldEdgesSet = EdgeKeySet{};
  EdgeKeySet newEdgesSet = EdgeKeySet{};
  std::shared_ptr<EdgeKeySet> oldEdges = std::make_shared<EdgeKeySet>(oldEdgesSet);
  std::shared_ptr<EdgeKeySet> newEdges = std::make_shared<EdgeKeySet>(newEdgesSet);
  const auto &[oldInEdges, newInEdges] = moveInEdgesTo(vertex);
  const auto &[oldOutEdges, newOutEdges] = moveOutEdgesTo(vertex);
  oldEdges->insert(oldInEdges->begin(), oldInEdges->end());
  oldEdges->insert(oldOutEdges->begin(), oldOutEdges->end());
  newEdges->insert(newInEdges->begin(), newInEdges->end());
  newEdges->insert(newOutEdges->begin(), newOutEdges->end());
  return {oldEdges, newEdges};
}

py::object
Vertex::moveEdgesToForPython(const std::shared_ptr<Vertex> &vertex) {
  auto [old, new_] = moveEdgesTo(vertex);
  py::object returnValue = py::make_tuple(py::cast(old), py::cast(new_));
  return returnValue;
}

void Vertex::addSimplex(Simplex *simplex) {
  CLOG(INFO_LEVEL, "Adding simplex to vertex", toString());
#if CASET_DEBUG
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
// #if CASET_DEBUG
  // throw std::runtime_error("You tried to remove a simplex that the Vertex does not contain!");
// #endif
}

std::unordered_set<Simplex *>
Vertex::getSimplices() const noexcept
{
  return simplices;
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
  if (!outEdges.contains(edge)) CLOG(WARN_LEVEL, "Edge ", edge->toString(), " not found in vertex ", toString());
  outEdges.erase(edge);
}
void Vertex::addInEdge(Edge *edge) noexcept { inEdges.insert(edge); }
void Vertex::addOutEdge(Edge *edge) noexcept { outEdges.insert(edge); }
void Vertex::removeInEdge(Edge *edge) noexcept {
  if (!inEdges.contains(edge)) CLOG(WARN_LEVEL, "Edge ", edge->toString(), " not found in vertex ", toString());
  inEdges.erase(edge);
}
std::size_t Vertex::degree() const noexcept { return inEdges.size() + outEdges.size(); }

std::unordered_set<Edge *>
Vertex::getInEdges() const noexcept { return inEdges; }

std::unordered_set<Edge *>
Vertex::getOutEdges() const noexcept { return outEdges; }
};
