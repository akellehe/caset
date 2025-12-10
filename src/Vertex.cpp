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

std::pair<std::shared_ptr<EdgeKeySet>, std::shared_ptr<EdgeKeySet>>
Vertex::moveInEdgesTo(
  const std::shared_ptr<Vertex> &vertex,
  const std::shared_ptr<EdgeList> &edgeList,
  const std::shared_ptr<VertexList> &vertexList
  ) {
  std::shared_ptr<EdgeKeySet> oldEdges = std::make_shared<EdgeKeySet>();
  std::shared_ptr<EdgeKeySet> newEdges = std::make_shared<EdgeKeySet>();

  for (auto edge_ : inEdges) {
    if (edge_ == nullptr) {
      CLOG(ERROR_LEVEL, "Found a nullptr (in) edge in vertex ", toString());
      throw std::runtime_error("Found a nullptr (in) edge in vertex.");
    }
    auto oldKey = edge_->getKey();
    auto edge = edgeList->remove(oldKey);
    auto raw = edge.get();
    CLOG(DEBUG_LEVEL, "Moving in-edge ", raw->toString(), " to ", vertex->toString());
    oldEdges->insert(oldKey);
    const auto sourceVertex = vertexList->get(raw->getSourceId());
    sourceVertex->removeOutEdge(raw);
    CLOG(DEBUG_LEVEL, "Changing target vertex from ", std::to_string(raw->getTargetId()), " to ", std::to_string(vertex->getId()));
    raw->replaceTargetVertex(vertex->getId());
    auto newKey = raw->getKey();
    newEdges->insert(newKey);
    // TODO: If there are issues with the edge pointer chanigng value; we may need to address it by extracting and storing via some other method here.
    vertex->addInEdge(raw);
    sourceVertex->addOutEdge(raw);
    edgeList->add(std::move(edge));
  }
  inEdges.clear();
  return {oldEdges, newEdges};
}

std::pair<EdgeKeySet, EdgeKeySet>
Vertex::moveInEdgesToForPython(
    const std::shared_ptr<Vertex> &vertex,
    const std::shared_ptr<EdgeList> &edgeList,
    const std::shared_ptr<VertexList> &vertexList) {
  auto [old, new_] = moveInEdgesTo(vertex, edgeList, vertexList);
  return {*old, *new_};
}

std::pair<EdgeKeySet, EdgeKeySet>
Vertex::moveOutEdgesToForPython(
    const std::shared_ptr<Vertex> &vertex,
    const std::shared_ptr<EdgeList> &edgeList,
    const std::shared_ptr<VertexList> &vertexList
    ) {
  auto [old, new_] = moveOutEdgesTo(vertex, edgeList, vertexList);
  return {*old, *new_};
}


std::pair<std::shared_ptr<EdgeKeySet>, std::shared_ptr<EdgeKeySet>>
Vertex::moveOutEdgesTo(const std::shared_ptr<Vertex> &vertex, const std::shared_ptr<EdgeList> &edgeList, const std::shared_ptr<VertexList> &vertexList) {
  std::shared_ptr<EdgeKeySet> oldEdges = std::make_shared<EdgeKeySet>();
  std::shared_ptr<EdgeKeySet> newEdges = std::make_shared<EdgeKeySet>();
  std::unordered_set<Edge *> newOutEdges{};
  for (auto edge_ : outEdges) {
    if (edge_ == nullptr) {
      CLOG(ERROR_LEVEL, "Found a nullptr (out) edge in vertex ", toString());
      throw std::runtime_error("Found a nullptr (out) edge in vertex.");
    }
    auto oldKey = edge_->getKey();
    auto edge = edgeList->remove(oldKey);
    auto raw = edge.get();
    CLOG(DEBUG_LEVEL, "Moving out-edge ", raw->toString(), " to ", vertex->toString());
    oldEdges->insert(oldKey);
    const auto targetVertex = vertexList->get(raw->getTargetId());
    targetVertex->removeInEdge(raw);
    CLOG(DEBUG_LEVEL, "Changing source vertex from ", std::to_string(raw->getSourceId()), " to ", std::to_string(vertex->getId()));
    raw->replaceSourceVertex(vertex->getId());
    auto newKey = raw->getKey();
    newEdges->insert(newKey);
    vertex->addOutEdge(raw);
    targetVertex->addInEdge(raw);
    edgeList->add(std::move(edge));
  }
  outEdges.clear();
  return {oldEdges, newEdges};
}

std::pair<EdgeKeySet, EdgeKeySet>
Vertex::moveEdgesTo(const std::shared_ptr<Vertex> &vertex, const std::shared_ptr<EdgeList> &edgeList, const std::shared_ptr<VertexList> &vertexList) {
  EdgeKeySet oldEdges = EdgeKeySet{};
  EdgeKeySet newEdges = EdgeKeySet{};
  const auto &[oldInEdges, newInEdges] = moveInEdgesTo(vertex, edgeList, vertexList);
  const auto &[oldOutEdges, newOutEdges] = moveOutEdgesTo(vertex, edgeList, vertexList);
  oldEdges.insert(oldInEdges->begin(), oldInEdges->end());
  oldEdges.insert(oldOutEdges->begin(), oldOutEdges->end());
  newEdges.insert(newInEdges->begin(), newInEdges->end());
  newEdges.insert(newOutEdges->begin(), newOutEdges->end());
  return {oldEdges, newEdges};
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
};
