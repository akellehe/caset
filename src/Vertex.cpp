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

[[nodiscard]] std::pair<std::shared_ptr<Edge>, std::shared_ptr<Vertex> >
Vertex::moveTo(const std::shared_ptr<Vertex> &vertex)
{
  if (outEdges.empty()) {
    throw std::runtime_error("Cannot execute move; outEdges is empty!");
  }
  std::shared_ptr<Edge> edge = std::make_shared<Edge>(getId(), vertex->getId());
  if (!outEdges.contains(edge)) {
    throw std::runtime_error("No edge to this vertex exists.");
  }
  return std::pair<std::shared_ptr<Edge>, std::shared_ptr<Vertex> >({
      *outEdges.find(edge), vertex
  });
}

std::unordered_set<std::shared_ptr<Edge>, EdgeHash, EdgeEq>
Vertex::getEdges() const noexcept {
  std::unordered_set<std::shared_ptr<Edge>, EdgeHash, EdgeEq> edges;
  edges.reserve(inEdges.size() + outEdges.size());
  edges.insert(inEdges.begin(), inEdges.end());
  edges.insert(outEdges.begin(), outEdges.end());
  return edges;
}

std::shared_ptr<Edge>
Vertex::getEdge(const EdgeKey &key)
{
  const auto testEdge = std::make_shared<Edge>(key.first, key.second);
  if (inEdges.contains(testEdge)) return *inEdges.find(testEdge);
  if (outEdges.contains(testEdge)) return *outEdges.find(testEdge);
  return nullptr;
}

std::shared_ptr<Edge> Vertex::getEdge(const EdgePtr &edge)
{
  if (inEdges.contains(edge)) return *inEdges.find(edge);
  if (outEdges.contains(edge)) return *outEdges.find(edge);
  return nullptr;
}

std::pair<std::shared_ptr<EdgeIdSet>, std::shared_ptr<EdgeIdSet>>
Vertex::moveInEdgesTo(
  const std::shared_ptr<Vertex> &vertex,
  const std::shared_ptr<EdgeList> &edgeList,
  const std::shared_ptr<VertexList> &vertexList
  ) {
  std::shared_ptr<EdgeIdSet> oldEdges = std::make_shared<EdgeIdSet>();
  std::shared_ptr<EdgeIdSet> newEdges = std::make_shared<EdgeIdSet>();
  for (const auto &edge : inEdges) {
    CLOG(DEBUG_LEVEL, "Moving in-edge ", edge->toString(), " to ", vertex->toString());
    oldEdges->insert(edge->getKey());
    edgeList->remove(edge);
    const auto sourceVertex = vertexList->get(edge->getSourceId());
    sourceVertex->removeOutEdge(edge);
    CLOG(DEBUG_LEVEL, "Changing target vertex from ", std::to_string(edge->getTargetId()), " to ", std::to_string(vertex->getId()));
    edge->replaceTargetVertex(vertex->getId());
    newEdges->insert(edge->getKey());
    vertex->addInEdge(edgeList->add(edge));
    sourceVertex->addOutEdge(edgeList->get(edge->getKey()));
  }
  inEdges.clear();
  return {oldEdges, newEdges};
}

std::pair<EdgeIdSet, EdgeIdSet>
Vertex::moveEdgesTo(const std::shared_ptr<Vertex> &vertex, const std::shared_ptr<EdgeList> &edgeList, const std::shared_ptr<VertexList> &vertexList) {
  EdgeIdSet oldEdges = EdgeIdSet{};
  EdgeIdSet newEdges = EdgeIdSet{};
  const auto &[oldInEdges, newInEdges] = moveInEdgesTo(vertex, edgeList, vertexList);
  const auto &[oldOutEdges, newOutEdges] = moveOutEdgesTo(vertex, edgeList, vertexList);
  oldEdges.insert(oldInEdges->begin(), oldInEdges->end());
  oldEdges.insert(oldOutEdges->begin(), oldOutEdges->end());
  newEdges.insert(newInEdges->begin(), newInEdges->end());
  newEdges.insert(newOutEdges->begin(), newOutEdges->end());
  return {oldEdges, newEdges};
}

std::pair<std::shared_ptr<EdgeIdSet>, std::shared_ptr<EdgeIdSet>>
Vertex::moveOutEdgesTo(const std::shared_ptr<Vertex> &vertex, const std::shared_ptr<EdgeList> &edgeList, const std::shared_ptr<VertexList> &vertexList) {
  std::shared_ptr<EdgeIdSet> oldEdges = std::make_shared<EdgeIdSet>();
  std::shared_ptr<EdgeIdSet> newEdges = std::make_shared<EdgeIdSet>();
  std::unordered_set<std::shared_ptr<Edge>, EdgeHash, EdgeEq> newOutEdges{};
  for (const auto &edge : outEdges) {
    CLOG(DEBUG_LEVEL, "Moving out-edge ", edge->toString(), " to ", vertex->toString());
    oldEdges->insert(edge->getKey());
    edgeList->remove(edge);
    const auto targetVertex = vertexList->get(edge->getTargetId());
    targetVertex->removeInEdge(edge);
    CLOG(DEBUG_LEVEL, "Changing source vertex from ", std::to_string(edge->getSourceId()), " to ", std::to_string(vertex->getId()));
    edge->replaceSourceVertex(vertex->getId());
    newEdges->insert(edge->getKey());
    vertex->addOutEdge(edgeList->add(edge));
    targetVertex->addInEdge(edgeList->get(edge->getKey()));
  }
  outEdges.clear();
  return {oldEdges, newEdges};
}

void Vertex::addSimplex(const std::shared_ptr<Simplex> &simplex) {
  CLOG(INFO_LEVEL, "Adding simplex to vertex", toString());
#if CASET_DEBUG
  for (const auto &simp : simplices) {
    if (simp == simplex) {
      CLOG(ERROR_LEVEL, "You tried to add a simplex more than once!");
      throw std::runtime_error("you tried to add a simplex more than once.");
    }
  }
#endif
  simplices.push_back(simplex);
}

void Vertex::removeSimplex(const std::shared_ptr<Simplex> &simplex) {
  for (auto i=0; i<simplices.size(); i++) {
    if (simplex == simplices[i]) {
      simplices.erase(simplices.begin() + i);
      return;
    }
  }
#if CASET_DEBUG
  throw std::runtime_error("You tried to remove a simplex that the Vertex does not contain!");
#endif
}

std::vector<std::shared_ptr<Simplex>>
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
