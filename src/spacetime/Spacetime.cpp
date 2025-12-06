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

//
// Created by andrew on 10/23/25.
//

#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include "Logger.h"
#include <memory>
#include "spacetime/Spacetime.h"

namespace caset {
void Spacetime::embedEuclidean(int dimensions = 4, double epsilon = 1e-8) {
  pybind11::gil_scoped_release no_gil;
  if (vertexList->size() == 0) return;
  if (edgeList->size() == 0) return;

  const int N = vertexList->size();
  const int E = edgeList->size();
  double lr = 10e-3;

  Edges edgeVector = edgeList->toVector();
  std::vector<std::shared_ptr<Vertex> > vertexVector = vertexList->toVector();

  if (vertexVector.empty()) {
    CLOG(WARN_LEVEL, "No vertices to embed!");
    return;
  }
  if (edgeVector.empty()) {
    CLOG(WARN_LEVEL, "No edges to embed!");
    return;
  }

  CLOG(INFO_LEVEL, "Embedding a ", dimensions, "-d Euclidean space with ", N, " vertices and ", E, " edges.");
  std::unordered_map<std::uint64_t, int> vertexIdToIndex;
  vertexIdToIndex.reserve(vertexVector.size());
  std::unordered_map<std::uint64_t, double> vertexIdToTime;
  vertexIdToTime.reserve(vertexVector.size());
  for (int i = 0; i < static_cast<int>(vertexVector.size()); ++i) {
    vertexIdToIndex[vertexVector[i]->getId()] = i;
    vertexIdToTime[vertexVector[i]->getId()] = vertexVector[i]->getTime();
  }

  std::vector<int64_t> edgeIdxToSourceIndex(E);
  std::vector<int64_t> edgeIdxToTargetIndex(E);
  std::vector<double> edgeIdxToSourceTime(E);
  std::vector<double> edgeIdxToTargetTime(E);
  std::vector<double> edgeIdxToAbsoluteSquaredLength(E);

  for (int e = 0; e < E; ++e) {
    const auto &edge = edgeVector[e];
    auto sourceIndexIterator = vertexIdToIndex.find(edge->getSourceId());
    auto targetIndexIterator = vertexIdToIndex.find(edge->getTargetId());
    if (sourceIndexIterator == vertexIdToIndex.end() || targetIndexIterator == vertexIdToIndex.end()) {
      throw std::runtime_error("Edge refers to unknown vertex id");
    }
    auto sourceTimeIterator = vertexIdToTime.find(edge->getSourceId());
    auto targetTimeIterator = vertexIdToTime.find(edge->getTargetId());

    edgeIdxToSourceIndex[e] = sourceIndexIterator->second;
    edgeIdxToTargetIndex[e] = targetIndexIterator->second;
    edgeIdxToSourceTime[e] = sourceTimeIterator->second;
    edgeIdxToTargetTime[e] = targetTimeIterator->second;

    double L = edge->getSquaredLength();
    // If you have Minkowski lengths and want magnitude-only, use std::abs(L).
    edgeIdxToAbsoluteSquaredLength[e] = std::abs(L)
                                          ? std::abs(L)
                                          : epsilon;
    // Avoid zero target distances which can cause issues in optimization;
  }

  auto edgeIdxToSourceIdxTensor = torch::from_blob(edgeIdxToSourceIndex.data(),
                                                   {E},
                                                   torch::TensorOptions().dtype(torch::kLong)).clone();
  auto edgeIdxToTargetIdxTensor = torch::from_blob(edgeIdxToTargetIndex.data(),
                                                   {E},
                                                   torch::TensorOptions().dtype(torch::kLong)).clone();
  auto edgeIdxToSourceTimeTensor = torch::from_blob(edgeIdxToSourceTime.data(),
                                                    {E},
                                                    torch::TensorOptions().dtype(torch::kDouble)).clone();
  auto edgeIdxToTargetTimeTensor = torch::from_blob(edgeIdxToTargetTime.data(),
                                                    {E},
                                                    torch::TensorOptions().dtype(torch::kDouble)).clone();

  auto edgeIdxToAbsoluteSquaredLengthTensor = torch::from_blob(edgeIdxToAbsoluteSquaredLength.data(),
                                                               {E},
                                                               torch::TensorOptions().dtype(torch::kDouble)).clone();

  // 4. Set up optimizer (Adam is simple and robust)
  torch::Tensor positions = torch::randn({N, dimensions},
                                         torch::TensorOptions()
                                         .dtype(torch::kDouble))
      .set_requires_grad(true);

  torch::Tensor vertexTimesTensor = torch::zeros(
    {N},
    torch::TensorOptions().dtype(torch::kDouble)
  );
  for (int i = 0; i < N; ++i) {
    vertexTimesTensor[i] = vertexVector[i]->getTime();
  }

  torch::optim::Adam optimizer({positions}, torch::optim::AdamOptions(lr));

  auto previousLoss = torch::tensor({0});
  auto loss = torch::tensor({0});
  auto iter = 0;
  auto epsilonTensor = torch::tensor({epsilon}, torch::TensorOptions().dtype(torch::kDouble));
  while (iter == 0 || ((loss - previousLoss).abs() > epsilonTensor).item<bool>()) {
    iter++;
    optimizer.zero_grad();

    // 5. Compute predicted squared distances for all edges
    auto srcPositions = positions.index_select(0, edgeIdxToSourceIdxTensor); // (E, dim)
    auto tgtPositions = positions.index_select(0, edgeIdxToTargetIdxTensor); // (E, dim)

    auto expectedSrcTimes = vertexTimesTensor.index_select(0, edgeIdxToSourceIdxTensor);
    auto expectedTgtTimes = vertexTimesTensor.index_select(0, edgeIdxToTargetIdxTensor);
    auto expectedTimes = (expectedSrcTimes + expectedTgtTimes) / 2.; // (E,)
    auto observedLengths = srcPositions - tgtPositions; // (E, dim - 1)

    auto sqdist = observedLengths.pow(2).sum(-1); // (E,)

    // The observed time is the 0th element of the coordinate vector
    auto observedSrcTimes = srcPositions.index({torch::arange(0, E), 0});
    auto observedTgtTimes = tgtPositions.index({torch::arange(0, E), 0});
    auto observedTimes = (observedSrcTimes + observedTgtTimes) / 2.; // (E,)

    auto sqtime = (observedTimes - expectedTimes).pow(2); // (E,)

    // 6. Loss: match squared distances
    auto residual = sqdist - edgeIdxToAbsoluteSquaredLengthTensor + (sqtime * dimensions);
    previousLoss = loss;
    loss = residual.pow(2).mean();

    loss.backward();
    optimizer.step();

    // Optional: early stopping / logging
    if (iter % 200 == 0) {
      std::cout << "[embedEuclidean] iter " << iter
          << " loss = " << loss.item<double>() << std::endl;
    }
  }

  // 7. Write back into Vertex coordinates
  auto posCpu = positions.detach().cpu();
  auto posAccessor = posCpu.accessor<double, 2>();

  for (int i = 0; i < N; ++i) {
    std::vector<double> coords(dimensions);
    coords[0] = vertexVector[i]->getTime();
    for (int d = 1; d < dimensions; ++d) {
      coords[d] = posAccessor[i][d];
    }
    vertexVector[i]->setCoordinates(coords);
  }
  CLOG(INFO_LEVEL,
       "Iteration: ",
       iter,
       " Loss: ",
       loss.item<double>(),
       " Previous Loss: ",
       previousLoss.item<double>());
}

void Spacetime::build(int numSimplices) {
  // TODO: Implement topologies instead of the default.
  return topology->build(this, numSimplices);
}

EdgePtr Spacetime::createEdge(
  const std::uint64_t src,
  const std::uint64_t tgt
) {
  EdgePtr edge = edgeList->add(src, tgt);
  vertexList->get(src)->addOutEdge(edge);
  vertexList->get(tgt)->addInEdge(edge);
  return edge;
}

EdgePtr Spacetime::createEdge(
  const std::uint64_t src,
  const std::uint64_t tgt,
  double squaredLength
) noexcept {
  EdgePtr edge = edgeList->add(src, tgt, squaredLength);
  vertexList->get(src)->addOutEdge(edge);
  vertexList->get(tgt)->addInEdge(edge);
  return edge;
}

SimplexPtr Spacetime::createSimplex(
  const Vertices &vertices,
  const Edges &edges
) {
  const SimplexOrientationPtr orientation = SimplexOrientation::orientationOf(vertices);

  SimplexPtr simplex = Simplex::create(vertices, edges);
  for (const auto &o : simplex->getOrientation()->getFacialOrientations()) {
    externalSimplices[o].insert(simplex);
  }
  return simplex;
}

SimplexPtr Spacetime::createSimplex(std::size_t k) {
  double squaredLength = alpha;
  Vertices vertices = {};
  vertices.reserve(k);
  Edges edges = {};
  edges.reserve(Simplex::computeNumberOfEdges(k));
  for (int i = 0; i < k; i++) {
    // Use coning to construct the vertex edges. For each new vertex; draw an edge to each existing vertex.
    VertexPtr newVertex = vertexList->add(vertexIdCounter++, {static_cast<double>(currentTime)});
    for (const auto &existingVertex : vertices) {
      EdgePtr edge = edgeList->add(existingVertex->getId(), newVertex->getId(), squaredLength);
      existingVertex->addOutEdge(edge);
      newVertex->addInEdge(edge);
      edges.push_back(edge);
    }
    vertices.push_back(newVertex);
  }
  return createSimplex(vertices, edges);
}

SimplexPtr Spacetime::createSimplex(const std::tuple<uint8_t, uint8_t> &numericOrientation) {
  double squaredLength = alpha;
  double timelikeSquaredLength = alpha;
  SimplexOrientationPtr orientation = std::make_shared<SimplexOrientation>(
    std::get<0>(numericOrientation),
    std::get<1>(numericOrientation));
  std::uint8_t k = orientation->getK();
  auto [ti, tf] = orientation->numeric();
  Vertices vertices = {};
  vertices.reserve(k);
  Edges edges = {};
  edges.reserve(Simplex::computeNumberOfEdges(k));
  for (int i = 0; i < ti; i++) {
    // Create ti Timelike vertices
    // Use coning to construct the vertex edges. For each new vertex; draw an edge to each existing vertex.
    VertexPtr newVertex = vertexList->add(vertexIdCounter++, {static_cast<double>(currentTime)});
    if (getMetric()->getSignature()->getSignatureType() == SignatureType::Lorentzian) {
      timelikeSquaredLength = -alpha;
    }
    for (const auto &existingVertex : vertices) {
      EdgePtr edge = edgeList->
          add(existingVertex->getId(), newVertex->getId(), timelikeSquaredLength);
      existingVertex->addOutEdge(edge);
      newVertex->addInEdge(edge);
      edges.push_back(edge);
    }
    vertices.push_back(newVertex);
  }
  for (int i = 0; i < tf; i++) {
    // Create ti Spacelike vertices
    // Use coning to construct the vertex edges. For each new vertex; draw an edge to each existing vertex.
    /// We can't just use the vertexList .size() here, because some vertices can be removed. We need to keep a
    /// counter:
    VertexPtr newVertex = vertexList->add(vertexIdCounter++, {static_cast<double>(currentTime + 1)});
    for (const auto &existingVertex : vertices) {
      EdgePtr edge;
      if (existingVertex->getTime() < newVertex->getTime()) {
        edge = edgeList->add(existingVertex->getId(), newVertex->getId(), squaredLength);
      } else {
        edge = edgeList->add(existingVertex->getId(), newVertex->getId(), timelikeSquaredLength);
      }
      existingVertex->addOutEdge(edge);
      newVertex->addInEdge(edge);
      edges.push_back(edge);
    }
    vertices.push_back(newVertex);
  }
  return createSimplex(vertices, edges);
}

[[nodiscard]] OptionalSimplexPair
Spacetime::getGluableFaces(const SimplexPtr &unattachedSimplex, const SimplexPtr &attachedSimplex) {
  CLOG(DEBUG_LEVEL, "Unattached simplex: ", unattachedSimplex->toString(), "\nAttached Simplex: ", attachedSimplex->toString());
  const auto orientations = unattachedSimplex->getGluableFaceOrientations();
  CLOG(DEBUG_LEVEL, "Got ", std::to_string(orientations.size()), " non-degenerate orientations");
  for (const auto &orientation : orientations) {
    CLOG(DEBUG_LEVEL, " Orientation ", orientation->toString());
    auto unattachedFacets = unattachedSimplex->getAvailableFacetsByOrientation(orientation);
    auto attachedFacets = attachedSimplex->getAvailableFacetsByOrientation(orientation);
    CLOG(INFO_LEVEL, "Got ", unattachedFacets.size(), " unattached facets and ", attachedFacets.size(), " attached facets.");

#if CASET_DEBUG
    for (const auto &f : unattachedFacets) {
      f->validate();
      if (!f->isCausallyAvailable()) throw std::runtime_error("Facet wasn't causally available");
    }
    for (const auto &f : attachedFacets) {
      f->validate();
      if (!f->isCausallyAvailable()) throw std::runtime_error("Facet wasn't causally available");
    }
#endif

    if (unattachedFacets.empty() || attachedFacets.empty()) continue;

    return std::make_optional(std::make_pair(*unattachedFacets.begin(), *attachedFacets.begin()));
  }
  return std::nullopt;
}

void Spacetime::moveInEdgesFromVertex(const VertexPtr &from, const VertexPtr &to) {
  for (const auto &edge : from->getInEdges()) {
    // The source is external to the face/simplex, the `from` node is going to be going away.
    const VertexPtr originalSource = vertexList->get(edge->getSourceId());
    originalSource->removeOutEdge(edge);
    from->removeInEdge(edge);
    edgeList->remove(edge);
    edge->replaceTargetVertex(to->getId());
    const EdgePtr newEdge = edgeList->add(edge);
    to->addInEdge(newEdge);
    originalSource->addOutEdge(newEdge);
  }
}

void Spacetime::moveOutEdgesFromVertex(const VertexPtr &from, const VertexPtr &to) {
  for (const auto &edge : from->getOutEdges()) {
    const VertexPtr originalTarget = vertexList->get(edge->getTargetId());
    originalTarget->removeInEdge(edge);
    from->removeOutEdge(edge);
    edgeList->remove(edge);
    edge->replaceSourceVertex(to->getId());
    const EdgePtr newEdge = edgeList->add(edge);
    to->addOutEdge(newEdge);
    originalTarget->addInEdge(newEdge);
  }
}

SimplexSet Spacetime::getSimplicesWithOrientation(std::tuple<uint8_t, uint8_t> orientation) {
  SimplexOrientationPtr o = std::make_shared<
    SimplexOrientation>(std::get<0>(orientation), std::get<1>(orientation));
  SimplexSet result{};
  for (const auto &bucket : externalSimplices | std::views::values) {
    for (const auto &simplex : bucket) {
      for (const auto &simplexFacialOrientation : simplex->getOrientation()->getFacialOrientations()) {
        if (simplex->getOrientation() == o) result.insert(simplex);
      }
    }
  }
  return result;
}

/// When we attach two simplices; the "attached" one is assumed to be part of a simplicial complex. The "unattached" one
/// is assumed to be part of another simplicial complex, but usually by itself. The "attached" simplex replaces
/// corresponding vertices in the "unattached" simplex with it's own vertices. Same goes for the _internal_ edges. Any
/// external edges in "unattached" are redirected from those vertices on "unattached" to the corresponding vertex in
/// "attached".
void Spacetime::attachAtVertices(
  const SimplexPtr &unattached,
  const SimplexPtr &attached,
  const std::vector<std::pair<VertexPtr, VertexPtr> > &vertexPairs // {unattached, attached}
) {
  CLOG(INFO_LEVEL, "attachAtVertices called. Pre-validating.");
  // Bone density in Regge calculus can be calculated as the size of the Simplex list on the Edge.
#if CASET_DEBUG
  unattached->validate();
  attached->validate();
#endif
  // Move external edges from unattached vertices to attached vertices.
  for (const auto &[unattachedVertex, attachedVertex] : vertexPairs) {
    unattached->attach(unattachedVertex, attachedVertex, edgeList, vertexList);
  }
#if CASET_DEBUG
  unattached->validate();
  attached->validate();
#endif
}

/// When this method is called it's assumed that faces are causally available and of the correct orientation, and do
/// not share a coface. When these things are true there should also exist a compatible vertex order, so we do not
/// check that unless CASET_DEBUG is set.
std::tuple<SimplexPtr, bool> Spacetime::causallyAttachFaces(
  const SimplexPtr &attachedFace,
  const SimplexPtr &unattachedFace
) {
#if CASET_DEBUG
  if (!attachedFace->isCausallyAvailable() || !unattachedFace->isCausallyAvailable()) {
    CLOG(ERROR_LEVEL,
         "One or more of attachedFace and unattachedFace was not causally available!\n",
         attachedFace->toString(),
         "\n",
         unattachedFace->toString());
    return {attachedFace, false};
  }
  if (attachedFace->fingerprint.fingerprint() == unattachedFace->fingerprint.fingerprint()) {
    CLOG(ERROR_LEVEL, "Faces are already attached!");
    return {attachedFace, false};
  }
  if (attachedFace->getOrientation() != unattachedFace->getOrientation()) {
    CLOG(ERROR_LEVEL,
         "Faces have different orientations: ",
         attachedFace->getOrientation()->toString(),
         " vs ",
         unattachedFace->getOrientation()->toString());
    return {attachedFace, false};
  }
  for (const auto &attachedCoface : attachedFace->getCofaces()) {
    for (const auto &unattachedCoface : unattachedFace->getCofaces()) {
      if (attachedCoface->fingerprint.fingerprint() == unattachedCoface->fingerprint.fingerprint()) {
        CLOG(ERROR_LEVEL, "Faces share a coface! (they are already attached.)");
        return {attachedFace, false};
      }
    }
  }
#endif

  Vertices vertices{};
  vertices.reserve(attachedFace->size());
  Edges edges{};
  edges.reserve(attachedFace->size());

  // Two vertices are compatible to attach iff they share the same time value.
  std::vector<std::pair<VertexPtr, VertexPtr> > vertexPairs{};
  vertexPairs.reserve(attachedFace->size());

  // These are in order of traversal, you can iterate them to walk the Face:
  const auto &unattachedVertices = unattachedFace->getVertices();

  // myVertices and yourVertices should have a sequence that lines up, but they're not necessarily at the correct
  // starting node. We should shuffle through until they are either compatible or we've tried all possible orders.
  const std::optional<Vertices> attachedOrderedVerticesOptional = attachedFace->getVerticesWithParityTo(unattachedFace);

#if CASET_DEBUG
  if (!attachedOrderedVerticesOptional.has_value()) {
    CLOG(WARN_LEVEL,
         "No compatible vertex order found for myFace and yourFace.\n",
         attachedFace->toString(),
         "\n",
         unattachedFace->toString());
    return {nullptr, false};
  }
#endif

  const Vertices &attachedOrderedVertices = attachedOrderedVerticesOptional.value();
  for (auto i = 0; i < attachedOrderedVertices.size(); i++) {
    std::pair<VertexPtr, VertexPtr> vp = std::make_pair(unattachedVertices[i], attachedOrderedVertices[i]);
    vertexPairs.push_back(vp);
  }

  for (const auto &facialOrientation : attachedFace->getOrientation()->getFacialOrientations()) {
    externalSimplices[facialOrientation].erase(attachedFace);
  }

  attachAtVertices(unattachedFace, attachedFace, vertexPairs);

  if (!unattachedFace->getCofaces().empty()) {
    for (const auto &newCoface : unattachedFace->getCofaces()) {
      attachedFace->addCoface(newCoface);
    }
  }

  if (!attachedFace->isCausallyAvailable()) {
    internalSimplices[attachedFace->getOrientation()].insert(attachedFace);
    attachedFace->markAsUnavailable();
  }
  if (!unattachedFace->isCausallyAvailable()) {
    unattachedFace->markAsUnavailable();
  }

  // TODO: Mark facet as unavailable on it's cofaces. I think this applies for a single causal attachment. Does a single causal attachment make it unavailable?
  //  Simplex tracks available facets now, see Simplex::availableFacetsByOrientation
  //  call Simplex::markAsUnavailable on the coface, not the facet. Maybe we should actually call that on the Facet, it's
  //  a bit more idiomatic.


  return {attachedFace, true};
}

OptionalSimplexPair Spacetime::chooseSimplexFacesToGlue(const SimplexPtr &unattachedSimplex) {
#if CASET_DEBUG
  if (!unattachedSimplex->hasCausallyAvailableFacet()) {
    CLOG(WARN_LEVEL, "Unattached simplex had no causally available facets.");
    return std::nullopt;
  }
  auto numOrientation = unattachedSimplex->getGluableFaceOrientations();
  CLOG(INFO_LEVEL, "Found ", std::to_string(numOrientation.size()), " gluable facial orientations");
#endif

  for (const auto &facialOrientation : unattachedSimplex->getGluableFaceOrientations()) {
    const auto &prospectiveCofaces = externalSimplices[facialOrientation];
    CLOG(INFO_LEVEL, "Found ", prospectiveCofaces.size(), " prospective cofaces");
    if (prospectiveCofaces.empty()) continue;
    for (
      auto attachedCofaceId = prospectiveCofaces.begin();
      attachedCofaceId != prospectiveCofaces.end();
      ++attachedCofaceId
      ) {
      if ((*attachedCofaceId)->fingerprint.fingerprint() == unattachedSimplex->fingerprint.fingerprint()) {
        CLOG(INFO_LEVEL, "Unattached matched attach3d. Continuing.");
        continue;
      }
      if (!(*attachedCofaceId)->hasCausallyAvailableFacet()) {
        CLOG(INFO_LEVEL, "Attached coface had no causally available facets!");
        continue;
      }
#if CASET_DEBUG
      (*attachedCofaceId)->validate();
#endif
      OptionalSimplexPair gluablePair = getGluableFaces(unattachedSimplex, *attachedCofaceId);
      if (gluablePair.has_value()) {
        CLOG(INFO_LEVEL, "Found a gluable pair.");
        const auto &[unattachedFace, attachedFace] = gluablePair.value();
        return gluablePair;
      } else {
        CLOG(INFO_LEVEL, "No gluable pair found");
      }
    }
  }
  CLOG(INFO_LEVEL, "Returning None");
  return std::nullopt;
}

SimplexSet Spacetime::getExternalSimplices() noexcept {
  SimplexSet simplices{};
  for (const auto &[facialOrientation, bucket] : externalSimplices) {
    for (const auto &simplex : bucket) {
      simplices.insert(simplex);
    }
  }
  return simplices;
}

std::vector<Vertices> Spacetime::getConnectedComponents() const {
  VertexSet seen{};
  std::vector<Vertices> components{};
  for (const auto &vertex : vertexList->toVector()) {
    if (seen.contains(vertex)) {
      continue;
    }
    Vertices component{};
    Vertices stack{vertex};
    while (!stack.empty()) {
      VertexPtr current = stack.back();
      stack.pop_back();
      if (seen.contains(current)) {
        continue;
      }
      seen.insert(current);
      component.push_back(current);
      for (const auto &edge : current->getOutEdges()) {
        VertexPtr neighbor = vertexList->get(edge->getTargetId());
        if (neighbor != nullptr && !seen.contains(neighbor)) {
          stack.push_back(neighbor);
        }
      }
      for (const auto &edge : current->getInEdges()) {
        VertexPtr neighbor = vertexList->get(edge->getSourceId());
        if (neighbor != nullptr && !seen.contains(neighbor)) {
          stack.push_back(neighbor);
        }
      }
    }
    components.push_back(component);
  }
  return components;
}

VertexPtr Spacetime::createVertex(const std::uint64_t id) noexcept {
  return vertexList->add(id);
}

VertexPtr Spacetime::createVertex(const std::uint64_t id, const std::vector<double> &coords) noexcept {
  return vertexList->add(id, coords);
}

bool Spacetime::removeIfIsolated(const VertexPtr &vertex) {
  if (vertex->degree() == 0) {
    CLOG(DEBUG_LEVEL, "Removing vertex: ", vertex->toString());
    vertexList->remove(vertex);
    return true;
  }
  CLOG(DEBUG_LEVEL, "NOT Removing vertex: ", vertex->toString());
  return false;
}
} // caset
