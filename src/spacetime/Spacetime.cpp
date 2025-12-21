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
#include "SimplexOrientation.h"
#include "ForwardDeclarations.h"
#include "SimplexGluer.h"
#include "EdgeList.h"
#include "Edge.h"

namespace caset {

// ========================================
// Constructors
// ========================================

Spacetime::Spacetime() {
  Signature signature(4, SignatureType::Lorentzian);
  metric = std::make_shared<Metric>(true, signature);
  spacetimeType = SpacetimeType::CDT;
  alpha = 1.;
  topology = std::make_shared<Toroid>();
  foliation = Foliation::PREFERRED;
}

Spacetime::Spacetime(
  std::shared_ptr<Metric> metric_,
  const SpacetimeType spacetimeType_,
  const Foliation foliation_,
  std::optional<double> alpha_,
  std::optional<std::shared_ptr<Topology> > topology_) : metric(metric_), spacetimeType(spacetimeType_), foliation(foliation_) {
  alpha = alpha_.value_or(1.);
  topology = topology_.value_or(std::make_shared<Toroid>());
}

// ========================================
// Creation Methods
// ========================================

std::pair<SimplexPtr, bool> Spacetime::createSimplex(
  const VertexPtrs &vertices,
  const Edges &edges
) {
  std::vector<IdType> ids{};
  ids.reserve(vertices.size());
  for (const auto &v : vertices) {
    ids.push_back(v->getId());
  }
  Fingerprint fp(ids);
  const auto found = simplices.find(fp.fingerprint());
  if (found == simplices.end()) {
#ifdef CASET_ASSERTIONS
    std::unordered_set<std::uint64_t> seen{};
    for (const auto &s : simplices) {
      if (seen.contains(s->fingerprint.fingerprint())) {
        CLOG(CRITICAL_LEVEL, "attempted to create a new simplex with the same fingerprint as an existing one!");
        CLOG(CRITICAL_LEVEL, "Attempted vertices: ");
        for (const auto &v : vertices) {
          CLOG(CRITICAL_LEVEL, "    - ", v->toString());
        }
        CLOG(CRITICAL_LEVEL,
             "Existing simplex: ",
             s->toString(),
             " with fingerprint ",
             std::to_string(s->fingerprint.fingerprint()),
             " vs ",
             std::to_string(fp.fingerprint()));
        throw std::runtime_error("Duplicate simplex: " + s->toString());
      }
      seen.insert(s->fingerprint.fingerprint());
    }
#endif
    SimplexPtr simplex = Simplex::create(this, vertices, edges);
    registerSimplex(simplex, false);
    return {simplex, true};
  }
#ifdef CASET_ASSERTIONS
  CLOG(CRITICAL_LEVEL, "You attempted to create a simplex that already exists: ", (*found)->toString());
#endif
  return {*found, false};
}

std::pair<SimplexPtr, bool> Spacetime::createSimplex(const std::tuple<uint8_t, uint8_t> &numericOrientation) {
  double squaredLength = alpha;
  double timelikeSquaredLength = alpha;
  SimplexOrientation orientation = {
    std::get<0>(numericOrientation),
    std::get<1>(numericOrientation)
  };
  std::uint8_t k = orientation.getK();
  auto [ti, tf] = orientation.numeric();
  VertexPtrs vertices = {};
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
          add(existingVertex, newVertex, timelikeSquaredLength);
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
        edge = edgeList->add(existingVertex, newVertex, squaredLength);
      } else {
        edge = edgeList->add(existingVertex, newVertex, timelikeSquaredLength);
      }
      existingVertex->addOutEdge(edge);
      newVertex->addInEdge(edge);
      edges.push_back(edge);
    }
    vertices.push_back(newVertex);
  }
  return createSimplex(vertices, edges);
}

std::pair<SimplexPtr, bool> Spacetime::createSimplex(std::size_t k) {
  double squaredLength = alpha;
  VertexPtrs vertices = {};
  vertices.reserve(k);
  Edges edges = {};
  edges.reserve(Simplex::computeNumberOfEdges(k));
  for (int i = 0; i < k; i++) {
    // Use coning to construct the vertex edges. For each new vertex; draw an edge to each existing vertex.
    VertexPtr newVertex = vertexList->add(vertexIdCounter++, {static_cast<double>(currentTime)});
    for (const auto &existingVertex : vertices) {
      EdgePtr edge = edgeList->add(existingVertex, newVertex, squaredLength);
      existingVertex->addOutEdge(edge);
      newVertex->addInEdge(edge);
      edges.push_back(edge);
    }
    vertices.push_back(newVertex);
  }
  return createSimplex(vertices, edges);
}

VertexPtr Spacetime::createVertex(const std::uint64_t id) noexcept {
  return vertexList->add(id);
}

VertexPtr Spacetime::createVertex(const std::uint64_t id, const std::vector<double> &coords) noexcept {
  return vertexList->add(id, coords);
}

EdgePtr Spacetime::createEdge(
  const VertexPtr &src,
  const VertexPtr &tgt
) const {
  EdgePtr edge = edgeList->add(src, tgt);
  src->addOutEdge(edge);
  tgt->addInEdge(edge);
  return edge;
}

EdgePtr Spacetime::createEdge(
  const VertexPtr &src,
  const VertexPtr &tgt,
  double squaredLength
) const noexcept {
  EdgePtr edge = edgeList->add(src, tgt, squaredLength);
  src->addOutEdge(edge);
  tgt->addInEdge(edge);
  return edge;
}

// ========================================
// Complex Building Methods
// ========================================

void Spacetime::build(int numSimplices) {
  // TODO: Implement topologies instead of the default.
  return topology->build(this, numSimplices);
}

void Spacetime::reserve(int numSimplices) {
  simplices.reserve(numSimplices + simplices.size());
  edgeList->reserve(numSimplices * Simplex::computeNumberOfEdges(4)); // TODO: Change this to dimensions.
  vertexList->reserve(numSimplices * 4 + 1);  // A k-simplex has k+1 vertices.
}


[[nodiscard]] OptionalSimplexPtrPair
Spacetime::getGluableFaces(const SimplexPtr &unattachedSimplex, const SimplexPtr &attachedSimplex) {
  auto unattachedFacets = unattachedSimplex->getFacets(); // vector<shared_ptr<Simplex>>
  auto attachedFacets = attachedSimplex->getFacets();
#if CASET_ASSERTIONS
  for (const auto &f : unattachedFacets) {
    f->validate();
  }
  for (const auto &f : attachedFacets) {
    f->validate();
  }
#endif

  for (auto &unattachedFace : unattachedFacets) {
    const auto [tia, tfa] = unattachedFace->getOrientation().numeric();
    if (tia == 0 || tfa == 0) continue; // Skip degenerate faces
    if (!unattachedFace->isCausallyAvailableFace()) continue;
    for (auto &attachedFace : attachedFacets) {
      if (unattachedFace->isTimelike() != attachedFace->isTimelike()) continue;
      // Skip faces that don't match in timelikeness
      const auto [tib, tfb] = attachedFace->getOrientation().numeric();
      if (tib == 0 || tfb == 0) continue; // Skip degenerate faces
      if (!attachedFace->isCausallyAvailableFace()) continue;
#if CASET_ASSERTIONS
      attachedFace->validate();
      unattachedFace->validate();
#endif
      if (tia == tfb && tfa == tib) return std::make_optional(std::make_pair(unattachedFace, attachedFace));
      if (tia == tib && tfa == tfb) return std::make_optional(std::make_pair(unattachedFace, attachedFace));
    }
  }
  return std::nullopt;
}

void Spacetime::replaceVertices(
  const VertexPtrs &oldVertices, const VertexPtrs &replacements
) {
  // Use RAII (resource acquisition is initialization) to detach the 'unattached' simplex from all its (hashtable)
  // references so that it's free to modify.
  {
    SimplexGluer gluer{this, oldVertices};
    // Move external edges from unattached vertices to attached vertices.
    for (int i = 0; i < oldVertices.size(); i++) {
      const auto &oldVertex = oldVertices[i];
      const auto &replacementVertex = replacements[i];
      auto [oldEdges, newEdges] = oldVertex->moveEdgesTo(
        replacementVertex, this);
      for (const auto &simplex : gluer.getSimplicesToProcess()) {
        simplex->replaceVertex(oldVertex, replacementVertex);
      }
      if (oldVertex->degree() == 0) vertexList->remove(oldVertex);
    }
  }  // ~SimplexGluer() called here.
}

std::tuple<SimplexPtr, bool> Spacetime::causallyAttachFaces(
  const SimplexPtr &attachedFace,
  const SimplexPtr &unattachedFace
) {
  // Fast path: early exit checks
  if (!attachedFace->isCausallyAvailableFace() || !unattachedFace->isCausallyAvailableFace()) {
    CLOG(ERROR_LEVEL,
         "One or more of attachedFace and unattachedFace was not causally available!\n",
         attachedFace->toString(),
         "\n",
         unattachedFace->toString());
    return {attachedFace, false};
  }

  // Cache fingerprints to avoid repeated calls
  const auto attachedFp = attachedFace->fingerprint.fingerprint();
  const auto unattachedFp = unattachedFace->fingerprint.fingerprint();

  if (attachedFp == unattachedFp) {
    CLOG(ERROR_LEVEL, "Faces are already attached!");
    return {attachedFace, false};
  }
  if (attachedFace->getOrientation() != unattachedFace->getOrientation()) {
    CLOG(ERROR_LEVEL,
         "Faces have different orientations: ",
         attachedFace->getOrientation().toString(),
         " vs ",
         unattachedFace->getOrientation().toString());
    return {attachedFace, false};
  }

#ifdef CASET_ASSERTIONS
  // Check for shared cofaces (already attached)
  const auto &attachedCofaces = attachedFace->getCofaces();
  const auto &unattachedCofaces = unattachedFace->getCofaces();
  for (const auto &attachedCoface : attachedCofaces) {
    const auto attachedCofaceFp = attachedCoface->fingerprint.fingerprint();
    for (const auto &unattachedCoface : unattachedCofaces) {
      if (attachedCofaceFp == unattachedCoface->fingerprint.fingerprint()) {
        CLOG(ERROR_LEVEL, "Faces share a coface! (they are already attached.)");
        return {attachedFace, false};
      }
    }
  }
#endif

  // Get vertex ordering for attachment
  const auto attachedOrderedVerticesOptional = attachedFace->getVerticesWithParityTo(unattachedFace);

  if (!attachedOrderedVerticesOptional.has_value()) {
    CLOG(WARN_LEVEL,
         "No compatible vertex order found for myFace and yourFace.\n",
         attachedFace->toString(),
         "\n",
         unattachedFace->toString());
    return {nullptr, false};
  }

  const VertexPtrs &attachedOrderedVertices = attachedOrderedVerticesOptional.value();
  const auto &unattachedVertices = unattachedFace->getVertices();

  // Perform the vertex replacement
  replaceVertices(unattachedVertices, attachedOrderedVertices);

  // OPTIMIZATION: Iterate cofaces directly without copying, and add bidirectionally
#ifndef CASET_ASSERTIONS
  // In release mode, get references directly
  const auto &attachedCofaces = attachedFace->getCofaces();
  const auto &unattachedCofaces = unattachedFace->getCofaces();
#endif

  for (const auto &c : attachedCofaces) {
    unattachedFace->addCoface(c);
  }
  for (const auto &c : unattachedCofaces) {
    attachedFace->addCoface(c);
  }

  return {attachedFace, true};
}

std::tuple<SimplexPtr, bool> Spacetime::glue(const SimplexPtr &simplex) {
  OptionalSimplexPtrPair leftFaceRightFace = chooseSimplexFacesToGlue(simplex);
  if (!leftFaceRightFace.has_value()) return {nullptr, false};
  const auto &[leftFace, rightFace] = leftFaceRightFace.value();
  return causallyAttachFaces(leftFace, rightFace);
}

OptionalSimplexPtrPair Spacetime::chooseSimplexFacesToGlue(const SimplexPtr &unattachedSimplex) {
  // OPTIMIZATION: Cache loop-invariant checks
  if (!unattachedSimplex->hasCausallyAvailableFacet()) {
    return std::nullopt;  // Early exit if no available facets
  }

  const auto unattachedFp = unattachedSimplex->fingerprint.fingerprint();
  for (const auto &facialOrientation : unattachedSimplex->getGluableFaceOrientations()) {
    const auto &prospectiveCofaces = externalSimplicesByFacialOrientation[facialOrientation];
    if (prospectiveCofaces.empty()) continue;
    for (const auto &attachedCoface : prospectiveCofaces) {
      // OPTIMIZATION: Cache fingerprint to avoid repeated member access
      if (attachedCoface->fingerprint.fingerprint() == unattachedFp) continue;
      if (!attachedCoface->hasCausallyAvailableFacet()) continue;
#if CASET_ASSERTIONS
      attachedCoface->validate();
#endif

      OptionalSimplexPtrPair gluablePair = getGluableFaces(unattachedSimplex, attachedCoface);
      if (gluablePair.has_value()) {
        return gluablePair;  // Return directly, no need to destructure
      }
    }
  }
  return std::nullopt;
}

// ========================================
// Query Methods
// ========================================

SpacetimeType Spacetime::getSpacetimeType() const noexcept {
  return spacetimeType;
}

double Spacetime::getCurrentTime() const noexcept {
  return static_cast<double>(currentTime);
}

std::shared_ptr<EdgeList> Spacetime::getEdgeList() const noexcept {
  return edgeList;
}

std::shared_ptr<Metric> Spacetime::getMetric() const noexcept {
  return metric;
}

std::shared_ptr<VertexList> Spacetime::getVertexList() const noexcept {
  return vertexList;
}

SimplexSet Spacetime::getExternalSimplices() noexcept {
  SimplexSet simplices_{};
  for (const auto &bucket : externalSimplicesByFacialOrientation | std::views::values) {
    for (const auto &simplex : bucket) {
      simplices_.insert(simplex);
    }
  }
  return simplices_;
}

// I think we're going to want to implement Qhull https://www.qhull.org for delaunay triangulations because we can't
// build the simplicial complex very easily just by attaching new simplices. We would have to expand from the complex in
// a manner similar to coning, where we treat e.g. a (1, 3) face or the corner made by two (1, 3) faces as a new
// attachment site, then we build off it by coning one new vertex and adding the Simplex abstraction on those two facets
// plus the new vertex.
throw std::runtime_error("Implement BFS on the boundary of the simplicial complex by using a deque for those facets that remain available (have fewer than 2 cofaces).")

SimplexSet Spacetime::getSimplicesWithOrientation(std::tuple<uint8_t, uint8_t> orientation) {
  SimplexOrientation o{std::get<0>(orientation), std::get<1>(orientation)};
  SimplexSet result{};
  for (const auto &bucket : externalSimplicesByFacialOrientation | std::views::values) {
    for (const auto &simplex : bucket) {
      if (simplex->getOrientation() == o) result.insert(simplex);
    }
  }
  return result;
}

std::vector<VertexPtrs> Spacetime::getConnectedComponents() const {
  VertexPtrSet seen{};
  std::vector<VertexPtrs> components{};
  for (auto vertex : vertexList->toVector()) {
    if (seen.contains(vertex)) {
      continue;
    }
    VertexPtrs component{};
    VertexPtrs stack{vertex};
    while (!stack.empty()) {
      VertexPtr current = stack.back();
      stack.pop_back();
      if (seen.contains(current)) {
        continue;
      }
      seen.insert(current);
      component.push_back(current);
      for (const auto &edge : current->getOutEdges()) {
        VertexPtr neighbor = vertexList->get(edge->getTarget()->getId());
        if (neighbor != nullptr && !seen.contains(neighbor)) {
          stack.push_back(neighbor);
        }
      }
      for (const auto &edge : current->getInEdges()) {
        VertexPtr neighbor = vertexList->get(edge->getSource()->getId());
        if (neighbor != nullptr && !seen.contains(neighbor)) {
          stack.push_back(neighbor);
        }
      }
    }
    components.push_back(component);
  }
  return components;
}

SimplexPtr Spacetime::getSimplex(SimplexPtr simplex) const {
  auto it = simplices.find(simplex);
  if (it == simplices.end()) {
    return nullptr;
  }
  return *it;
}

SimplexPtr Spacetime::getSimplex(std::uint64_t fingerprint) const {
  auto it = simplices.find(fingerprint);
  if (it == simplices.end()) {
    return nullptr;
  }
  return *it;
}

// ========================================
// Manipulation & Helper Methods
// ========================================

double Spacetime::incrementTime() noexcept {
  currentTime++;
  return static_cast<double>(currentTime);
}

void Spacetime::moveInEdgesFromVertex(const VertexPtr &from, const VertexPtr &to) {
  for (const auto &edge : from->getInEdges()) {
    // The source is external to the face/simplex, the `from` node is going to be going away.
    const VertexPtr originalSource = vertexList->get(edge->getSource()->getId());
    originalSource->removeOutEdge(edge);
    from->removeInEdge(edge);
    edgeList->remove(edge);
    edge->replaceTargetVertex(to);
    const EdgePtr newEdge = edgeList->add(edge);
    to->addInEdge(newEdge);
    originalSource->addOutEdge(newEdge);
  }
}

void Spacetime::moveOutEdgesFromVertex(const VertexPtr &from, const VertexPtr &to) {
  for (const auto &edge : from->getOutEdges()) {
    const VertexPtr originalTarget = vertexList->get(edge->getTarget()->getId());
    originalTarget->removeInEdge(edge);
    from->removeOutEdge(edge);
    edgeList->remove(edge);
    edge->replaceSourceVertex(to);
    const EdgePtr newEdge = edgeList->add(edge);
    to->addOutEdge(newEdge);
    originalTarget->addInEdge(newEdge);
  }
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

void Spacetime::embedEuclidean(int dimensions, double epsilon) {
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
    auto sourceIndexIterator = vertexIdToIndex.find(edge->getSource()->getId());
    auto targetIndexIterator = vertexIdToIndex.find(edge->getTarget()->getId());
    if (sourceIndexIterator == vertexIdToIndex.end() || targetIndexIterator == vertexIdToIndex.end()) {
      throw std::runtime_error("Edge refers to unknown vertex id");
    }
    auto sourceTimeIterator = vertexIdToTime.find(edge->getSource()->getId());
    auto targetTimeIterator = vertexIdToTime.find(edge->getTarget()->getId());

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

void Spacetime::addObservable(const std::shared_ptr<Observable> &observable) {
  observables.push_back(observable);
}

// ========================================
// Internal Management
// ========================================

SimplexPtr Spacetime::registerSimplex(const SimplexPtr &simplex, bool internal) {
#ifdef CASET_ASSERTIONS
  std::unordered_set<std::uint64_t> seen{};
  for (const auto &simp : simplices) {
    if (seen.contains(simp->fingerprint.fingerprint())) {
      CLOG(CRITICAL_LEVEL, "Duplicate simplex!");
      throw std::runtime_error("Duplicate simplex!");
    }
    seen.insert(simp->fingerprint.fingerprint());
  }
#endif
  const auto &[it, inserted] = simplices.emplace(simplex);
  if (!inserted) {
    return *it;
  }
  if (internal) {
    internalSimplicesByOrientation[simplex->getOrientation()].emplace(*it);
  } else {
    for (const auto &orientation : simplex->getOrientation().getFacialOrientations()) {
      externalSimplicesByFacialOrientation[orientation].emplace(*it);
    }
  }
#ifdef CASET_ASSERTIONS
  std::unordered_set<std::uint64_t> seen2{};
  for (const auto &simp : simplices) {
    if (seen2.contains(simp->fingerprint.fingerprint())) {
      CLOG(CRITICAL_LEVEL, "Duplicate simplex!");
      throw std::runtime_error("Duplicate simplex!");
    }
    seen2.insert(simp->fingerprint.fingerprint());
  }
#endif
  return *it;
}

void Spacetime::unregisterSimplex(const SimplexPtr &simplex) {
  if (!simplices.contains(simplex)) {
#ifdef CASET_ASSERTIONS
    CLOG(CRITICAL_LEVEL, "You attempted to unregister a simplex that does not exist! ", simplex->toString(), " existing simplices are: ");
    for (const auto &s : simplices) {
      CLOG(CRITICAL_LEVEL, "    - ", s->toString());
    }
    for (const auto &s : simplices) {
      if (s->fingerprint.fingerprint() == simplex->fingerprint.fingerprint()) {
        CLOG(CRITICAL_LEVEL, "Hash table said a simplex was not registered, but one was found!");
        throw std::runtime_error("registered simplex unexpectedly found. hash table corrupted.");
      }
    }
#endif
    return;
  }
  const auto orientation = simplex->getOrientation();
  internalSimplicesByOrientation[orientation].erase(simplex);
  for (const auto &o : orientation.getFacialOrientations()) {
    externalSimplicesByFacialOrientation[o].erase(simplex);
  }
  simplices.erase(simplex);
}

} // caset
