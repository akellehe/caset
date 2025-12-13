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
#include "utils.h"
#include <memory>
#include "spacetime/Spacetime.h"

namespace caset {

void Spacetime::build(int numSimplices) {
  // TODO: Implement topologies instead of the default.
  // return topology->build(this);
  std::vector<std::tuple<uint8_t, uint8_t> > orientations = {{1, 2}, {2, 1}};
  createSimplex(orientations[1]);
  for (int i = 0; i < numSimplices; i++) {
    SimplexRawPtr rightSimplex = createSimplex(orientations[i % 2]);
    OptionalSimplexPtrPair leftFaceRightFace = chooseSimplexFacesToGlue(rightSimplex);
    if (!leftFaceRightFace.has_value()) return;
    auto [leftFace, rightFace] = leftFaceRightFace.value();
    auto [left, succeeded] = causallyAttachFaces(leftFace, rightFace);
  }
}

EdgeRawPtr Spacetime::createEdge(
  VertexPtr src,
  VertexPtr tgt
) {
  EdgeRawPtr edge = edgeList->add(src, tgt);
  src->addOutEdge(edge);
  tgt->addInEdge(edge);
  return edge;
}

EdgeRawPtr Spacetime::createEdge(
  VertexPtr src,
  VertexPtr tgt,
  double squaredLength
) noexcept {
  EdgeRawPtr edge = edgeList->add(src, tgt, squaredLength);
  src->addOutEdge(edge);
  tgt->addInEdge(edge);
  return edge;
}

SimplexRawPtr Spacetime::createSimplex(
  const VertexPtrs &vertices, const Edges &edges
) {
  CLOG(INFO_LEVEL, "Creating Orientation...");
  const SimplexOrientationPtr orientation = SimplexOrientation::orientationOf(vertices);
  CLOG(INFO_LEVEL, "Creating simplex...");
  std::unique_ptr<Simplex> simplex = Simplex::create(vertices, edges);
  CLOG(INFO_LEVEL, "Getting raw pointer....");
  auto raw = simplex.get();
  for (const auto &o : raw->getOrientation()->getFacialOrientations()) {
    externalSimplices[o].insert(raw);
  }
  auto [it, _] = simplices.insert(std::move(simplex));
  return it->get();
}

py::object Spacetime::createSimplexForPython(const VertexPtrs &vertices, const Edges &edges) {
  return wrap_non_owning(createSimplex(vertices, edges));
}

py::object Spacetime::createSimplexForPython(const std::tuple<uint8_t, uint8_t> &numericOrientation) {
  return wrap_non_owning(createSimplex(numericOrientation));
}

py::object Spacetime::createSimplexForPython(std::size_t k) {
  return wrap_non_owning(createSimplex(k));
}

SimplexRawPtr Spacetime::createSimplex(std::size_t k) {
  double squaredLength = alpha;
  VertexPtrs vertices = {};
  vertices.reserve(k);
  Edges edges = {};
  edges.reserve(Simplex::computeNumberOfEdges(k));
  for (int i = 0; i < k; i++) {
    // Use coning to construct the vertex edges. For each new vertex; draw an edge to each existing vertex.
    VertexPtr newVertex = vertexList->add(vertexIdCounter++, {static_cast<double>(currentTime)});
    CLOG(INFO_LEVEL, "Made vertex id=", newVertex->getId(), " (counter now ", vertexIdCounter, ")");
    for (const auto &existingVertex : vertices) {
      EdgeRawPtr edge = edgeList->add(existingVertex, newVertex, squaredLength);
      existingVertex->addOutEdge(edge);
      newVertex->addInEdge(edge);
      edges.push_back(edge);
    }
    vertices.push_back(newVertex);
  }
  return createSimplex(vertices, edges);
}

SimplexRawPtr Spacetime::createSimplex(const std::tuple<uint8_t, uint8_t> &numericOrientation) {
  CLOG(INFO_LEVEL, "Creating Orientation...");
  double squaredLength = alpha;
  double timelikeSquaredLength = alpha;
  SimplexOrientationPtr orientation = std::make_shared<SimplexOrientation>(
    std::get<0>(numericOrientation),
    std::get<1>(numericOrientation));
  CLOG(INFO_LEVEL, "Created.");
  std::uint8_t k = orientation->getK();
  auto [ti, tf] = orientation->numeric();
  VertexPtrs vertices = {};
  vertices.reserve(k);
  Edges edges = {};
  edges.reserve(Simplex::computeNumberOfEdges(k));
  CLOG(INFO_LEVEL, "vertexIdCounter: ", std::to_string(vertexIdCounter));
  for (int i = 0; i < ti; i++) {
    // Create ti Timelike vertices
    // Use coning to construct the vertex edges. For each new vertex; draw an edge to each existing vertex.
    VertexPtr newVertex = vertexList->add(vertexIdCounter++, {static_cast<double>(currentTime)});
    CLOG(INFO_LEVEL, "Made vertex id=", newVertex->getId(), " (counter now ", vertexIdCounter, ")");
    if (getMetric()->getSignature()->getSignatureType() == SignatureType::Lorentzian) {
      timelikeSquaredLength = -alpha;
    }
    for (const auto &existingVertex : vertices) {
      CLOG(INFO_LEVEL, "Creating edge...");
      EdgeRawPtr edge = edgeList->add(
        existingVertex,
        newVertex,
        timelikeSquaredLength
        );
      CLOG(INFO_LEVEL, "Registering edges...");
      existingVertex->addOutEdge(edge);
      newVertex->addInEdge(edge);
      edges.push_back(edge);
    }
    CLOG(INFO_LEVEL, "Storing vertex...");
    vertices.push_back(newVertex);
  }
  for (int i = 0; i < tf; i++) {
    // Create ti Spacelike vertices
    // Use coning to construct the vertex edges. For each new vertex; draw an edge to each existing vertex.
    /// We can't just use the vertexList .size() here, because some vertices can be removed. We need to keep a
    /// counter:
    VertexPtr newVertex = vertexList->add(vertexIdCounter++, {static_cast<double>(currentTime + 1)});
    CLOG(INFO_LEVEL, "Made vertex id=", newVertex->getId(), " (counter now ", vertexIdCounter, ")");
    for (const auto &existingVertex : vertices) {
      EdgeRawPtr edge;
      if (existingVertex->getTime() < newVertex->getTime()) {
        CLOG(INFO_LEVEL, "Creating tf edge with ", existingVertex->getId(), " and ", newVertex->getId());
        edge = edgeList->add(existingVertex, newVertex, squaredLength);
        CLOG(INFO_LEVEL, "Created...");
      } else {
        CLOG(INFO_LEVEL, "Creating tf edge...");
        edge = edgeList->add(existingVertex, newVertex, timelikeSquaredLength);
        CLOG(INFO_LEVEL, "Created...");
      }
      existingVertex->addOutEdge(edge);
      newVertex->addInEdge(edge);
      CLOG(INFO_LEVEL, "Registering edge...");
      edges.push_back(edge);
    }
    vertices.push_back(newVertex);
  }
  CLOG(INFO_LEVEL, "Handing off vertices and edges...");
  return createSimplex(vertices, edges);
}

[[nodiscard]] py::tuple
Spacetime::getGluableFacesForPython(
  SimplexRawPtr unattachedSimplex,
  SimplexRawPtr attachedSimplex
  ) {
  auto result = getGluableFaces(unattachedSimplex, attachedSimplex);
  if (!result.has_value()) return py::make_tuple();
  auto [first, second] = result.value();
  return py::make_tuple(wrap_non_owning(first), wrap_non_owning(second));
}

[[nodiscard]] OptionalSimplexPtrPair
Spacetime::getGluableFaces(
  SimplexRawPtr unattachedSimplex,
  SimplexRawPtr attachedSimplex
  )
{
  auto unattachedFacets = unattachedSimplex->getFacets(); // vector<shared_ptr<Simplex>>
  auto attachedFacets = attachedSimplex->getFacets();
#if CASET_DEBUG
  for (const auto &f : unattachedFacets) {
    f->validate();
  }
  for (const auto &f : attachedFacets) {
    f->validate();
  }
#endif

  for (auto &unattachedFace : unattachedFacets) {
    const auto [tia, tfa] = unattachedFace->getOrientation()->numeric();
    if (tia == 0 || tfa == 0) continue; // Skip degenerate faces
    if (!unattachedFace->isCausallyAvailable()) continue;
    for (auto &attachedFace : attachedFacets) {
      if (unattachedFace->isTimelike() != attachedFace->isTimelike()) continue;
      // Skip faces that don't match in timelikeness
      const auto [tib, tfb] = attachedFace->getOrientation()->numeric();
      if (tib == 0 || tfb == 0) continue; // Skip degenerate faces
      if (attachedFace->isInternal()) continue;
#if CASET_DEBUG
      attachedFace->validate();
      unattachedFace->validate();
#endif
      if (tia == tfb && tfa == tib) return std::make_optional(std::make_pair(unattachedFace, attachedFace));
      if (tia == tib && tfa == tfb) return std::make_optional(std::make_pair(unattachedFace, attachedFace));
    }
  }
  return std::nullopt;
}

void Spacetime::moveInEdgesFromVertex(const VertexPtr &from, const VertexPtr &to) {
  for (const auto &edge : from->getInEdges()) {
    // The source is external to the face/simplex, the `from` node is going to be going away.
    const VertexPtr originalSource = edge->getSource();
    originalSource->removeOutEdge(edge);
    from->removeInEdge(edge);
    auto canonicalEdge = edgeList->remove(edge);
    canonicalEdge->replaceTargetVertex(to);
    to->addInEdge(canonicalEdge.get());
    originalSource->addOutEdge(canonicalEdge.get());
    edgeList->add(std::move(canonicalEdge));
  }
}

void Spacetime::moveOutEdgesFromVertex(const VertexPtr &from, const VertexPtr &to) {
  for (const auto &edge : from->getOutEdges()) {
    const VertexPtr originalTarget = edge->getTarget();
    originalTarget->removeInEdge(edge);
    from->removeOutEdge(edge);
    auto canonicalEdge = edgeList->remove(edge);
    canonicalEdge->replaceSourceVertex(to);
    to->addOutEdge(canonicalEdge.get());
    originalTarget->addInEdge(canonicalEdge.get());
    edgeList->add(std::move(canonicalEdge));
  }
}


SimplexPtrSet Spacetime::getSimplicesWithOrientation(std::tuple<uint8_t, uint8_t> orientation) {
  SimplexOrientationPtr o = std::make_shared<
    SimplexOrientation>(std::get<0>(orientation), std::get<1>(orientation));
  SimplexPtrSet result{};
  for (const auto &bucket : externalSimplices | std::views::values) {
    for (const auto &simplex : bucket) {
      for (const auto &simplexFacialOrientation : simplex->getOrientation()->getFacialOrientations()) {
        if (simplex->getOrientation() == o) result.insert(simplex);
      }
    }
  }
  return result;
}

py::list Spacetime::getSimplicesWithOrientationForPython(std::tuple<uint8_t, uint8_t> orientation) {
  SimplexPtrSet simplices_ = getSimplicesWithOrientation(orientation);
  py::list result{};
  for (auto simplex : simplices_) {
    result.append(wrap_non_owning(simplex));
  }
  return result;
}


/// When we attach two simplices; the "attached" one is assumed to be part of a simplicial complex. The "unattached" one
/// is assumed to be part of another simplicial complex, but usually by itself. The "attached" simplex replaces
/// corresponding vertices in the "unattached" simplex with it's own vertices. Same goes for the _internal_ edges. Any
/// external edges in "unattached" are redirected from those vertices on "unattached" to the corresponding vertex in
/// "attached".
void Spacetime::attachAtVertices(
  const SimplexRawPtr &unattached,
  const SimplexRawPtr &attached,
  const std::vector<std::pair<VertexPtr, VertexPtr> > &vertexPairs // {unattached, attached}
) {
  CLOG(INFO_LEVEL, "attachAtVertices called. Pre-validating.");
  // Bone density in Regge calculus can be calculated as the size of the Simplex list on the Edge.
#if CASET_DEBUG
  CLOG(INFO_LEVEL, "Validating unattached simplex...");
  unattached->validate();
  CLOG(INFO_LEVEL, "Validated unattached simplex.");
  CLOG(INFO_LEVEL, "Validating attached simplex...");
  attached->validate();
  CLOG(INFO_LEVEL, "Validated attached simplex.");
#endif
  // Move external edges from unattached vertices to attached vertices.
  for (const auto &[unattachedVertex, attachedVertex] : vertexPairs) {
    CLOG(INFO_LEVEL, "Attaching at vertices ", unattachedVertex->toString() , "<>", attachedVertex->toString());
    attachAtVertex(unattached, attached, unattachedVertex, attachedVertex);
  }
#if CASET_DEBUG
  CLOG(INFO_LEVEL, "Validating unattached simplex...");
  unattached->validate();
  CLOG(INFO_LEVEL, "Validated unattached simplex.");
  CLOG(INFO_LEVEL, "Validating attached simplex...");
  attached->validate();
  CLOG(INFO_LEVEL, "Validated attached simplex.");
#endif
  CLOG(INFO_LEVEL, "Validated.");
}

void Spacetime::attachAtVertex(SimplexRawPtr unattachedSimplex, SimplexRawPtr attachedSimplex, const VertexPtr &unattached, const VertexPtr &attached) {
  // After this; attached will have some new edges and unattached will have zero edges. That means any simplex
  // containing one of unattached's old edges will need that edge removed.
  const auto [oldEdges, newEdges] = unattached->moveEdgesTo(attached);

  for (const auto &simplex : unattached->getSimplices()) {
    // Replacing the vertex handles updating the state associated with the Simplex on the Vertex, but not the state
    // associated with the Vertex on the Simplex.
    simplex->replaceVertex(unattached, attached);
    // To finish updating state; we need to remove defunct edges from the simplex?
  }

  // Now we need to re-key the oldEdges to their keys match newEdges.
  for (const auto &oldKey : *oldEdges) {
    const auto &canonicalEdge = edgeList->updateKey(oldKey);
    canonicalEdge->removeSimplex(unattachedSimplex);
  }
  // for (const auto &edgeKey : oldEdges) {
    // throw std::runtime_error("We need to ensure we update the edges in each simplex. Python can keep them around longer than it should. I think getEdges() is still returning the same pointers. Who is responsible for updating the edges on the simplex itself? I think it should be the Simplex. In the tests; we get edges from the simplex, and i think the edges that are invalidated are being returned there.");
  // }

  if (unattached->degree() == 0) vertexList->remove(unattached);
#if CASET_DEBUG
  unattachedSimplex->validate();
  attachedSimplex->validate();
#endif
}

std::tuple<SimplexRawPtr, bool> Spacetime::causallyAttachFaces(
  SimplexRawPtr attachedFace,
  SimplexRawPtr unattachedFace
) {
  if (!attachedFace->isCausallyAvailable() || !unattachedFace->isCausallyAvailable()) {
    CLOG(ERROR_LEVEL, "One or more of attachedFace and unattachedFace was not causally available!\n", attachedFace->toString(), "\n", unattachedFace->toString());
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

  VertexPtrs vertices{};
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
  const std::optional<VertexPtrs> attachedOrderedVerticesOptional = attachedFace->getVerticesWithParityTo(unattachedFace);

  if (!attachedOrderedVerticesOptional.has_value()) {
    CLOG(WARN_LEVEL,
         "No compatible vertex order found for myFace and yourFace.\n",
         attachedFace->toString(),
         "\n",
         unattachedFace->toString());
    return {nullptr, false};
  }

  const VertexPtrs &attachedOrderedVertices = attachedOrderedVerticesOptional.value();
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
  }

  return {attachedFace, true};
}

std::tuple<py::object, bool> Spacetime::causallyAttachFacesForPython(SimplexRawPtr attachedFace, SimplexRawPtr unattachedFace) {
  auto [simplex, success] = causallyAttachFaces(attachedFace, unattachedFace);
  return {wrap_non_owning(simplex), success};
}

OptionalSimplexPtrPair Spacetime::chooseSimplexFacesToGlue(SimplexRawPtr unattachedSimplex) {
  for (const auto &facialOrientation : unattachedSimplex->getGluableFaceOrientations()) {
    const auto &prospectiveCofaces = externalSimplices[facialOrientation];
    if (prospectiveCofaces.empty()) continue;
    for (auto attachedCofaceId = prospectiveCofaces.begin(); attachedCofaceId != prospectiveCofaces.end(); ++
         attachedCofaceId) {
      if ((*attachedCofaceId)->fingerprint.fingerprint() == unattachedSimplex->fingerprint.fingerprint()) continue;
      if (!unattachedSimplex->hasCausallyAvailableFacet() || !(*attachedCofaceId)->hasCausallyAvailableFacet()) continue;
#if CASET_DEBUG
      (*attachedCofaceId)->validate();
#endif
      OptionalSimplexPtrPair gluablePair = getGluableFaces(unattachedSimplex, *attachedCofaceId);
      if (gluablePair.has_value()) {
        const auto &[unattachedFace, attachedFace] = gluablePair.value();
        return gluablePair;
      }
    }
  }
  return std::nullopt;
}

py::tuple Spacetime::chooseSimplexFacesToGlueForPython(
  SimplexRawPtr unattachedSimplex
  ) {
  auto chosen = chooseSimplexFacesToGlue(unattachedSimplex);
  if (!chosen.has_value()) return py::make_tuple();
  auto [first, second] = chosen.value();
  return py::make_tuple(wrap_non_owning(first), wrap_non_owning(second));
}

SimplexPtrSet Spacetime::getExternalSimplices() noexcept {
  SimplexPtrSet simplices{};
  for (const auto &[facialOrientation, bucket] : externalSimplices) {
    for (const auto &simplex : bucket) {
      simplices.insert(simplex);
    }
  }
  return simplices;
}

py::list Spacetime::getExternalSimplicesForPython() noexcept {
  py::list result{};
  for (const auto &[facialOrientation, bucket] : externalSimplices) {
    for (const auto &simplex : bucket) {
      result.append(wrap_non_owning(simplex));
    }
  }
  return result;
}

std::vector<VertexPtrs> Spacetime::getConnectedComponents() const {
  VertexPtrSet seen{};
  std::vector<VertexPtrs> components{};
  for (const auto &vertex : vertexList->toVector()) {
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
        VertexPtr neighbor = edge->getTarget();
        if (neighbor != nullptr && !seen.contains(neighbor)) {
          stack.push_back(neighbor);
        }
      }
      for (const auto &edge : current->getInEdges()) {
        VertexPtr neighbor = edge->getSource();
        if (neighbor != nullptr && !seen.contains(neighbor)) {
          stack.push_back(neighbor);
        }
      }
    }
    components.push_back(component);
  }
  return components;
}

VertexPtr Spacetime::createVertex(const IdType id) {
#ifdef CASET_DEBUG
  if (id == 0) throw std::runtime_error("Invalid vertex ID: 0");
#endif
  return vertexList->add(id);
}

VertexPtr Spacetime::createVertex(const IdType id, const std::vector<double> &coords) {
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
