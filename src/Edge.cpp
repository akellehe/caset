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

#include "Edge.h"
#include "Vertex.h"
#include "Fingerprint.h"
#include "EdgeKey.h"
#include "Simplex.h"

#include <vector>
#include <random>
#include <memory>

namespace caset {

class Simplex;

    Edge::Edge(
      VertexPtr source_,
      VertexPtr target_,
      double squaredLength_
    ) : source(source_), target(target_), squaredLength(squaredLength_), fingerprint({source_->getId(), target_->getId()}) {
#if CASET_DEBUG
      if (source_ == target_) throw std::runtime_error("You can't create self-referential edges.");
#endif
    }

    Edge::Edge(
      VertexPtr source_,
      VertexPtr target_
    ) : source(source_), target(target_), fingerprint({source_->getId(), target_->getId()}) {
      // Set squaredLength to a random value between -1 and 1
#if CASET_DEBUG
      if (source_ == target_) throw std::runtime_error("You can't create self-referential edges.");
#endif
      squaredLength = random_uniform(); // TODO: Should we use a poisson dist here for coset theory?
    }

    [[nodiscard]] VertexPtr Edge::getSource() const {
#if CASET_DEBUG
      if (source == 0) throw std::runtime_error("You attempted to get an edge with a 0 source vertex.");
#endif
      return source;
    }

    [[nodiscard]] VertexPtr Edge::getTarget() const {
#if CASET_DEBUG
      if (target == 0) throw std::runtime_error("You attempted to get an edge with a 0 target vertex.");
#endif
      return target;
    }

    [[nodiscard]] double Edge::getSquaredLength() const noexcept {
      return squaredLength;
    }

    [[nodiscard]] std::string Edge::toString() const noexcept {
      return std::to_string(source->getId()) + "->" + std::to_string(target->getId());
    }

    void Edge::copyInPlaceTo(Edge *other) {
      other->source = source;
      other->target = target;
      other->simplices = getSimplices();
      other->refreshFingerprint();
    }

    /// This method changes the target source in-place. Note that if this edge is registered elsewhere (e.g. in a
    /// std::unordered_map in the Spacetime) then it needs to be unregistered first, modified, then re-registered to
    /// ensure consistent hashing/lookup.
    void Edge::replaceSourceVertex(VertexPtr source_) {
#if CASET_DEBUG
      if (source_ == target) throw std::runtime_error(
        "You can't replace a source vertex with the same as the target since that would create a self-reference.");
#endif
      source = source_;
      refreshFingerprint();
    }

    /// This method changes the target Vertex in-place. Note that if this edge is registered elsewhere (e.g. in a
    /// std::unordered_map in the Spacetime) then it needs to be unregistered first, modified, then re-registered to
    /// ensure consistent hashing/lookup.
    void Edge::replaceTargetVertex(VertexPtr target_) {
#if CASET_DEBUG
      if (target_ == source) throw std::runtime_error(
        "You can't replace a target vertex with the same as the source since that would create a self-reference.");
#endif
      target = target_;
      refreshFingerprint();
    }

    ///
    /// @param vertexId The ID of a Vertex for which ownership should be checked.
    /// @return true if the Vertex exists as an endpoint of this edge
    bool Edge::hasVertex(const Vertex *vertex) const {
      auto vid = vertex->getId();
      if (getSource()->getId() == vid || getTarget()->getId() == vid) return true;
      return false;
    }

    void Edge::assertUnused() const {
      for (const auto &simplex : getSimplices()) {
        if (simplex->hasEdge(this)) {
          throw std::runtime_error("Edge is currently in use by a simplex.");
        }
      }
      if (source->hasEdge(this)) {
        throw std::runtime_error("Edge is in use by it's source vertex");
      }
      if (target->hasEdge(this)) {
        throw std::runtime_error("Edge is in use by it's target vertex");
      }
    }

    ///
    /// @param from the ID of a vertex to or from which this Edge should no longer point.
    /// @param to the ID of a source or target vertex to which this Edge should now point.
    void Edge::redirect(VertexPtr from, VertexPtr to) {
#if CASET_DEBUG
      if (from == to) throw std::runtime_error("You attempted to redirect an edge from a vertex to the same vertex.");
#endif
      if (getSource() == from) {
        replaceSourceVertex(to);
      }
      if (getTarget() == from) {
        replaceTargetVertex(to);
      }
    }

    void Edge::replaceOnReferents(EdgeRawPtr replacement) {
      for (const auto &simplex : getSimplices()) {
        simplex->removeEdge(this);
        simplex->addEdge(replacement);
        simplex->replaceVertex(source, replacement->getSource());
        simplex->replaceVertex(target, replacement->getTarget());
      }

      source->removeOutEdge(this);
      source->addOutEdge(replacement);

      target->removeInEdge(this);
      target->addInEdge(replacement);
    }

    bool Edge::operator==(const Edge &other) const {
      return fingerprint.fingerprint() == other.fingerprint.fingerprint();
    }

    [[nodiscard]] std::uint64_t Edge::toHash() const {
      return fingerprint.fingerprint();
    }


    [[nodiscard]] EdgeKey Edge::getKey() const noexcept {
      return {source->getId(), target->getId()};
    }

    [[nodiscard]] SimplexSet Edge::getSimplices() const noexcept { return simplices; }

    void Edge::addSimplex(Simplex *simplex) noexcept { simplices.insert(simplex); }

    void Edge::removeSimplex(Simplex *simplex) noexcept {
      simplices.erase(simplex);
    }

    void Edge::refreshFingerprint() noexcept {
      fingerprint = Fingerprint({source->getId(), target->getId()});
    }

};


