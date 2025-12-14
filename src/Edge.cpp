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

#include "Fingerprint.h"
#include "Edge.h"
#include "EdgeKey.h"
#include "Vertex.h"

#include <vector>
#include <memory>


namespace caset {

class Simplex;

    Edge::Edge(
      const VertexPtr &source_,
      const VertexPtr &target_,
      double squaredLength_
    ) : source(source_), target(target_), squaredLength(squaredLength_), fingerprint({source_->getId(), target_->getId()}) {
    }

    Edge::Edge(
      const VertexPtr &source_,
      const VertexPtr &target_
    ) : source(source_), target(target_), fingerprint({source_->getId(), target_->getId()}) {
      // Set squaredLength to a random value between -1 and 1
      squaredLength = random_uniform(); // TODO: Should we use a poisson dist here for coset theory?
    }

    [[nodiscard]] VertexPtr Edge::getSource() const noexcept {
      return source;
    }

    [[nodiscard]] VertexPtr Edge::getTarget() const noexcept {
      return target;
    }

    [[nodiscard]] double Edge::getSquaredLength() const noexcept {
      return squaredLength;
    }

    [[nodiscard]] std::string Edge::toString() const noexcept {
      return source->toString() + "->" + target->toString();
    }

    /// This method changes the target source in-place. Note that if this edge is registered elsewhere (e.g. in a
    /// std::unordered_map in the Spacetime) then it needs to be unregistered first, modified, then re-registered to
    /// ensure consistent hashing/lookup. This method also updates the fingerprint hastily. If you want to update in
    /// batches remove the fingerprint.refresh() call.
    void Edge::replaceSourceVertex(const VertexPtr &newSource) {
      fingerprint.removeId(source->getId());
      source = newSource;
      fingerprint.addId(newSource->getId());
      fingerprint.refresh();
    }

    /// Same as replaceSourceVertex above, but for targets.
    void Edge::replaceTargetVertex(const VertexPtr &newTarget) {
      fingerprint.removeId(target->getId());
      target = newTarget;
      fingerprint.addId(newTarget->getId());
      fingerprint.refresh();
    }

    ///
    /// @param vertexId The ID of a Vertex for which ownership should be checked.
    /// @return true if the Vertex exists as an endpoint of this edge
    bool Edge::hasVertex(std::uint64_t vertexId) {
      if (getSource()->getId() == vertexId || getTarget()->getId() == vertexId) return true;
      return false;
    }

    ///
    /// @param from the ID of a vertex to or from which this Edge should no longer point.
    /// @param to the ID of a source or target vertex to which this Edge should now point.
    void Edge::redirect(const VertexPtr &from, const VertexPtr &to) noexcept {
      if (getSource()->getId() == from->getId()) {
        replaceSourceVertex(to);
      }
      if (getTarget()->getId() == from->getId()) {
        replaceTargetVertex(to);
      }
    }

    bool Edge::operator==(const Edge &other) const {
      return fingerprint.fingerprint() == other.fingerprint.fingerprint();
    }

    [[nodiscard]] std::uint64_t Edge::toHash() const {
      return fingerprint.fingerprint();
    }

    EdgeKey Edge::getKey() const noexcept {
      return {source->getId(), target->getId()};
    }

    void Edge::refreshFingerprint() noexcept {
      // May want to update in place with fingerprint.setIds()
      fingerprint = Fingerprint({source->getId(), target->getId()});
    }


}

