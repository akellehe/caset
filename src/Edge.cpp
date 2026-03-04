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
#include "utils.h"

#include <vector>
#include <memory>

namespace caset {
template<int D>
class Simplex;

template<int D>
Edge<D>::Edge(
  const VertexPtr<D> &source_,
  const VertexPtr<D> &target_,
  double squaredLength_
) : source(source_), target(target_), squaredLength(squaredLength_), fingerprint({source_->getId(), target_->getId()}) {
}

template<int D>
Edge<D>::Edge(
  const VertexPtr<D> &source_,
  const VertexPtr<D> &target_
) : source(source_), target(target_), fingerprint({source_->getId(), target_->getId()}) {
  // Set squaredLength to a random value between -1 and 1
  squaredLength = random_uniform(); // TODO: Should we use a poisson dist here for coset theory?
}

template<int D>
[[nodiscard]] VertexPtr<D> Edge<D>::getSource() const noexcept {
  return source;
}

template<int D>
[[nodiscard]] VertexPtr<D> Edge<D>::getTarget() const noexcept {
  return target;
}

template<int D>
[[nodiscard]] double Edge<D>::getSquaredLength() const noexcept {
  return squaredLength;
}

#ifdef CASET_VERBOSE
template<int D>
[[nodiscard]] std::string Edge<D>::toString() const noexcept {
  return source->toString() + "->" + target->toString();
}
#endif

template<int D>
void Edge<D>::replaceSourceVertex(const VertexPtr<D> &newSource) {
  fingerprint.removeId(source->getId());
  source = newSource;
  fingerprint.addId(newSource->getId());
  fingerprint.refresh();
}

template<int D>
void Edge<D>::replaceTargetVertex(const VertexPtr<D> &newTarget) {
  fingerprint.removeId(target->getId());
  target = newTarget;
  fingerprint.addId(newTarget->getId());
  fingerprint.refresh();
}

template<int D>
bool Edge<D>::hasVertex(std::uint64_t vertexId) {
  if (getSource()->getId() == vertexId || getTarget()->getId() == vertexId) return true;
  return false;
}

template<int D>
bool Edge<D>::operator==(const Edge &other) const {
  return fingerprint.fingerprint() == other.fingerprint.fingerprint();
}

template<int D>
[[nodiscard]] std::uint64_t Edge<D>::toHash() const {
  return fingerprint.fingerprint();
}

template<int D>
EdgeKey<D> Edge<D>::getKey() const noexcept {
  return {source->getId(), target->getId()};
}
}

