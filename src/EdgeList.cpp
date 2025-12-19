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

#include "EdgeList.h"
#include <memory>
#include <vector>

#include "Edge.h"
#include "EdgeKey.h"
#include "Logger.h"

namespace caset {
EdgePtr EdgeList::add(const EdgePtr &edge) {
  return getOrInsert(edge);
}

EdgePtr EdgeList::add(const VertexPtr &source, const VertexPtr &target) {
  auto edge = std::make_shared<Edge>(source, target);
  return getOrInsert(edge);
}

EdgePtr EdgeList::add(const VertexPtr &source, const VertexPtr &target, double squaredLength) noexcept {
  auto edge = std::make_shared<Edge>(source, target, squaredLength);
  return getOrInsert(edge);
}

void EdgeList::remove(const EdgePtr &edge) noexcept {
#ifdef CASET_ASSERTIONS
  if (!edgeList.contains(edge->fingerprint.fingerprint())) {
    CLOG(WARN_LEVEL, "You attempted to remove an edge that does not exist: ", edge->toString());
    for (const auto &[fp, e] : edgeList) {
      CLOG(WARN_LEVEL, "    - ", e->toString());
    }
    return;
  }
#endif
  edgeList.erase(edge->fingerprint.fingerprint());
#ifdef CASET_ASSERTIONS
  if (edgeList.contains(edge->fingerprint.fingerprint())) {
    CLOG(CRITICAL_LEVEL, "Failed to completely remove an edge from the spacetime: " + edge->toString());
    for (const auto &[fp, e] : edgeList) {
      CLOG(CRITICAL_LEVEL, "    - ", e->toString());
    }
    return;
  }
#endif
}

void EdgeList::replace(const EdgePtr &toRemove, const EdgePtr &toAdd) noexcept {
  edgeList.erase(toRemove->fingerprint.fingerprint());
  edgeList.emplace(toAdd->fingerprint.fingerprint(), toAdd);
}

[[nodiscard]] Edges EdgeList::toVector() const noexcept {
  Edges result{};
  result.reserve(edgeList.size());
  for (auto &[fp, edge] : edgeList) {
    result.push_back(edge);
  }
  return result;
}

[[nodiscard]] std::size_t EdgeList::size() const {
  return edgeList.size();
}

EdgePtr EdgeList::get(const std::uint64_t &fingerprint) {
  return edgeList.at(fingerprint);
}

EdgePtr EdgeList::getOrInsert(const EdgePtr &edge) {
  if (edge->getSource()->getId() == edge->getTarget()->getId()) {
    throw std::runtime_error("You cannot create an edge from a vertex to itself: " + edge->toString());
  }
  if (edgeList.contains(edge->fingerprint.fingerprint())) {
    const auto &[fingerprint, found] = *edgeList.find(edge->fingerprint.fingerprint());
#ifdef CASET_ASSERTIONS
    if (found->getSource()->getId() != edge->getSource()->getId() || found->getTarget()->getId() != edge->getTarget()->getId()) {
      throw std::runtime_error(
        "Fingerprint collision between edges: " + edge->toString() + " and " + found->toString());
    }
#endif
    return found;
  }
  // CLOG(DEBUG_LEVEL, "Adding edge: ", edge->toString());
  edgeList.emplace(edge->fingerprint.fingerprint(), edge);
  return edge;
}

void EdgeList::reserve(std::size_t size) {
  edgeList.reserve(size);
}
} // caset
