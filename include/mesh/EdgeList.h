// MIT License -- Copyright (c) 2025 Andrew Kelleher
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

#ifndef TESSERA_EDGELIST_H
#define TESSERA_EDGELIST_H

#include <cstdint>
#include <deque>
#include <vector>
#include <unordered_map>

#include "mesh/Edge.h"
#include "Logger.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::mesh {
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;

/// Flat-pool edge container.
///
/// Edges live in a `std::deque` (stable element addresses) with a free-list
/// for slot reuse.  `fpToSlot_` maps edge fingerprint → pool slot for O(1)
/// deduplication.
class EdgeList {
  public:
    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] const Edges &toVector() const noexcept;

    EdgePtr add(const VertexPtr &source, const VertexPtr &target);
    EdgePtr add(const VertexPtr &source, const VertexPtr &target, double squaredLength) noexcept;
    /// Insert if absent, otherwise return the existing edge.
    /// Returns {ptr, true} on fresh insert, {ptr, false} on dedupe-hit.
    /// Used by transactional Pachner moves to record which edges they
    /// freshly created (so rollback knows which to remove).
    std::pair<EdgePtr, bool> tryAdd(const VertexPtr &source, const VertexPtr &target,
                                    double squaredLength);
    EdgePtr get(const std::uint64_t &fingerprint);
    void remove(const EdgePtr &edge) noexcept;
    void replace(const EdgePtr &toRemove, const EdgePtr &toAdd) noexcept;

    /// Re-key an edge's fingerprint in the lookup map without moving the object.
    void rekeyEdge(std::uint64_t oldFp, std::uint64_t newFp);

    /// Detach an edge from the fingerprint lookup (but keep it in the pool).
    /// Returns the pool slot so the caller can update the fingerprint and call
    /// reattachEdge().  Returns UINT32_MAX if not found.
    std::uint32_t detachEdge(std::uint64_t fp) {
      auto it = fpToSlot_.find(fp);
      if (it == fpToSlot_.end()) return UINT32_MAX;
      auto slot = it->second;
      fpToSlot_.erase(it);
      return slot;
    }

    /// Re-attach a previously detached edge under its (possibly new) fingerprint.
    void reattachEdge(std::uint32_t slot) {
      auto fp = pool_[slot].fingerprint.fingerprint();
      fpToSlot_.emplace(fp, slot);
    }

    void reserve(std::size_t nSimplices);

  private:
    std::deque<Edge> pool_;
    std::vector<std::uint32_t> freeSlots_;
    std::unordered_map<std::uint64_t, std::uint32_t> fpToSlot_;
    std::vector<EdgePtr> liveVec_;

    EdgePtr getOrInsert(const VertexPtr &source, const VertexPtr &target, double squaredLength);
    std::uint32_t allocSlot(const VertexPtr &source, const VertexPtr &target, double squaredLength);
};
} // namespace tessera::mesh

#endif //TESSERA_EDGELIST_H
