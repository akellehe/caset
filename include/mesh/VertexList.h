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

#ifndef TESSERA_VERTEXLIST_H
#define TESSERA_VERTEXLIST_H

#include <cstdint>
#include <deque>
#include <vector>
#include <unordered_map>

#include "mesh/Vertex.h"

namespace tessera {

/// Flat-pool vertex container.
///
/// Objects live in a `std::deque` (stable element addresses) with a free-list
/// for slot reuse.  A parallel `liveVec_` provides O(1) random access without
/// copying, and `idToIndex_` maps vertex ID → pool slot.
class VertexList {
  public:
    Vertex* operator[](const std::uint64_t vertexId) {
      auto it = idToIndex_.find(vertexId);
      if (it == idToIndex_.end()) return nullptr;
      return &pool_[it->second];
    }

    Vertex* get(std::uint64_t id) {
      auto it = idToIndex_.find(id);
      if (it == idToIndex_.end()) return nullptr;
      return &pool_[it->second];
    }

    bool contains(const std::uint64_t id) const noexcept {
      return idToIndex_.contains(id);
    }

    Vertex* add(const std::uint64_t id, const std::vector<double> &coords) noexcept {
      auto found = idToIndex_.find(id);
      if (found != idToIndex_.end()) {
        CLOG(INFO_LEVEL, "Vertex ", std::to_string(id), " already exists!");
        return &pool_[found->second];
      }
      std::uint32_t slot;
      if (!freeSlots_.empty()) {
        slot = freeSlots_.back();
        freeSlots_.pop_back();
        pool_[slot] = Vertex(id, coords);
      } else {
        slot = static_cast<std::uint32_t>(pool_.size());
        pool_.emplace_back(id, coords);
      }
      idToIndex_.emplace(id, slot);
      Vertex* raw = &pool_[slot];
      raw->liveIdx_ = static_cast<std::uint32_t>(liveVec_.size());
      liveVec_.push_back(raw);
      return raw;
    }

    Vertex* add(const std::uint64_t id) noexcept {
      return add(id, std::vector<double>{});
    }

    void replace(Vertex* toRemove, Vertex* toAdd) {
#if TESSERA_ASSERTIONS
      if (toAdd == nullptr) throw std::invalid_argument("Cannot remove a nullptr vertex");
      if (toRemove == nullptr) throw std::invalid_argument("Cannot remove a nullptr vertex");
#endif
      remove(toRemove);
      add(toAdd->getId(), toAdd->getCoordinates());
    }

    void remove(Vertex* vertex) noexcept {
      auto id = vertex->getId();
      auto poolIt = idToIndex_.find(id);
      if (poolIt == idToIndex_.end()) return;
      freeSlots_.push_back(poolIt->second);
      idToIndex_.erase(poolIt);

      // Swap-and-pop from liveVec_ using the index stored on the Vertex
      auto idx = vertex->liveIdx_;
      if (idx < liveVec_.size()) {
        auto lastIdx = static_cast<std::uint32_t>(liveVec_.size() - 1);
        if (idx != lastIdx) {
          liveVec_[idx] = liveVec_[lastIdx];
          liveVec_[idx]->liveIdx_ = idx;
        }
        liveVec_.pop_back();
      }
      vertex->liveIdx_ = UINT32_MAX;
    }

    /// Swap the map keys of two vertices without destroying either object.
    /// Used by Spacetime::swapVertexLabels for Brunekreef relabeling.
    /// The vertices' internal IDs must already have been swapped before calling.
    void swapKeys(std::uint64_t oldId1, std::uint64_t oldId2) {
      auto it1 = idToIndex_.find(oldId1);
      auto it2 = idToIndex_.find(oldId2);
      if (it1 == idToIndex_.end() || it2 == idToIndex_.end()) return;
      auto slot1 = it1->second;
      auto slot2 = it2->second;
      idToIndex_.erase(it1);
      idToIndex_.erase(it2);
      idToIndex_.emplace(oldId2, slot1);
      idToIndex_.emplace(oldId1, slot2);
      // liveIdx_ on the Vertex objects doesn't change — only the map keys do
    }

    std::size_t size() const noexcept {
      return idToIndex_.size();
    }

    /// O(1) — returns the live-vertex vector directly (no copy).
    const std::vector<Vertex*>& liveVector() const noexcept {
      return liveVec_;
    }

    /// Compatibility: returns a copy of the live-vertex vector.
    std::vector<Vertex*> toVector() const noexcept {
      return liveVec_;
    }

    void reserve(std::size_t nSimplices) noexcept {
      idToIndex_.reserve(nSimplices);
    }

  private:
    std::deque<Vertex> pool_;                                  ///< Stable-address storage
    std::vector<std::uint32_t> freeSlots_;                     ///< Recycled pool indices
    std::unordered_map<std::uint64_t, std::uint32_t> idToIndex_; ///< vertex ID → pool slot
    std::vector<Vertex*> liveVec_;                             ///< Flat array of live vertices
};
} // tessera

#endif //TESSERA_VERTEXLIST_H
