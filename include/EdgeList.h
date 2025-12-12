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

#ifndef CASET_EDGELIST_H
#define CASET_EDGELIST_H

#include <memory>
#include <ranges>
#include <vector>
#include <unordered_set>

#include "Edge.h"
#include "Logger.h"

namespace caset {
class EdgeList {
  public:
    Edge *add(std::unique_ptr<Edge> edge) {
      const auto key = edge->getKey(); // compute key *before* move
      auto [it, inserted] = edgeList.insert_or_assign(key, std::move(edge));
      return it->second.get();
    }

    Edge *add(std::uint64_t src, std::uint64_t tgt) {
      auto edge = std::make_unique<Edge>(src, tgt);
      return getOrInsert(std::move(edge));
    }

    Edge *add(std::uint64_t src, std::uint64_t tgt, double squaredLength) {
#ifdef CASET_DEBUG
      if (src == 0 || tgt == 0) {
        CLOG(WARN_LEVEL, "Source = ", std::to_string(src), " Target = ", std::to_string(tgt));
        throw std::runtime_error("Invalid src and target ids!");
      }
      if (src == tgt) {
        CLOG(WARN_LEVEL, "Source and target were equal (self-referential)");
      throw std::runtime_error("Invalid src and target ids!");
      }
#endif
      auto edge = std::make_unique<Edge>(src, tgt, squaredLength);
      return getOrInsert(std::move(edge));
    }

    std::unique_ptr<Edge> remove(const EdgeKey &edgeKey) noexcept {
      auto it = edgeList.find(edgeKey);
      if (it == edgeList.end()) {
        return nullptr;
      }
      std::unique_ptr<Edge> removed = std::move(it->second);
      edgeList.erase(it);
      return removed;
    }

    std::unique_ptr<Edge> remove(Edge *edge) noexcept {
#ifdef CASET_DEBUG
      if (!edgeList.contains(edge->getKey())) {
        CLOG(WARN_LEVEL, "You attempted to remove an edge that does not exist: ", edge->toString());
        for (const auto &[k, e] : edgeList) {
          CLOG(WARN_LEVEL, "    - ", e->toString());
        }
        return nullptr;
      }
#endif
      auto e = std::move(edgeList[edge->getKey()]);
      edgeList.erase(edge->getKey());
      return e;
    }

    [[nodiscard]] std::vector<Edge *> toVector() const noexcept {
      std::vector<Edge *> result;
      result.reserve(edgeList.size());
      for (const auto &edge : edgeList | std::views::values) {
        result.push_back(edge.get());
      }
      return result;
    }

    [[nodiscard]] std::size_t size() const {
      return edgeList.size();
    }

    Edge *get(const EdgeKey &edgeKey) {
      if (!edgeList.contains(edgeKey)) {
        CLOG(WARN_LEVEL,
             std::to_string(edgeKey.first),
             "->",
             std::to_string(edgeKey.second),
             " not found! Returning nullptr.");
        return nullptr;
      }
      return edgeList[edgeKey].get();
    }

  private:
    std::unordered_map<EdgeKey, std::unique_ptr<Edge>, EdgeKeyHash, EdgeKeyEqual> edgeList{};

    Edge *getOrInsert(std::unique_ptr<Edge> edge) {
#ifdef CASET_DEBUG
      if (edge->getSourceId() == edge->getTargetId()) {
        throw std::runtime_error("You cannot create an edge from a vertex to itself: " + edge->toString());
      }
#endif
      const auto edgeKey = edge->getKey();
      if (edgeList.contains(edgeKey)) {
        auto found = edgeList.find(edgeKey)->second.get();
#ifdef CASET_DEBUG
        if (found->getSourceId() != edge->getSourceId() || found->getTargetId() != edge->getTargetId()) {
          throw std::runtime_error(
            "Fingerprint collision between edges: " + edge->toString() + " and " + found->toString());
        }
#endif
        return found;
      }
      // CLOG(DEBUG_LEVEL, "Adding edge: ", edge->toString());
      auto [it, _] = edgeList.insert_or_assign(edge->getKey(), std::move(edge));
      return it->second.get();
    }
};
} // caset

#endif //CASET_EDGELIST_H
