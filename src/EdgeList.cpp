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


#include <memory>
#include <ranges>
#include <vector>
#include <unordered_map>

#include "EdgeList.h"
#include "Vertex.h"
#include "Edge.h"
#include "Logger.h"
#include "Simplex.h"

namespace caset {
    Edge *EdgeList::add(std::unique_ptr<Edge> edge) {
      const auto key = edge->getKey(); // compute key *before* move
      const auto found = edgeList.find(key);
      if (found == edgeList.end()) {
        auto [it, inserted] = edgeList.emplace(key, std::move(edge));
#ifdef CASET_DEBUG
        if (!inserted)
          throw std::runtime_error("You attempted to overwrite an edge even though it didn't appear to exist!");
#endif
        return it->second.get();
      }
#ifdef CASET_DEBUG
      CLOG(CRITICAL_LEVEL, "---------------------------------------------------------------------------------------------------");
      CLOG(CRITICAL_LEVEL, "You attempted to insert an edge, ", edge->toString(), " that already exists!");
      CLOG(CRITICAL_LEVEL, "The existing edge was ", found->second->toString(), " the key to insert it at was ", key.toString());
      CLOG(CRITICAL_LEVEL, "The incoming unique pointer is: ", static_cast<const void *>(&edge));
      CLOG(CRITICAL_LEVEL, "The incoming raw pointer it wraps is: ", static_cast<const void *>(edge.get()));
      CLOG(CRITICAL_LEVEL, "The existing unique pointer is: ", static_cast<const void *>(&found->second));
      CLOG(CRITICAL_LEVEL, "The existing raw pointer it wraps is: ", static_cast<const void *>(found->second.get()));
      CLOG(CRITICAL_LEVEL, "---------------------------------------------------------------------------------------------------");
      throw std::runtime_error("You attempted to insert an edge that already exists. This will result in un unexpected free()");
#endif
      // edge->copyInPlaceTo(found->second.get());
      // return found->second.get();
    }

    Edge *EdgeList::add(VertexPtr src, VertexPtr tgt) {
#ifdef CASET_DEBUG
      if (src->getId() == 0 || tgt->getId() == 0) {
        throw std::runtime_error("Invalid src and target ids!");
      }
      if (src == tgt) {
        CLOG(WARN_LEVEL, "Source and target were equal (self-referential)");
        throw std::runtime_error("Invalid src and target ids!");
      }
      if (get({src->getId(), tgt->getId()}) != nullptr) {
        CLOG(WARN_LEVEL, "An edge with this source and target already exists!");
      }
#endif
      auto edge = std::make_unique<Edge>(src, tgt);
      return getOrInsert(std::move(edge));
    }

    Edge *EdgeList::add(VertexPtr src, VertexPtr tgt, double squaredLength) {
#ifdef CASET_DEBUG
      if (src->getId() == 0 || tgt->getId() == 0) {
        throw std::runtime_error("Invalid src and target ids!");
      }
      if (src == tgt) {
        CLOG(WARN_LEVEL, "Source and target were equal (self-referential)");
      throw std::runtime_error("Invalid src and target ids!");
      }
      if (get({src->getId(), tgt->getId()}) != nullptr) {
        CLOG(WARN_LEVEL, "An edge with this source and target already exists!");
      }
#endif
      auto edge = std::make_unique<Edge>(src, tgt, squaredLength);
      return getOrInsert(std::move(edge));
    }

    std::unique_ptr<Edge> EdgeList::remove(const EdgeKey &edgeKey) noexcept {
      auto it = edgeList.find(edgeKey);
      if (it == edgeList.end()) return nullptr;
#ifdef CASET_DEBUG
      it->second->assertUnused();
#endif
      std::unique_ptr<Edge> removed = std::move(it->second);
      edgeList.erase(it);
      return removed;
    }

    Edge* EdgeList::updateKey(const EdgeKey &oldKey) {
      auto node = edgeList.extract(oldKey);
      if (!node.empty()) {
        EdgeKey newKey(node.mapped()->getSource()->getId(), node.mapped()->getTarget()->getId());

        // Check if an edge with the new key already exists
        auto existing = edgeList.find(newKey);
        if (existing != edgeList.end()) {
          CLOG(CRITICAL_LEVEL, "Edge with (new) key ", newKey.toString(), " already exists during updateKey for old key ", oldKey.toString(), ".");
          CLOG(CRITICAL_LEVEL, "This means there already existed a canonical edge at the time a would-be canonical edge was redirected. We need to replace the non-canonical edge with the canonical one.");
          node.mapped()->replaceOnReferents(existing->second.get());
          return existing->second.get();
        }

        // No conflict - modify the edge, update the key and reinsert
        node.key() = newKey;
        auto [it, inserted, _] = edgeList.insert(std::move(node));
#ifdef CASET_DEBUG
        if (!inserted) {
          throw std::runtime_error("Failed to reinsert edge with updated key: " +
            newKey.toString());
        }
#endif
        CLOG(DEBUG_LEVEL, "Updated key from ", oldKey.toString(), " to ", newKey.toString());
        return it->second.get();
      }
      CLOG(DEBUG_LEVEL, "Old edge was not found!");
      return nullptr;
    }

    std::unique_ptr<Edge> EdgeList::remove(Edge *edge) noexcept {
      return remove(edge->getKey());
    }

    [[nodiscard]] std::vector<Edge *> EdgeList::toVector() const noexcept {
      std::vector<Edge *> result;
      result.reserve(edgeList.size());
      for (const auto &edge : edgeList | std::views::values) {
        result.push_back(edge.get());
      }
      return result;
    }

    [[nodiscard]] std::size_t EdgeList::size() const {
      return edgeList.size();
    }

    Edge *EdgeList::get(const EdgeKey &edgeKey) {
      auto found = edgeList.find(edgeKey);
      if (found == edgeList.end()) {
        CLOG(WARN_LEVEL,
             std::to_string(edgeKey.first),
             "->",
             std::to_string(edgeKey.second),
             " not found! Returning nullptr.");
        return nullptr;
      }
      return found->second.get();
    }


    Edge *EdgeList::getOrInsert(std::unique_ptr<Edge> edge) {
#ifdef CASET_DEBUG
      if (edge->getSource() == edge->getTarget()) {
        throw std::runtime_error("You cannot create an edge from a vertex to itself: " + edge->toString());
      }
#endif
      const auto found = edgeList.find(edge->getKey());
      if (found != edgeList.end()) {
#ifdef CASET_DEBUG
        if (found->second == nullptr) {
          throw std::runtime_error("Found a key, but the corresponding edge was a nullptr.");
        }
        if (found->second->getSource() != edge->getSource() || found->second->getTarget() != edge->getTarget()) {
          throw std::runtime_error(
            "Fingerprint collision between edges: " + edge->toString() + " and " + found->second->toString());
        }
#endif
        return found->second.get();
      }

      auto [it, inserted] = edgeList.emplace(edge->getKey(), std::move(edge));
#ifdef CASET_DEBUG
      if (!inserted) {
        throw std::runtime_error("You attempted to overwrite an edge!");
      }
#endif
      return it->second.get();
    }
} // caset
