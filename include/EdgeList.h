//
// Created by andrew on 10/23/25.
//

#ifndef CASET_EDGELIST_H
#define CASET_EDGELIST_H

#include <memory>
#include <vector>
#include <unordered_set>

#include "Edge.h"
#include "Logger.h"

namespace caset {
class EdgeList {
  public:
    std::shared_ptr<Edge> add(const std::shared_ptr<Edge> &edge) {
      return getOrInsert(edge);
    }

    std::shared_ptr<Edge> add(std::uint64_t src, std::uint64_t tgt) {
      auto edge = std::make_shared<Edge>(src, tgt);
      return getOrInsert(edge);
    }

    std::shared_ptr<Edge> add(std::uint64_t src, std::uint64_t tgt, double squaredLength) noexcept {
      auto edge = std::make_shared<Edge>(src, tgt, squaredLength);
      return getOrInsert(edge);
    }

    void remove(const EdgeKey &edgeKey) noexcept {
      const auto &[srcId, tgtId] = edgeKey;
      auto tempEdge = std::make_shared<Edge>(srcId, tgtId);
      if (edgeList.contains(tempEdge)) {
        remove(*edgeList.find(tempEdge));
      } else {
        CLOG(WARN_LEVEL, "-----------------------------------------------");
        CLOG(WARN_LEVEL, "Edge: ", tempEdge->toString(), " not found in: ");
        for (const auto &e : edgeList) {
          CLOG(WARN_LEVEL, "    - ", e->toString());
        }
        CLOG(WARN_LEVEL, "----------------------------------------------");
      }
    }

    void remove(const EdgePtr &edge) noexcept {
      if (!edgeList.contains(edge)) {
        CLOG(WARN_LEVEL, "You attempted to remove an edge that does not exist: ", edge->toString());
        for (const auto &e : edgeList) {
          CLOG(WARN_LEVEL, "    - ", e->toString());
        }
        return;
      }
      edgeList.erase(edge);
    }

    void replace(std::shared_ptr<Edge> &toRemove, std::shared_ptr<Edge> &toAdd) noexcept {
      edgeList.erase(toRemove);
      edgeList.insert(toAdd);
    }

    [[nodiscard]] std::vector<std::shared_ptr<Edge> > toVector() const noexcept {
      std::vector<std::shared_ptr<Edge>> result;
      result.reserve(edgeList.size());
      for (auto &edge : edgeList) {
        result.push_back(edge);
      }
      return result;
    }

    [[nodiscard]] std::size_t size() const {
      return edgeList.size();
    }

    std::shared_ptr<Edge> get(const EdgeKey &edgeKey) const {
      const auto &[srcId, tgtId] = edgeKey;
      auto tempEdge = std::make_shared<Edge>(srcId, tgtId);
      if (!edgeList.contains(tempEdge)) {
        CLOG(WARN_LEVEL, tempEdge->toString(), " not found! Returning nullptr.");
        return nullptr;
      }
      return *edgeList.find(tempEdge);
    }

  private:
    std::unordered_set<std::shared_ptr<Edge>, EdgeHash, EdgeEq> edgeList{};

    std::shared_ptr<Edge> getOrInsert(const std::shared_ptr<Edge> &edge) {
      if (edge->getSourceId() == edge->getTargetId()) {
        throw std::runtime_error("You cannot create an edge from a vertex to itself: " + edge->toString());
      }
      if (edgeList.contains(edge)) {
        auto found = *edgeList.find(edge);
        if (found->getSourceId() != edge->getSourceId() || found->getTargetId() != edge->getTargetId()) {
          throw std::runtime_error("Fingerprint collision between edges: " + edge->toString() + " and " + found->toString());
        }
        return found;
      }
      CLOG(DEBUG_LEVEL, "Adding edge: ", edge->toString());
      edgeList.insert(edge);
      return edge;
    }
};
} // caset

#endif //CASET_EDGELIST_H
