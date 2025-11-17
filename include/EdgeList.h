//
// Created by andrew on 10/23/25.
//

#ifndef CASET_EDGELIST_H
#define CASET_EDGELIST_H

#include <memory>
#include <vector>
#include <unordered_set>

#include "Edge.h"

namespace caset {
class EdgeList {
  public:
    std::shared_ptr<Edge> add(const std::shared_ptr<Edge> &edge) {
      if (edgeList.contains(edge)) {
        return *edgeList.find(edge);
      }
      edgeList.insert(edge);
      return edge;
    }

    std::shared_ptr<Edge> add(std::uint64_t src, std::uint64_t tgt) noexcept {
      auto edge = std::make_shared<Edge>(src, tgt);
      if (edgeList.contains(edge)) {
        return *edgeList.find(edge);
      }
      edgeList.insert(edge);
      return edge;
    }

    std::shared_ptr<Edge> add(std::uint64_t src, std::uint64_t tgt, double squaredLength) noexcept {
      auto edge = std::make_shared<Edge>(src, tgt, squaredLength);
      if (edgeList.contains(edge)) {
        return *edgeList.find(edge);
      }
      edgeList.insert(edge);
      return edge;
    }

    void remove(const std::shared_ptr<Edge> &edge) noexcept {
      edgeList.erase(edge);
    }

    void replace(std::shared_ptr<Edge> &toRemove, std::shared_ptr<Edge> &toAdd) noexcept {
      edgeList.erase(toRemove);
      edgeList.insert(toAdd);
    }

    [[nodiscard]] std::size_t size() const {
      return edgeList.size();
    }

  private:
    std::unordered_set<std::shared_ptr<Edge>, EdgeHash, EdgeEq> edgeList{};
};
} // caset

#endif //CASET_EDGELIST_H