//
// Created by andrew on 10/23/25.
//

#ifndef CASET_EDGELIST_H
#define CASET_EDGELIST_H

#include <memory>
#include <vector>

#include "Edge.h"

namespace caset {
class EdgeList {
  public:
    std::shared_ptr<Edge> operator[](int index) {
      return edgeList[index];
    }

    std::shared_ptr<Edge> get(int index) {
      return edgeList[index];
    }

    void add(const std::shared_ptr<Edge> &edge) {
      edgeList.push_back(edge);
    }

    void add(const std::shared_ptr<Vertex> &src, const std::shared_ptr<Vertex> &tgt) noexcept {
      edgeList.emplace_back(std::make_shared<Edge>(src, tgt));
    }

    void add(const std::shared_ptr<Vertex> &src, const std::shared_ptr<Vertex> &tgt, double squaredLength) noexcept {
      edgeList.emplace_back(std::make_shared<Edge>(src, tgt, squaredLength));
    }

    std::size_t size() const {
      return edgeList.size();
    }

  private:
    std::vector<std::shared_ptr<Edge>> edgeList;
};
} // caset

#endif //CASET_EDGELIST_H