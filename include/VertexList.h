//
// Created by andrew on 10/23/25.
//

#ifndef CASET_VERTEXLIST_H
#define CASET_VERTEXLIST_H

#include <memory>
#include <vector>

#include "Vertex.h"

namespace caset {
class VertexList {
  public:
    std::shared_ptr<Vertex> operator[](int index) {
      return vertexList[index];
    }
    void add(const std::shared_ptr<Vertex> &vertex) noexcept {
      vertexList.push_back(vertex);
    }
    void add(const std::size_t id, const std::vector<double> &coords) noexcept {
      vertexList.emplace_back(std::make_shared<Vertex>(id, coords));
    }
    void add(const std::size_t id) noexcept {
      vertexList.emplace_back(std::make_shared<Vertex>(id));
    }
  private:
    std::vector<std::shared_ptr<Vertex>> vertexList{};
};
} // caset

#endif //CASET_VERTEXLIST_H