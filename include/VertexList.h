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
    std::shared_ptr<Vertex> add(const std::shared_ptr<Vertex> &vertex) noexcept {
      vertexList.push_back(vertex);
      return vertex;
    }
    std::shared_ptr<Vertex> add(const std::size_t id, const std::vector<double> &coords) noexcept {
      vertexList.emplace_back(std::make_shared<Vertex>(id, coords));
      return vertexList.back();
    }
    std::shared_ptr<Vertex> add(const std::size_t id) noexcept {
      vertexList.emplace_back(std::make_shared<Vertex>(id));
      return vertexList.back();
    }
  private:
    std::vector<std::shared_ptr<Vertex>> vertexList{};
};
} // caset

#endif //CASET_VERTEXLIST_H