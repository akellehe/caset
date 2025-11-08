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
    void add(const std::shared_ptr<Vertex> &vertex) {
      vertexList.push_back(vertex);
    }
  private:
    std::vector<std::shared_ptr<Vertex>> vertexList;
};
} // caset

#endif //CASET_VERTEXLIST_H