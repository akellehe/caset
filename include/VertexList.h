//
// Created by andrew on 10/23/25.
//

#ifndef CASET_VERTEXLIST_H
#define CASET_VERTEXLIST_H

#include <memory>
#include <vector>

#include "Vertex.h"

namespace caset {
template<int N>
class VertexList {
  public:
    std::shared_ptr<Vertex<N>> operator[](int index) {
      return vertexList[index];
    }
  private:
    std::vector<std::shared_ptr<Vertex<N>>> vertexList;
};
} // caset

#endif //CASET_VERTEXLIST_H