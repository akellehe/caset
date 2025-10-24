//
// Created by andrew on 10/23/25.
//

#ifndef CASET_SPACETIME_H
#define CASET_SPACETIME_H

#include <memory>

#include "EdgeList.h"
#include "VertexList.h"

namespace caset {
template<int N>
class Spacetime {
  public:
    std::shared_ptr<EdgeList<N>> getEdgeList() {
      return edgeList;
    }
    std::shared_ptr<VertexList<N>> getVertexList() {
      return vertexList;
    }
  private:
    std::shared_ptr<EdgeList<N>> edgeList;
    std::shared_ptr<VertexList<N>> vertexList;

};
} // caset

#endif //CASET_SPACETIME_H