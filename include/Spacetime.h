//
// Created by andrew on 10/23/25.
//

#ifndef CASET_SPACETIME_H
#define CASET_SPACETIME_H

#include <memory>

#include "EdgeList.h"
#include "VertexList.h"
#include "Metric.h"

namespace caset {
template<int N>
class Spacetime {
 public:
  using Metric = caset::Metric<N>;

    static std::shared_ptr<EdgeList<N>> getEdgeList() {
      return edgeList;
    }
    static std::shared_ptr<VertexList<N>> getVertexList() {
      return vertexList;
    }

  static inline Metric metric{};  // C++17 inline static data member

 private:
    static inline std::shared_ptr<EdgeList<N>> edgeList;
    static inline std::shared_ptr<VertexList<N>> vertexList;

};
} // caset

#endif //CASET_SPACETIME_H