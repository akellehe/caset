//
// Created by andrew on 10/23/25.
//

#ifndef CASET_EDGELIST_H
#define CASET_EDGELIST_H

#include <memory>
#include <vector>
#include "Edge.h"

namespace caset {
template<int N>
class EdgeList {
  public:
    std::shared_ptr<Edge<N>> operator[](int index) {
      return edgeList[index];
    }

  private:
    std::vector<std::shared_ptr<Edge<N>>> edgeList;
};
} // caset

#endif //CASET_EDGELIST_H