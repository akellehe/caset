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
    std::shared_ptr<caset::Edge<N>> operator[](int index) {
      return edgeList[index];
    }

    std::size_t size() const {
      return edgeList.size();
    }

  private:
    std::vector<std::shared_ptr<caset::Edge<N>>> edgeList;
};
} // caset

#endif //CASET_EDGELIST_H