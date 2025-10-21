//
// Created by Andrew Kelleher on 10/20/25.
//

#ifndef CASET_CASET_SRC_EDGE_H_
#define CASET_CASET_SRC_EDGE_H_

#include "Vertex.h"

#include <memory>

class Edge {

 public:
  Edge(
      std::shared_ptr<Vertex> source_,
      std::shared_ptr<Vertex> target_,
      double weight_)
      : source(source_), target(target_), weight(weight_) {}

  double getWeight() const {
    return weight;
  }

 private:
  std::shared_ptr<Vertex> source;
  std::shared_ptr<Vertex> target;

  double weight;
};

#endif //CASET_CASET_SRC_EDGE_H_
