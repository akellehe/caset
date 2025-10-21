//
// Created by Andrew Kelleher on 10/20/25.
//

#ifndef CASET_CASET_SRC_EDGE_H_
#define CASET_CASET_SRC_EDGE_H_

#include "Vertex.h"

#include <memory>

/// # Edge Class
/// An edge that links two points (vertices) in spacetime.
///
/// - Stores references to the source and target vertices.
///
/// The edge has weight \f$ w_i \f$, which can represent various physical
/// properties depending on the context (e.g., distance, time interval, etc.).
///
///
namespace caset {
class Edge {
  public:
    Edge(
      std::shared_ptr<Vertex> source_,
      std::shared_ptr<Vertex> target_,
      double weight_)
      : source(source_), target(target_), weight(weight_) {
    }

    /// Returns the weight of the edge, \f$ w_i \f$
    ///
    double getWeight() const {
      return weight;
    }

  private:
    std::shared_ptr<Vertex> source;
    std::shared_ptr<Vertex> target;

    double weight;
};
}

#endif //CASET_CASET_SRC_EDGE_H_
