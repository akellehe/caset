//
// Created by Andrew Kelleher on 10/20/25.
//

#ifndef CASET_CASET_SRC_EDGE_H_
#define CASET_CASET_SRC_EDGE_H_

#include "Vertex.h"

#include <memory>

namespace caset {
/// # Edge Class
/// An edge that links two points (vertices) in spacetime.
///
/// - Stores references to the source and target vertices.
///
/// The edge has length \f$ l_{ij} \f$, which can represent various physical
/// properties depending on the context (e.g., distance, time interval, etc.).
///
///
class Edge {
  public:
    Edge(
      std::shared_ptr<Vertex> source_,
      std::shared_ptr<Vertex> target_,
      double length_)
      : source(source_), target(target_), length(length_) {
    }

    /// Returns the length of the edge, \f$ l_{ij} \f$
    ///
    double getLength() const {
      return length;
    }

  private:
    std::shared_ptr<Vertex> source;
    std::shared_ptr<Vertex> target;

    double length;
};
}

#endif //CASET_CASET_SRC_EDGE_H_
