#ifndef CASET_CASET_SRC_EDGE_H_
#define CASET_CASET_SRC_EDGE_H_

#include "Vertex.h"
#include "Signature.h"

#include <random>
#include <algorithm>
#include <memory>

inline double random_uniform(double min = -1.0, double max = 1.0) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(min, max);
  return dist(gen);
}

namespace caset {
/// # Edge Class
/// An edge that links two points (vertices) in spacetime.
///
class Edge {
  public:
    Edge(
      std::shared_ptr<Vertex> source_,
      std::shared_ptr<Vertex> target_,
      double squaredLength_
    ) : source(source_), target(target_), squaredLength(squaredLength_) {}

    Edge(
      std::shared_ptr<Vertex> source_,
      std::shared_ptr<Vertex> target_
    ) : source(source_), target(target_) {

      // Set squaredLength to a random value between -1 and 1
      squaredLength = random_uniform(); // TODO: Should we use a poisson dist here for coset theory?
    }

    const std::shared_ptr<Vertex> getSource() const noexcept {
      return source;
    }

    const std::shared_ptr<Vertex> getTarget() const noexcept {
      return target;
    }

    const double getSquaredLength() const noexcept {
      return squaredLength;
    }

  private:
    std::shared_ptr<Vertex> source;
    std::shared_ptr<Vertex> target;
    double squaredLength;
};
}

#endif //CASET_CASET_SRC_EDGE_H_
