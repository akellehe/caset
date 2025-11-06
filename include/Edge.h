#ifndef CASET_CASET_SRC_EDGE_H_
#define CASET_CASET_SRC_EDGE_H_

#include "Vertex.h"
#include "Signature.h"

#include <memory>

namespace caset {
/// # Edge Class
/// An edge that links two points (vertices) in spacetime.
///
template<int N>
class Edge {
  public:
    static_assert(N >= 1 && N <= 11, "Dimension N must be between 1 and 11");

    Edge(
      std::shared_ptr<caset::Vertex<N> > source_,
      std::shared_ptr<caset::Vertex<N> > target_
    ) : source(source_), target(target_) {}

    const std::shared_ptr<caset::Vertex<N> > getSource() const noexcept {
      return source;
    }

    const std::shared_ptr<caset::Vertex<N> > getTarget() const noexcept {
      return target;
    }

  private:
    std::shared_ptr<caset::Vertex<N> > source;
    std::shared_ptr<caset::Vertex<N> > target;
};
}

#endif //CASET_CASET_SRC_EDGE_H_
