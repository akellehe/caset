#ifndef CASET_CASET_SRC_EDGE_H_
#define CASET_CASET_SRC_EDGE_H_

#include "Vertex.h"
#include "Signature.h"

#include <memory>

#include "Spacetime.h"

namespace caset {
/// # Edge Class
/// An edge that links two points (vertices) in spacetime.
///
template<int N>
class Edge {
  public:
    static_assert(N >= 1 && N <= 11, "Dimension N must be between 1 and 11");

    Edge(
      std::shared_ptr<Vertex<N> > source_,
      std::shared_ptr<Vertex<N> > target_
    ) : source(source_), target(target_) {}

    /// This method computes the length of the edge between the source and target vertices. This uses the metric,
    /// \f$ g_{\mu \nu} \f$, to compute the distance between vertex \f$ i \f$ and vertex \f$ j \f$ as
    ///
    /// \f[
    /// l_{ij}^2 = g_{\mu \nu} \Delta x^{\mu} \Delta x^{\nu}
    /// \f]
    ///
    /// where
    ///
    /// \f[
    /// \Delta x^{\mu} := x_i^{\mu} - x_j^{\mu}
    /// \f]
    ///
    /// with signature (-,+,+,+).
    ///
    /// Timelike edges will have negative squared lengths, spacelike edges positive squared lengths, and null/lightlike
    /// edges zero squared lengths.
    ///
    const double getLength() const noexcept {
      double lengthSquared = 0.0;
      const auto &sourceCoords = source->getCoordinates();
      const auto &targetCoords = target->getCoordinates();
      for (int i = 0; i < N; ++i) {
        double delta = sourceCoords[i] - targetCoords[i];
        lengthSquared += static_cast<double>(Sig::diag[i]) * delta * delta;
      }
      return lengthSquared;
    }

  private:
    std::shared_ptr<Vertex<N> > source;
    std::shared_ptr<Vertex<N> > target;
};
}

#endif //CASET_CASET_SRC_EDGE_H_
