//
// Created by andrew on 10/23/25.
//

#ifndef CASET_METRIC_H
#define CASET_METRIC_H

#include <memory>

#include "Edge.h"
#include "Signature.h"

namespace caset {
template<int N>
class Metric {
  public:
    using Sig = caset::Signature<N>;

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
  /// Note that the CDT (Causal Dynamical Triangulations) approach typically uses fixed length spacelike edges to
  /// build (and update) the triangulation while Regge Calculus allows for dynamically updated edge lengths. See
  /// Quantum Gravity from Causal Dynamical Triangulations: A Review by R. Loll Section 4, p 11-12 for more details.
  ///
  const double getLength(const std::shared_ptr<caset::Edge<N>> &edge) const noexcept {
    double lengthSquared = 0.0;
    const auto &sourceCoords = edge->getSource()->getCoordinates();
    const auto &targetCoords = edge->getTarget()->getCoordinates();
    for (int i = 0; i < N; ++i) {
      double delta = sourceCoords[i] - targetCoords[i];
      lengthSquared += static_cast<double>(Sig::diag[i]) * delta * delta;
    }
    return lengthSquared;
  }
};
} // caset

#endif //CASET_METRIC_H