//
// Created by andrew on 10/23/25.
//

#ifndef CASET_METRIC_H
#define CASET_METRIC_H

#include <memory>

#include "Signature.h"
#include "Logger.h"

namespace caset {
/// # The Metric
///
class Metric {
  public:
    Metric(bool coordinateFree_, Signature &signature_) : signature(std::make_shared<Signature>(signature_)), coordinateFree(coordinateFree_) {
    }

    ///
    /// This method computes the length of the edge between the source and target vertices when we're using a coordinate
    /// system/euclidean metric. This uses the metric, \f$ g_{\mu \nu} \f$, to compute the distance between vertex
    /// \f$ i \f$ and vertex \f$ j \f$ as
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
    [[nodiscard]] double getSquaredLength(
      const std::vector<double> &sourceCoords,
      const std::vector<double> &targetCoords
      ) const {

      if (coordinateFree) {
        CASET_LOG(ERROR_LEVEL, "You asked a coordinate free metric to compute the squared length of an edge. That data should be store directly on the edge already.");
        throw std::runtime_error("You asked a coordinate free metric to compute the squared length of an edge. That data should be store directly on the edge already.");
      }

      auto diag = signature->getDiagonal();
      double lengthSquared = 0.0;
      for (int i = 0; i < diag.size(); ++i) {
        double delta = sourceCoords[i] - targetCoords[i];
        lengthSquared += static_cast<double>(diag[i]) * delta * delta;
      }
      return lengthSquared;
    }

    [[nodiscard]] std::shared_ptr<Signature> getSignature() const noexcept {
      return signature;
    }

  private:
    std::shared_ptr<Signature> signature;
    bool coordinateFree;
};
} // caset

#endif //CASET_METRIC_H
