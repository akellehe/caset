//
// Created by andrew on 10/23/25.
//

#ifndef CASET_METRIC_H
#define CASET_METRIC_H

#include <memory>

#include "Edge.h"
#include "Signature.h"

namespace caset {
class Metric {
  public:
    Metric(bool coordinateFree_, Signature &signature_) : signature(signature_), coordinateFree(coordinateFree_) {
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
    [[nodiscard]] double getSquaredLength(const std::shared_ptr<Edge> &edge) const noexcept {
      if (coordinateFree) {
        return edge->getSquaredLength();
      }
      auto diag = signature.getDiagonal();
      double lengthSquared = 0.0;
      const auto &sourceCoords = edge->getSource()->getCoordinates();
      const auto &targetCoords = edge->getTarget()->getCoordinates();
      for (int i = 0; i < diag.size(); ++i) {
        double delta = sourceCoords[i] - targetCoords[i];
        lengthSquared += static_cast<double>(diag[i]) * delta * delta;
      }
      return lengthSquared;
    }

    ///
    /// This method isn't perfect. But neither is the convention in existing implementations. Not all implementations
    /// have a concept of imaginary edge lengths, and none of which I'm currently aware handle light-like edges.
    ///
    /// Passing an edge with \f$ ds^2 = 0 \f$ results in an out_of_range error.
    ///
    /// @param edge The edge for which you would like to understand it's disposition.
    /// @return The disposition (space/time/light) of the edge.
    EdgeDisposition getDisposition(const std::shared_ptr<Edge> &edge) const {
      double sourceTime = edge->getSource()->getTime();
      double targetTime = edge->getTarget()->getTime();
      if (coordinateFree) {
        if (sourceTime == targetTime) {
          return EdgeDisposition::Timelike;
        }
        return EdgeDisposition::Spacelike;
      }
      double squaredLength = edge->getSquaredLength();
      if (squaredLength == 0) {
        throw std::out_of_range("light-like edges are not currently handled by Metric::getDisposition.");
      }

      auto diag = signature.getDiagonal();
      if (diag[0] < 0) {
        if (squaredLength < 0) {
          return EdgeDisposition::Spacelike;
        }
        return EdgeDisposition::Timelike;
      }
      if (squaredLength < 0) {
        return EdgeDisposition::Spacelike;
      }
      return EdgeDisposition::Timelike;
    }

  private:
    Signature signature;
    bool coordinateFree;
};
} // caset

#endif //CASET_METRIC_H
