// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

//
// Created by andrew on 10/23/25.
//

#ifndef TESSERA_METRIC_H
#define TESSERA_METRIC_H

#include <memory>

#include "spacetime/Signature.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;
/// # The Metric
///
class Metric {
  public:
    Metric(bool coordinateFree_, const Signature &signature_);

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
      ) const;

    [[nodiscard]] const std::shared_ptr<Signature> &getSignature() const noexcept;

  private:
    std::shared_ptr<Signature> signature;
    bool coordinateFree;
};
} // namespace tessera::spacetime

#endif //TESSERA_METRIC_H
