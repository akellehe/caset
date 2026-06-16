// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

//
// Created by andrew on 10/23/25.
//

#include "spacetime/Metric.h"

#include <memory>

#include "spacetime/Signature.h"
#include "Logger.h"

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

    Metric::Metric(bool coordinateFree_, const Signature &signature_) : signature(std::make_shared<Signature>(signature_)), coordinateFree(coordinateFree_) {
    }

    [[nodiscard]] double Metric::getSquaredLength(
      const std::vector<double> &sourceCoords,
      const std::vector<double> &targetCoords
      ) const {

      if (coordinateFree) {
        CLOG(ERROR_LEVEL, "You asked a coordinate free metric to compute the squared length of an edge. That data should be store directly on the edge already.");
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

    [[nodiscard]] const std::shared_ptr<Signature> &Metric::getSignature() const noexcept {
      return signature;
    }

};
