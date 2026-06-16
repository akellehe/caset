// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

//
// Created by andrew on 10/22/25.
//

#include "spacetime/Signature.h"

#include <vector>
#include <cstdint>

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
Signature::Signature(int dimensions_, SignatureType signatureType_) {
  dimensions = dimensions_;
  signatureType = signatureType_;
  diag = std::vector<int>(dimensions_);
  std::fill_n(diag.begin(), dimensions_, 1);
  if (signatureType_ == SignatureType::Lorentzian) {
    diag[0] = -1;
  }
}

[[nodiscard]] const std::vector<int> &Signature::getDiagonal() const noexcept {
  return diag;
}

[[nodiscard]] int Signature::getDimensions() const noexcept {
  return dimensions;
}

[[nodiscard]] SignatureType Signature::getSignatureType() const noexcept {
  return signatureType;
}
} // namespace tessera::spacetime
