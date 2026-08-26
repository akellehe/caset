// Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tessera::cobordism {

/// Structural boundary role in dW = -M0 + M1.  It is never inferred from a
/// metric value or a spectral sign.
enum class BoundaryRole : std::int8_t {
  Incoming = -1,
  Outgoing = 1,
};

/// Structural choice of normal direction for a separating cut.
enum class Coorientation : std::int8_t {
  Negative = -1,
  Positive = 1,
};

[[nodiscard]] constexpr BoundaryRole reverse(BoundaryRole role) noexcept {
  return role == BoundaryRole::Incoming ? BoundaryRole::Outgoing
                                        : BoundaryRole::Incoming;
}

[[nodiscard]] constexpr Coorientation reverse(Coorientation value) noexcept {
  return value == Coorientation::Positive ? Coorientation::Negative
                                          : Coorientation::Positive;
}

[[nodiscard]] constexpr int boundaryCoefficient(BoundaryRole role) noexcept {
  return static_cast<int>(role);
}

[[nodiscard]] constexpr int coorientationSign(Coorientation value) noexcept {
  return static_cast<int>(value);
}

/// A separating cut described entirely by oriented simplices plus a chosen
/// coorientation.  Metric data deliberately does not appear in this type.
struct CoorientedCut {
  std::string id;
  std::vector<std::vector<std::uint64_t>> orientedSimplices;
  Coorientation coorientation{Coorientation::Positive};

  [[nodiscard]] CoorientedCut reversed() const {
    CoorientedCut result = *this;
    result.coorientation = reverse(coorientation);
    return result;
  }
};

}  // namespace tessera::cobordism
