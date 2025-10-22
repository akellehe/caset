//
// Created by andrew on 10/22/25.
//

#ifndef CASET_SIGNATURE_H
#define CASET_SIGNATURE_H

#include <array>

namespace caset {

template<int N>
struct Signature {
  static inline const double c = 1.;
  static inline std::array<int, N> diag = [] {
    std::array<int, N> d{};
    if constexpr (N>0) d[0] = -1; // Time dimension
    for (int i = 1; i < N; ++i) {
      d[i] = 1; // Spatial dimensions
    }
    return d;
  }();
  static void set(const std::array<int, N> &s) noexcept {diag = s;}
};

} // caset

#endif //CASET_SIGNATURE_H