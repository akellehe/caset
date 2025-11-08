//
// Created by andrew on 10/22/25.
//

#ifndef CASET_SIGNATURE_H
#define CASET_SIGNATURE_H

#include <array>

namespace caset {

enum class SignatureType : uint8_t {
  Lorentzian = 0,
  Euclidean = 1
};

template<int N, SignatureType signatureType>
struct Signature {
  static inline const double c = 1.;
  static inline std::array<int, N> diag = [] {
    std::array<int, N> d{};
    d[0] = signatureType == SignatureType::Euclidean ? 1 : (signatureType == SignatureType::Lorentzian ? -1 : -1); // Default to Lorentzian
    for (int i = 1; i < N; ++i) {
      d[i] = 1; // Spatial dimensions
    }
    return d;
  }();
  static void set(const std::array<int, N> &s) noexcept {diag = s;}
};

} // caset

#endif //CASET_SIGNATURE_H