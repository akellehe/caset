//
// Created by andrew on 10/22/25.
//

#ifndef CASET_SIGNATURE_H
#define CASET_SIGNATURE_H

#include <vector>

namespace caset {

enum class SignatureType : uint8_t {
  Lorentzian = 0,
  Euclidean = 1
};

class Signature {
  public:
    Signature(int dimensions_, SignatureType signatureType_) {
      dimensions = dimensions_;
      signatureType = signatureType_;
      diag = std::vector<int>(dimensions_);
      std::fill_n(diag.begin(), dimensions_, 1);
      if (signatureType_ == SignatureType::Lorentzian) {
        diag[0] = -1;
      }
    }

    std::vector<int> getDiagonal() const noexcept {
      return diag;
    };

  private:
    std::vector<int> diag;
    int dimensions;
    SignatureType signatureType;

  static inline const double c = 1.;

};

} // caset

#endif //CASET_SIGNATURE_H