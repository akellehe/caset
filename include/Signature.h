// MIT License
// Copyright (c) 2025 Andrew Kelleher
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//
// Created by andrew on 10/22/25.
//

#ifndef CASET_SIGNATURE_H
#define CASET_SIGNATURE_H

#include <vector>
#include <cstdint>

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

    [[nodiscard]] std::vector<int> getDiagonal() const noexcept {
      return diag;
    };

    [[nodiscard]] int getDimensions() const noexcept {
      return dimensions;
    }

    [[nodiscard]] SignatureType getSignatureType() const noexcept {
      return signatureType;
    }

  private:
    std::vector<int> diag;
    int dimensions;
    SignatureType signatureType;

  static inline const double c = 1.;

};

} // caset

#endif //CASET_SIGNATURE_H