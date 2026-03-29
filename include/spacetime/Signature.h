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

/// The signature type of the spacetime metric tensor \f$ g_{\mu\nu} \f$.
///
///   - **Lorentzian** \f$(-,+,+,\ldots,+)\f$: physical Minkowski-type signature
///     with one timelike and \f$ d-1 \f$ spacelike dimensions. Used in CDT.
///   - **Euclidean** \f$(+,+,+,\ldots,+)\f$: all positive signature, obtained after
///     Wick rotation \f$ t \to -i\tau \f$. Used in Euclidean quantum gravity.
enum class SignatureType : uint8_t {
  Lorentzian = 0,
  Euclidean = 1
};

/// # Metric Signature
///
/// Encodes the diagonal of the metric tensor \f$ g_{\mu\nu} \f$ for a
/// \f$ d \f$-dimensional spacetime. The signature determines which edges
/// are timelike (negative squared length) and which are spacelike (positive).
///
/// For a 4D Lorentzian spacetime, the diagonal is \f$ (-1, +1, +1, +1) \f$,
/// giving the Minkowski metric
///
/// \f[
///   ds^2 = g_{\mu\nu}\, dx^\mu\, dx^\nu = -c^2\, dt^2 + dx^2 + dy^2 + dz^2
/// \f]
///
/// with \f$ c = 1 \f$ in natural units.
///
class Signature {
  public:
    /// @param dimensions_ The spacetime dimension \f$ d \f$ (typically 2, 3, or 4)
    /// @param signatureType_ Lorentzian \f$(-,+,\ldots)\f$ or Euclidean \f$(+,+,\ldots)\f$
    Signature(int dimensions_, SignatureType signatureType_);

    /// @return The diagonal entries of \f$ g_{\mu\nu} \f$, e.g., \f$\{-1, 1, 1, 1\}\f$ for 4D Lorentzian.
    [[nodiscard]] const std::vector<int> &getDiagonal() const noexcept;

    /// @return The spacetime dimension \f$ d \f$.
    [[nodiscard]] int getDimensions() const noexcept;

    /// @return The signature type (Lorentzian or Euclidean).
    [[nodiscard]] SignatureType getSignatureType() const noexcept;

  private:
    std::vector<int> diag;
    int dimensions;
    SignatureType signatureType;

    static inline const double c = 1.;
};
} // caset

#endif //CASET_SIGNATURE_H
