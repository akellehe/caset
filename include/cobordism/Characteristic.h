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

#ifndef TESSERA_COBORDISM_CHARACTERISTIC_H
#define TESSERA_COBORDISM_CHARACTERISTIC_H

#include <map>
#include <memory>
#include <optional>
#include <string>

#include "observables/Observable.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime { class Spacetime; }
namespace tessera::cobordism {
using namespace ::tessera::observables;
using namespace ::tessera::spacetime;

/// # EulerCharacteristic
///
/// Observable: the Euler characteristic
/// \f$ \chi(K) = \sum_{k=0}^{n} (-1)^k\, |C_k(K)| \f$ of a triangulation.
/// Computed from the chain complex's \f$ f \f$-vector.
class EulerCharacteristic : public Observable {
  public:
    double compute(const std::shared_ptr<Spacetime> &spacetime) override;
};

/// # Signature
///
/// Observable: the signature \f$ \sigma(K) = b_+ - b_- \f$ of the intersection
/// form on \f$ H_2(K;\mathbb{Z}) \f$, for a closed oriented 4-manifold. Returns
/// 0 for \f$ n \neq 4 \f$ or \f$ b_2 = 0 \f$. See
/// :func:`ChainComplex::signature`.
class Signature : public Observable {
  public:
    double compute(const std::shared_ptr<Spacetime> &spacetime) override;
};

/// # Characteristic numbers (Capability A)
///
/// The full set of characteristic numbers of a closed PL \f$ n \f$-manifold.
/// Scalars that are naturally Observables (\f$ \chi \f$, \f$ \sigma \f$) are
/// also exposed as such; this aggregate additionally carries the families that
/// are not a single scalar (Stiefel–Whitney and Pontryagin numbers).
struct CharacteristicNumbers {
  /// Euler characteristic \f$ \chi \f$.
  int euler{0};
  /// Signature \f$ \sigma \f$ (defined for \f$ n = 4 \f$; unset for a
  /// non-orientable 4-manifold or \f$ n \neq 4 \f$).
  std::optional<int> signature{};
  /// Stiefel–Whitney numbers \f$ \langle w_{i_1}\cdots w_{i_j}, [K]\rangle \in
  /// \mathbb{Z}/2 \f$, keyed by monomial (e.g. "w1^2", "w2^2").
  /// @note Not yet computed (pending the Wu-class / Steenrod-square work); empty.
  std::map<std::string, int> sw{};
  /// Pontryagin numbers (oriented, \f$ n = 4k \f$). For \f$ n = 4 \f$ the only
  /// one is \f$ \langle p_1, [K]\rangle = 3\sigma \f$ (Hirzebruch signature
  /// theorem), keyed "p1".
  std::map<std::string, long> pontryagin{};

  /// Compute the characteristic numbers of \f$ K \f$. When \f$ n = 4 \f$ and
  /// the manifold is closed-orientable, fills \f$ \sigma \f$ and
  /// \f$ p_1 = 3\sigma \f$. (The \a oriented flag will gate oriented vs
  /// unoriented invariants once Stiefel–Whitney numbers are added.)
  static CharacteristicNumbers of(const Spacetime &K, bool oriented = true);
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_CHARACTERISTIC_H
