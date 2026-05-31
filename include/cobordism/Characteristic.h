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
/// Observable measuring the Euler characteristic of a triangulation: the
/// alternating count of its cells,
/// \f$ \chi = (\text{vertices}) - (\text{edges}) + (\text{triangles}) - \cdots
/// = \sum_{k} (-1)^k\, |C_k| \f$. It is one of the most basic topological
/// invariants — two triangulations of the same shape always give the same
/// value (for example every triangulation of a 2-sphere has \f$ \chi = 2 \f$).
/// Computed from the chain complex's face counts.
class EulerCharacteristic : public Observable {
  public:
    double compute(const std::shared_ptr<Spacetime> &spacetime) override;
};

/// # Signature
///
/// Observable measuring the signature of a closed, orientable 4-dimensional
/// manifold. The manifold's two-dimensional "holes" carry a symmetric pairing
/// (the *intersection form*): given two such holes it returns an integer
/// counting how they cross. The signature is
/// \f$ \sigma = (\text{number of positive directions}) - (\text{number of
/// negative directions}) \f$ of that pairing — equivalently the count of
/// positive minus negative eigenvalues. It is what tells apart manifolds that
/// share the same Euler characteristic and homology (and is the key to telling
/// two different fillings of the same boundary apart).
///
/// Returns 0 when there are no two-dimensional holes, or when the manifold is
/// not 4-dimensional. See :func:`ChainComplex::signature`.
class Signature : public Observable {
  public:
    double compute(const std::shared_ptr<Spacetime> &spacetime) override;
};

/// # Characteristic numbers (Capability A)
///
/// A bundle of topological invariants of a closed PL \f$ n \f$-manifold. The
/// invariants that are a single number (Euler characteristic, signature) are
/// also available individually as Observables above; this struct additionally
/// carries the ones that are *families* of numbers (Stiefel–Whitney and
/// Pontryagin numbers).
struct CharacteristicNumbers {
  /// Euler characteristic (see EulerCharacteristic above).
  int euler{0};

  /// Signature (see Signature above). Present only for an orientable
  /// 4-manifold; left empty for a non-orientable 4-manifold or any dimension
  /// other than 4, where it is not defined.
  std::optional<int> signature{};

  /// Stiefel–Whitney numbers: a family of yes/no (mod-2) invariants that detect
  /// orientability and related "twisting" of the manifold. Each entry is keyed
  /// by the characteristic-class monomial it comes from (for example the key
  /// "w1^2" is the number obtained from the first Stiefel–Whitney class
  /// squared), with value in \f$ \{0, 1\} \f$.
  /// @note Not computed yet — pending the Stiefel–Whitney / Wu-class work; the
  ///   map is currently always empty.
  std::map<std::string, int> stiefelWhitneyNumbers{};

  /// Pontryagin numbers: integer invariants coming from curvature, defined for
  /// orientable manifolds whose dimension is a multiple of 4. In dimension 4
  /// there is only one, conventionally keyed "p1", and the Hirzebruch signature
  /// theorem says it equals three times the signature
  /// (\f$ \langle p_1, [K]\rangle = 3\sigma \f$).
  std::map<std::string, long> pontryaginNumbers{};

  /// Compute the characteristic numbers of the manifold \a K. For an orientable
  /// 4-manifold this fills in the signature and the Pontryagin number
  /// \f$ p_1 = 3\sigma \f$. The \a oriented flag will later select which
  /// invariants to report (orientation-dependent ones vs. the mod-2
  /// Stiefel–Whitney numbers) once those are implemented.
  static CharacteristicNumbers of(const Spacetime &K, bool oriented = true);
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_CHARACTERISTIC_H
