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

#include "cobordism/Characteristic.h"

#include <exception>

#include "cobordism/ChainComplex.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

double EulerCharacteristic::compute(const std::shared_ptr<Spacetime> &spacetime) {
  if (spacetime == nullptr) return 0.0;
  return static_cast<double>(ChainComplex::fromSpacetime(*spacetime).eulerCharacteristic());
}

double Signature::compute(const std::shared_ptr<Spacetime> &spacetime) {
  if (spacetime == nullptr) return 0.0;
  return static_cast<double>(ChainComplex::fromSpacetime(*spacetime).signature());
}

CharacteristicNumbers CharacteristicNumbers::of(const Spacetime &K, bool oriented) {
  const ChainComplex cc = ChainComplex::fromSpacetime(K);
  CharacteristicNumbers out;
  out.euler = cc.eulerCharacteristic();
  if (oriented && cc.dimension() == 4) {
    try {
      const int signatureValue = cc.signature();
      out.signature = signatureValue;
      // Hirzebruch signature theorem: the first Pontryagin number equals three
      // times the signature.
      out.pontryaginNumbers["p1"] = 3 * static_cast<long>(signatureValue);
    } catch (const std::exception &) {
      // Non-orientable / no fundamental class: signature (hence p1) undefined.
    }
  }
  // Stiefel–Whitney numbers are unoriented invariants — computed regardless of
  // the `oriented` flag. A class genuinely needing a deferred higher Steenrod
  // square (issue #65) leaves the family empty rather than failing the call.
  try {
    out.stiefelWhitneyNumbers = cc.stiefelWhitneyNumbers();
  } catch (const std::exception &) {
    // Higher cup-i square required (deferred) or not a closed manifold.
  }
  return out;
}

}  // namespace tessera::cobordism
