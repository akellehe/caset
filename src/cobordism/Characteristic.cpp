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
  (void)oriented;  // gates Stiefel–Whitney (unoriented) vs oriented numbers once SW lands
  const ChainComplex cc = ChainComplex::fromSpacetime(K);
  CharacteristicNumbers out;
  out.euler = cc.eulerCharacteristic();
  if (cc.dimension() == 4) {
    try {
      const int s = cc.signature();
      out.signature = s;
      // Hirzebruch signature theorem: <p_1, [K]> = 3 sigma.
      out.pontryagin["p1"] = 3 * static_cast<long>(s);
    } catch (const std::exception &) {
      // Non-orientable / no fundamental class: signature (hence p_1) undefined.
    }
  }
  // Stiefel–Whitney numbers: pending the Wu-class / Steenrod-square work.
  return out;
}

}  // namespace tessera::cobordism
