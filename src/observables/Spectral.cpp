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

#include "observables/Spectral.h"

#include <cstddef>
#include <vector>

#include "cobordism/HodgeLaplacian.h"
#include "spacetime/Spacetime.h"

namespace tessera::observables {

double SpectralGap::compute(const std::shared_ptr<Spacetime> &spacetime) {
  if (spacetime == nullptr) return 0.0;
  // Eigenvalues are ascending; the first gap is lambda_1 - lambda_0.
  const std::vector<double> evals =
      ::tessera::cobordism::HodgeLaplacian(spacetime).eigenvalues();
  if (evals.size() < 2) return 0.0;
  return evals[1] - evals[0];
}

double HarmonicDimension::compute(const std::shared_ptr<Spacetime> &spacetime) {
  if (spacetime == nullptr) return 0.0;
  const ::tessera::cobordism::HodgeLaplacian hodge(spacetime);
  const std::size_t order = hodge.eigenvalues().size();  // N = |V|
  if (order == 0) return 0.0;
  // harmonics() is a flat N*M array whose M columns span ker L_0; M = dim ker.
  return static_cast<double>(hodge.harmonics().size() / order);
}

}  // namespace tessera::observables
