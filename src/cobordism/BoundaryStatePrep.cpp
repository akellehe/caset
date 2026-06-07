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

#include "cobordism/BoundaryStatePrep.h"

#include <stdexcept>
#include <string>
#include <utility>

#include "cobordism/ChainComplex.h"
#include "cobordism/HodgeLaplacian.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

BoundaryStatePrep::BoundaryStatePrep(std::shared_ptr<Spacetime> sigma,
                                     double tol, bool metric)
    : sigma_(std::move(sigma)) {
  if (sigma_ == nullptr)
    throw std::runtime_error("BoundaryStatePrep: the surface Sigma is null");

  // |C_1(Sigma)| fixes the harmonic-form length and lets us recover b_1 from the
  // flat harmonics array (size = |C_1| * b_1).
  numEdges_ = static_cast<int>(ChainComplex::fromSpacetime(*sigma_).numSimplices(1));

  // ker L_1(Sigma) via the k=1 Hodge Laplacian — reused, not reimplemented. The
  // columns are W_k-orthonormal (the symmetric SelfAdjointEigenSolver basis), so
  // the standard inner product on them is the identity and `prepare` is an
  // isometry for either weight choice.
  harmonics_ = HodgeLaplacian(sigma_).harmonics(1, tol, metric);
  if (numEdges_ > 0)
    b1_ = static_cast<int>(harmonics_.size()) / numEdges_;
  else
    b1_ = 0;  // no edges ⇒ no 1-forms (e.g. a point/empty complex)

  // Z(Sigma) = C[H^1(Sigma; Z_2)] has dimension 2^{b_1}; cap at the same nullity
  // the gf2Span materialization refuses, so boundaryDimension() cannot overflow
  // or demand an unmaterializable vector.
  if (b1_ > 24)
    throw std::runtime_error(
        "BoundaryStatePrep: b_1(Sigma) = " + std::to_string(b1_) +
        " is too large; Z(Sigma) = 2^{b_1} would exceed the materialization cap "
        "(24)");
}

int BoundaryStatePrep::harmonicDimension() const { return b1_; }

int BoundaryStatePrep::boundaryDimension() const { return 1 << b1_; }

int BoundaryStatePrep::numEdges() const { return numEdges_; }

std::vector<std::complex<double>> BoundaryStatePrep::harmonics() const {
  return harmonics_;
}

std::vector<int> BoundaryStatePrep::generatorIndices() const {
  // The b_1 single-generator flat-connection classes: gf2Span index 2^i carries
  // the i-th harmonic 1-form's amplitude.
  std::vector<int> indices;
  indices.reserve(static_cast<std::size_t>(b1_));
  for (int i = 0; i < b1_; ++i) indices.push_back(1 << i);
  return indices;
}

std::vector<std::complex<double>> BoundaryStatePrep::prepare(
    const std::vector<std::complex<double>> &form) const {
  if (static_cast<int>(form.size()) != numEdges_)
    throw std::invalid_argument(
        "BoundaryStatePrep::prepare: the 1-form length must be |C_1(Sigma)| = " +
        std::to_string(numEdges_));

  // Scatter c_i = <h_i, form> onto the flat-connection class at index 2^i; all
  // other amplitudes (the trivial class and every multi-generator class) are 0.
  std::vector<std::complex<double>> state(
      static_cast<std::size_t>(1 << b1_), {0.0, 0.0});
  for (int i = 0; i < b1_; ++i) {
    std::complex<double> coordinate{0.0, 0.0};
    for (int e = 0; e < numEdges_; ++e)
      coordinate +=
          std::conj(harmonics_[static_cast<std::size_t>(e) * b1_ + i]) *
          form[static_cast<std::size_t>(e)];
    state[static_cast<std::size_t>(1 << i)] = coordinate;
  }
  return state;
}

std::vector<std::complex<double>> BoundaryStatePrep::readout(
    const std::vector<std::complex<double>> &state) const {
  if (static_cast<int>(state.size()) != (1 << b1_))
    throw std::invalid_argument(
        "BoundaryStatePrep::readout: the boundary state length must be "
        "2^{b_1(Sigma)} = " + std::to_string(1 << b1_));

  // Gather the generator amplitudes c_i = state[2^i] and rebuild sum_i c_i h_i,
  // landing back in ker L_1(Sigma).
  std::vector<std::complex<double>> form(
      static_cast<std::size_t>(numEdges_), {0.0, 0.0});
  for (int i = 0; i < b1_; ++i) {
    const std::complex<double> coordinate = state[static_cast<std::size_t>(1 << i)];
    for (int e = 0; e < numEdges_; ++e)
      form[static_cast<std::size_t>(e)] +=
          coordinate * harmonics_[static_cast<std::size_t>(e) * b1_ + i];
  }
  return form;
}

}  // namespace tessera::cobordism
