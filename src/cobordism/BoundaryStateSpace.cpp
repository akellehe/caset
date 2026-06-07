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

#include "cobordism/BoundaryStateSpace.h"

#include <stdexcept>
#include <string>
#include <utility>

#include "cobordism/ChainComplex.h"
#include "cobordism/HodgeLaplacian.h"
#include "cobordism/PreparedBoundaryState.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

BoundaryStateSpace::BoundaryStateSpace(std::shared_ptr<Spacetime> sigma,
                                       double tol, bool metric)
    : sigma_(std::move(sigma)) {
  if (sigma_ == nullptr)
    throw std::runtime_error("BoundaryStateSpace: the surface Sigma is null");

  // |C_1(Sigma)| and the canonical degree-1 simplex ordering fix the index space
  // a harmonic 1-form Cochain lives over (kept so reconstruct() can build one
  // even when b_1 = 0, where the cached basis is empty).
  const ChainComplex chain = ChainComplex::fromSpacetime(*sigma_);
  numEdges_ = static_cast<int>(chain.numSimplices(1));
  edges_ = chain.kSimplexVertices(1);

  // ker L_1(Sigma) via the k=1 Hodge Laplacian — reused, not reimplemented. The
  // basis Cochains are W_k-orthonormal (the symmetric SelfAdjointEigenSolver
  // eigenvectors), so the standard inner product on them is the identity and
  // `prepare` is an isometry for either weight choice.
  harmonics_ = HodgeLaplacian(sigma_).harmonics(1, tol, metric);
  b1_ = static_cast<int>(harmonics_.size());  // one Cochain per ker L_1 vector

  // Z(Sigma) = C[H^1(Sigma; Z_2)] has dimension 2^{b_1}; cap at the same nullity
  // the gf2Span materialization refuses, so boundaryDimension() cannot overflow
  // or demand an unmaterializable vector.
  if (b1_ > 24)
    throw std::runtime_error(
        "BoundaryStateSpace: b_1(Sigma) = " + std::to_string(b1_) +
        " is too large; Z(Sigma) = 2^{b_1} would exceed the materialization cap "
        "(24)");

  // Dense |C_1| x b_1 view of the basis (column i = harmonics_[i].coeffs()) for
  // the prepare/readout projection math.
  harmonicMatrix_.resize(numEdges_, b1_);
  for (int i = 0; i < b1_; ++i)
    harmonicMatrix_.col(i) = harmonics_[static_cast<std::size_t>(i)].coeffs();
}

int BoundaryStateSpace::harmonicDimension() const { return b1_; }

int BoundaryStateSpace::boundaryDimension() const { return 1 << b1_; }

int BoundaryStateSpace::numEdges() const { return numEdges_; }

const std::vector<Cochain> &BoundaryStateSpace::harmonics() const noexcept {
  return harmonics_;
}

int BoundaryStateSpace::generatorIndex(int harmonic) const noexcept {
  // The b_1 vs 2^{b_1} reconciliation, in one place: harmonic i is carried by the
  // single-generator flat-connection class at gf2Span index 2^i.
  return 1 << harmonic;
}

std::vector<int> BoundaryStateSpace::generatorIndices() const {
  std::vector<int> indices;
  indices.reserve(static_cast<std::size_t>(b1_));
  for (int i = 0; i < b1_; ++i) indices.push_back(generatorIndex(i));
  return indices;
}

PreparedBoundaryState BoundaryStateSpace::prepare(const Cochain &form) const {
  if (form.degree() != 1)
    throw std::invalid_argument(
        "BoundaryStateSpace::prepare: the harmonic form must be a degree-1 "
        "Cochain (a 1-form), got degree " + std::to_string(form.degree()));
  if (static_cast<int>(form.size()) != numEdges_)
    throw std::invalid_argument(
        "BoundaryStateSpace::prepare: the 1-form length must be |C_1(Sigma)| = " +
        std::to_string(numEdges_));

  // Harmonic-basis coordinates c_i = <h_i, form> = conj(h_i) . form, then scatter
  // c_i onto the flat-connection class at index 2^i; the trivial class and every
  // multi-generator class stay 0. (H^dagger projects out any non-harmonic part.)
  const Eigen::VectorXcd coordinates = harmonicMatrix_.adjoint() * form.coeffs();
  Eigen::VectorXcd amplitudes =
      Eigen::VectorXcd::Zero(boundaryDimension());
  for (int i = 0; i < b1_; ++i)
    amplitudes[generatorIndex(i)] = coordinates[i];
  return PreparedBoundaryState(shared_from_this(), std::move(amplitudes));
}

PreparedBoundaryState BoundaryStateSpace::state(
    const Eigen::VectorXcd &amplitudes) const {
  if (static_cast<int>(amplitudes.size()) != boundaryDimension())
    throw std::invalid_argument(
        "BoundaryStateSpace::state: the amplitude vector length must be "
        "2^{b_1(Sigma)} = " + std::to_string(boundaryDimension()));
  return PreparedBoundaryState(shared_from_this(), amplitudes);
}

Cochain BoundaryStateSpace::reconstruct(const Eigen::VectorXcd &amplitudes) const {
  if (static_cast<int>(amplitudes.size()) != boundaryDimension())
    throw std::invalid_argument(
        "BoundaryStateSpace::reconstruct: the boundary state length must be "
        "2^{b_1(Sigma)} = " + std::to_string(boundaryDimension()));

  // Gather the generator amplitudes c_i = amplitudes[2^i] and rebuild the 1-form
  // sum_i c_i h_i = H c, landing back in ker L_1(Sigma). For b_1 = 0 the (|C_1| x
  // 0) * (0) product is the zero 1-form, the only state of the 1-dim Z(Sigma).
  Eigen::VectorXcd coordinates(b1_);
  for (int i = 0; i < b1_; ++i)
    coordinates[i] = amplitudes[generatorIndex(i)];
  Eigen::VectorXcd form = harmonicMatrix_ * coordinates;
  return Cochain(1, edges_, std::move(form));
}

}  // namespace tessera::cobordism
