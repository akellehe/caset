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

#include "cobordism/PreparedBoundaryState.h"

#include <stdexcept>
#include <string>
#include <utility>

#include "cobordism/BoundaryStateSpace.h"

namespace tessera::cobordism {

PreparedBoundaryState::PreparedBoundaryState(
    std::shared_ptr<const BoundaryStateSpace> space, Eigen::VectorXcd amplitudes)
    : space_(std::move(space)), amplitudes_(std::move(amplitudes)) {
  if (space_ == nullptr)
    throw std::invalid_argument(
        "PreparedBoundaryState: the BoundaryStateSpace is null");
  if (static_cast<int>(amplitudes_.size()) != space_->boundaryDimension())
    throw std::invalid_argument(
        "PreparedBoundaryState: the amplitude vector length must be "
        "dim Z(Sigma) = 2^{b_1(Sigma)} = " +
        std::to_string(space_->boundaryDimension()));
}

std::complex<double> PreparedBoundaryState::amplitude(
    std::size_t holonomyClass) const {
  if (holonomyClass >= size())
    throw std::out_of_range(
        "PreparedBoundaryState::amplitude: holonomy class " +
        std::to_string(holonomyClass) + " out of range (dim Z(Sigma) = " +
        std::to_string(size()) + ")");
  return amplitudes_[static_cast<Eigen::Index>(holonomyClass)];
}

std::complex<double> PreparedBoundaryState::generatorAmplitude(
    int harmonic) const {
  if (harmonic < 0 || harmonic >= space_->harmonicDimension())
    throw std::out_of_range(
        "PreparedBoundaryState::generatorAmplitude: harmonic " +
        std::to_string(harmonic) + " out of range (b_1 = " +
        std::to_string(space_->harmonicDimension()) + ")");
  return amplitudes_[space_->generatorIndex(harmonic)];
}

Cochain PreparedBoundaryState::readout() const {
  return space_->reconstruct(amplitudes_);
}

std::complex<double> PreparedBoundaryState::overlap(
    const PreparedBoundaryState &other) const {
  if (amplitudes_.size() != other.amplitudes_.size())
    throw std::invalid_argument(
        "PreparedBoundaryState::overlap: the two states have different boundary "
        "dimensions (" + std::to_string(amplitudes_.size()) + " vs " +
        std::to_string(other.amplitudes_.size()) + ")");
  // Hermitian, conjugate-linear in the first argument: <a, b> = sum conj(a) b
  // (= np.vdot), matching Cochain::innerProduct.
  return amplitudes_.dot(other.amplitudes_);
}

double PreparedBoundaryState::norm() const { return amplitudes_.norm(); }

}  // namespace tessera::cobordism
