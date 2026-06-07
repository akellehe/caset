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

#include "cobordism/Spectrum.h"

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace tessera::cobordism {

Spectrum::Spectrum(Eigen::VectorXcd eigenvalues,
                   std::vector<Cochain> eigenvectors, bool hermitian)
    : eigenvalues_(std::move(eigenvalues)),
      eigenvectors_(std::move(eigenvectors)),
      hermitian_(hermitian) {
  if (static_cast<std::size_t>(eigenvalues_.size()) != eigenvectors_.size())
    throw std::invalid_argument(
        "Spectrum: " + std::to_string(eigenvalues_.size()) + " eigenvalues but " +
        std::to_string(eigenvectors_.size()) + " eigenvectors");
}

std::vector<Cochain> Spectrum::harmonics(double tol) const {
  std::vector<Cochain> out;
  for (std::size_t i = 0; i < eigenvectors_.size(); ++i)
    if (std::abs(eigenvalues_[static_cast<Eigen::Index>(i)]) < tol)
      out.push_back(eigenvectors_[i]);
  return out;
}

std::complex<double> Spectrum::eigenvalue(std::size_t i) const {
  if (i >= size())
    throw std::out_of_range("Spectrum::eigenvalue: index " + std::to_string(i) +
                            " out of range (size " + std::to_string(size()) + ")");
  return eigenvalues_[static_cast<Eigen::Index>(i)];
}

const Cochain &Spectrum::operator[](std::size_t i) const {
  if (i >= size())
    throw std::out_of_range("Spectrum::operator[]: index " + std::to_string(i) +
                            " out of range (size " + std::to_string(size()) + ")");
  return eigenvectors_[i];
}

}  // namespace tessera::cobordism
