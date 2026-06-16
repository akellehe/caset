// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

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
