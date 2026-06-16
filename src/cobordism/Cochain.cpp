// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/Cochain.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace tessera::cobordism {

Cochain::Cochain(int degree, std::vector<std::vector<std::uint64_t>> simplices,
                 Eigen::VectorXcd coeffs)
    : degree_(degree), simplices_(std::move(simplices)), coeffs_(std::move(coeffs)) {
  if (simplices_.size() != static_cast<std::size_t>(coeffs_.size()))
    throw std::invalid_argument(
        "Cochain: ordering has " + std::to_string(simplices_.size()) +
        " simplices but coeffs has " + std::to_string(coeffs_.size()) +
        " amplitudes");
}

std::complex<double> Cochain::amplitude(std::size_t index) const {
  if (index >= size())
    throw std::out_of_range("Cochain::amplitude: index " +
                            std::to_string(index) + " out of range (size " +
                            std::to_string(size()) + ")");
  return coeffs_[static_cast<Eigen::Index>(index)];
}

std::complex<double> Cochain::amplitudeFor(
    const std::vector<std::uint64_t> &simplex) const {
  for (std::size_t i = 0; i < simplices_.size(); ++i)
    if (simplices_[i] == simplex)
      return coeffs_[static_cast<Eigen::Index>(i)];
  throw std::out_of_range(
      "Cochain::amplitudeFor: simplex is not in this cochain's ordering");
}

std::complex<double> Cochain::innerProduct(const Cochain &other) const {
  if (degree_ != other.degree_)
    throw std::invalid_argument(
        "Cochain::innerProduct: degree mismatch (" + std::to_string(degree_) +
        " vs " + std::to_string(other.degree_) + ")");
  if (simplices_ != other.simplices_)
    throw std::invalid_argument(
        "Cochain::innerProduct: the two cochains are indexed over different "
        "k-simplex orderings");
  // Hermitian, conjugate-linear in the first argument: <a, b> = sum conj(a_i) b_i.
  return coeffs_.dot(other.coeffs_);
}

double Cochain::norm() const { return coeffs_.norm(); }

Cochain Cochain::normalized() const {
  const double n = norm();
  if (n <= 0.0) return *this;
  return Cochain(degree_, simplices_, Eigen::VectorXcd(coeffs_ / n));
}

}  // namespace tessera::cobordism
