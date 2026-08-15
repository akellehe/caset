// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "observables/Spectral.h"

#include <cstddef>
#include <vector>

#include "cobordism/HodgeLaplacian.h"
#include "spacetime/Spacetime.h"

namespace tessera::observables {

double SpectralGap::compute(const std::shared_ptr<Spacetime> &spacetime) {
  if (spacetime == nullptr) return 0.0;
  // Eigenvalues are ascending; the first gap is lambda_1 - lambda_0.
  // Degree 0 is the graph Laplacian D - A, Hermitian, so the spectrum is real;
  // eigenvalues() is complex-typed for parity with the k >= 1 d'Alembertian.
  const std::vector<std::complex<double>> evals =
      ::tessera::cobordism::HodgeLaplacian(spacetime).eigenvalues();
  if (evals.size() < 2) return 0.0;
  return (evals[1] - evals[0]).real();
}

double HarmonicDimension::compute(const std::shared_ptr<Spacetime> &spacetime) {
  if (spacetime == nullptr) return 0.0;
  // harmonics() is one Cochain per basis vector of ker L_0, so dim ker = its count.
  return static_cast<double>(
      ::tessera::cobordism::HodgeLaplacian(spacetime).harmonics().size());
}

}  // namespace tessera::observables
