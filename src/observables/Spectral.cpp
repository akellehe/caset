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
  // The U(1) CONNECTION Laplacian D - A, not the Hodge L_0 (#805). This
  // observable's content is Aharonov-Bohm -- the gap collapses at flux pi --
  // which is a statement about the connection operator; L_0's gap is a
  // different number and carries no flux dependence at all. The connection
  // operator is Hermitian, so eigenvalues are real and ascending and the first
  // gap is lambda_1 - lambda_0; connectionEigenvalues() is complex-typed for
  // parity with the L_k family.
  const std::vector<std::complex<double>> evals =
      ::tessera::cobordism::HodgeLaplacian(spacetime).connectionEigenvalues();
  if (evals.size() < 2) return 0.0;
  return (evals[1] - evals[0]).real();
}

double HarmonicDimension::compute(const std::shared_ptr<Spacetime> &spacetime) {
  if (spacetime == nullptr) return 0.0;
  // dim ker of the U(1) CONNECTION Laplacian, not of L_0 (#805). A nonzero flux
  // lifts this zero mode, which is the observable's whole content; dim ker L_0
  // is always b_0, already reported by ChainComplex::bettiNumbers.
  return static_cast<double>(
      ::tessera::cobordism::HodgeLaplacian(spacetime)
          .connectionHarmonics().size());
}

}  // namespace tessera::observables
