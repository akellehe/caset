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

#ifndef TESSERA_COBORDISM_SPECTRUM_H
#define TESSERA_COBORDISM_SPECTRUM_H

#include <Eigen/Core>

#include <complex>
#include <cstddef>
#include <vector>

#include "cobordism/Cochain.h"

namespace tessera::cobordism {

/// # Spectrum
///
/// The eigendecomposition of a Hodge Laplacian \f$ L_k \f$ as a value object:
/// the eigenvalues paired with their eigenvectors-as-`Cochain`s, in matching
/// order. Replaces the old `(flat evals, flat M\times M evecs)` pair so the
/// eigenvector indexing carries its degree and \f$ k \f$-simplex ordering.
///
/// ## Representation (covers both regimes)
///
/// Eigenvalues are stored as a single complex vector (`Eigen::VectorXcd`),
/// the cleanest representation that covers BOTH cases uniformly:
///
///  - **Hermitian / metric** (`isHermitian() == true`): the self-adjoint
///    \f$ k = 0 \f$ graph Laplacian and the symmetric metric Hodge Laplacian
///    (\f$ k \geq 1 \f$). The eigenvalues are mathematically real (their stored
///    imaginary parts are zero) and **ascending**.
///  - **Lorentzian** (`isHermitian() == false`): the signed-weight, generally
///    non-self-adjoint d'Alembertian. The eigenvalues may be negative or come in
///    complex-conjugate pairs, sorted ascending by \f$ (\mathrm{Re},\mathrm{Im}) \f$.
///
/// `eigenvalues()[i]` is the eigenvalue of `eigenvectors()[i]`. `harmonics(tol)`
/// is the kernel subset \f$ \{v : |\lambda_v| < \text{tol}\} = \ker L_k \f$ as
/// `Cochain`s (a basis for \f$ H_k \f$ in the Hermitian case; the small-\f$ |\lambda| \f$
/// near-kernel pseudo-Hodge subset in the Lorentzian case).
class Spectrum {
  public:
    /// The empty spectrum (no modes).
    Spectrum() = default;

    /// `eigenvalues[i]` pairs with `eigenvectors[i]`; `hermitian` records whether
    /// the eigenvalues are guaranteed real & ascending (the metric/self-adjoint
    /// regime) vs. the indefinite Lorentzian regime. @throws std::invalid_argument
    /// if the eigenvalue and eigenvector counts differ.
    Spectrum(Eigen::VectorXcd eigenvalues, std::vector<Cochain> eigenvectors,
             bool hermitian);

    /// The eigenvalues; pybind exposes them as a 1-D complex `numpy.ndarray`.
    /// Ascending real values (imag 0) in the Hermitian regime; sorted by
    /// \f$ (\mathrm{Re},\mathrm{Im}) \f$ in the Lorentzian regime.
    [[nodiscard]] const Eigen::VectorXcd &eigenvalues() const noexcept {
      return eigenvalues_;
    }

    /// The eigenvectors, one `Cochain` per eigenvalue, in matching order.
    [[nodiscard]] const std::vector<Cochain> &eigenvectors() const noexcept {
      return eigenvectors_;
    }

    /// The harmonic subset: the eigenvectors whose eigenvalue has
    /// \f$ |\lambda| < \text{tol} \f$ (a basis for \f$ \ker L_k \f$), as `Cochain`s.
    [[nodiscard]] std::vector<Cochain> harmonics(double tol = 1e-9) const;

    /// The number of modes (eigenvalues = eigenvectors).
    [[nodiscard]] std::size_t size() const noexcept { return eigenvectors_.size(); }

    /// Whether the eigenvalues are guaranteed real & ascending (the
    /// metric/self-adjoint regime) rather than the indefinite Lorentzian one.
    [[nodiscard]] bool isHermitian() const noexcept { return hermitian_; }

    /// The \f$ i \f$-th eigenvalue. @throws std::out_of_range if `i >= size()`.
    [[nodiscard]] std::complex<double> eigenvalue(std::size_t i) const;

    /// The \f$ i \f$-th eigenvector. @throws std::out_of_range if `i >= size()`.
    [[nodiscard]] const Cochain &operator[](std::size_t i) const;

  private:
    Eigen::VectorXcd eigenvalues_{};
    std::vector<Cochain> eigenvectors_{};
    bool hermitian_{true};
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_SPECTRUM_H
