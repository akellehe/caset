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

#ifndef TESSERA_COBORDISM_PREPAREDBOUNDARYSTATE_H
#define TESSERA_COBORDISM_PREPAREDBOUNDARYSTATE_H

#include <Eigen/Core>

#include <complex>
#include <cstddef>
#include <memory>

#include "cobordism/Cochain.h"

namespace tessera::cobordism {

class BoundaryStateSpace;

/// # PreparedBoundaryState
///
/// A value object: a state in the DW boundary Hilbert space
/// \f$ Z(\Sigma) = \mathbb{C}[H^1(\Sigma;\mathbb{Z}_2)] \f$ of a closed surface
/// \f$ \Sigma \f$. It wraps the \f$ 2^{b_1(\Sigma)} \f$-long complex amplitude
/// vector (`Eigen::VectorXcd`, the flat-connection-class basis) together with a
/// handle to the `BoundaryStateSpace` that defines \f$ \Sigma \f$ — the holonomy
/// indexing and the harmonic basis it reads against. Produced by
/// `BoundaryStateSpace::prepare()` (from a harmonic 1-form) or
/// `BoundaryStateSpace::state()` (from a raw amplitude vector); the held
/// `shared_ptr` keeps its space alive.
///
/// The `harmonic i \to amplitude index 2^i` convention lives in the
/// `BoundaryStateSpace`, not here and not in callers: `generatorAmplitude(i)` and
/// `readout()` delegate to it, so a caller never spells out a power of two.
///
/// `readout()` is the adjoint of `prepare`: it recovers the harmonic 1-form
/// \f$ \sum_i c_i h_i \in \ker L_1(\Sigma) \f$ from the generator amplitudes
/// \f$ c_i = \texttt{coeffs}[2^i] \f$. Because the harmonic basis is orthonormal,
/// `readout()` on `space.prepare(form)` returns `form` (for `form`
/// \f$ \in \ker L_1 \f$), and `overlap` between two prepared states reproduces
/// the harmonic inner product.
class PreparedBoundaryState {
  public:
    /// Wrap `amplitudes` (the \f$ Z(\Sigma) \f$ flat-connection-class vector,
    /// length \f$ 2^{b_1} \f$) as a state over `space`. Prefer the factories
    /// `BoundaryStateSpace::prepare()` / `state()`.
    /// @throws std::invalid_argument if `space` is null or `amplitudes.size()`
    ///   does not equal `space->boundaryDimension()`.
    PreparedBoundaryState(std::shared_ptr<const BoundaryStateSpace> space,
                          Eigen::VectorXcd amplitudes);

    /// The \f$ Z(\Sigma) \f$ amplitude vector (the flat-connection-class basis),
    /// Eigen-backed; pybind exposes it as a 1-D complex `numpy.ndarray`.
    [[nodiscard]] const Eigen::VectorXcd &coeffs() const noexcept {
      return amplitudes_;
    }

    /// \f$ \dim Z(\Sigma) = 2^{b_1(\Sigma)} \f$, the number of holonomy classes.
    [[nodiscard]] std::size_t size() const noexcept {
      return static_cast<std::size_t>(amplitudes_.size());
    }

    /// The `BoundaryStateSpace` \f$ Z(\Sigma) \f$ this state belongs to.
    [[nodiscard]] std::shared_ptr<const BoundaryStateSpace> space()
        const noexcept {
      return space_;
    }

    /// The amplitude for a single holonomy class (a flat-connection class index
    /// \f$ 0\dots2^{b_1}-1 \f$). @throws std::out_of_range if out of range.
    [[nodiscard]] std::complex<double> amplitude(
        std::size_t holonomyClass) const;

    /// The amplitude carried by the \f$ i \f$-th harmonic 1-form — the
    /// single-generator class at \f$ Z(\Sigma) \f$ index \f$ 2^i \f$ (the
    /// convention living in the `BoundaryStateSpace`).
    /// @throws std::out_of_range if `harmonic` is not in \f$ [0, b_1) \f$.
    [[nodiscard]] std::complex<double> generatorAmplitude(int harmonic) const;

    /// Read the harmonic 1-form back out (the adjoint of `prepare`): the degree-1
    /// `Cochain` \f$ \sum_i c_i h_i \in \ker L_1(\Sigma) \f$ with
    /// \f$ c_i = \texttt{coeffs}[2^i] \f$. `space.prepare(form).readout() == form`
    /// for `form` \f$ \in \ker L_1(\Sigma) \f$.
    [[nodiscard]] Cochain readout() const;

    /// The Hermitian inner product \f$ \langle \text{this}, \text{other}\rangle =
    /// \sum_a \overline{\text{this}_a}\,\text{other}_a \f$ (= `np.vdot`) of the
    /// two \f$ Z(\Sigma) \f$ amplitude vectors. For states prepared from harmonic
    /// 1-forms this reproduces the harmonic inner product (prepare is an
    /// isometry). @throws std::invalid_argument if the boundary dimensions differ.
    [[nodiscard]] std::complex<double> overlap(
        const PreparedBoundaryState &other) const;

    /// The Euclidean norm \f$ \sqrt{\sum_a |c_a|^2} \f$ of the amplitude vector.
    [[nodiscard]] double norm() const;

  private:
    std::shared_ptr<const BoundaryStateSpace> space_;
    Eigen::VectorXcd amplitudes_{};
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_PREPAREDBOUNDARYSTATE_H
