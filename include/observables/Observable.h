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

#ifndef CASET_OBSERVABLE_H
#define CASET_OBSERVABLE_H

#include <memory>

namespace caset {

class Spacetime;

/// # Observable Base Class
///
/// Interface for physical observables measured on a triangulated spacetime
/// configuration. In lattice quantum gravity, observables are functions
/// \f$ \mathcal{O}[\mathcal{T}] \f$ of the triangulation \f$ \mathcal{T} \f$
/// whose expectation values are estimated by averaging over Monte Carlo samples:
///
/// \f[
///   \langle \mathcal{O} \rangle
///     = \frac{1}{Z} \sum_{\mathcal{T}} \mathcal{O}[\mathcal{T}]\, e^{-S[\mathcal{T}]}
///     \approx \frac{1}{N_{\text{meas}}} \sum_{i=1}^{N_{\text{meas}}} \mathcal{O}[\mathcal{T}_i]
/// \f]
///
/// Subclasses include:
///   - **SpacetimeVolume**: total number of \f$ d \f$-simplices \f$ N_d \f$,
///   - **VolumeProfile**: spatial volume per time slice \f$ N_{d-1}(t) \f$.
///
class Observable {
  public:
    /// Compute the observable from scratch for the given spacetime.
    ///
    /// @param spacetime The spacetime configuration to measure
    /// @return The scalar value of the observable
    virtual double compute(std::shared_ptr<Spacetime> &spacetime);

    /// Incrementally update the observable after a local move.
    /// Default implementation delegates to compute().
    ///
    /// @param spacetime The spacetime after the most recent move
    /// @return The updated scalar value
    virtual double update(std::shared_ptr<Spacetime> &spacetime);

    virtual ~Observable() = default;
};

} // caset

#endif //CASET_OBSERVABLE_H
