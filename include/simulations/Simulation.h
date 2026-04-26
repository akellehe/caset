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

#ifndef TESSERA_SIMULATION_H
#define TESSERA_SIMULATION_H

#include <functional>

namespace tessera {

/// # Simulation Base Class
///
/// Abstract interface for Monte Carlo simulations of discrete spacetime geometries.
/// Each subclass implements a specific approach to quantum gravity:
///
///   - **CDT** (Causal Dynamical Triangulations): Metropolis sampling over causal
///     triangulations weighted by the Regge action. Preserves a global time foliation.
///   - **Regge calculus**: Varies edge lengths on a fixed triangulation, computing
///     the Einstein-Hilbert action via deficit angles at hinges.
///   - **Causal Set Theory (CST)**: Poisson sprinklings of points in a Lorentzian
///     manifold, with order relations encoding causal structure.
///
/// The simulation lifecycle has two phases:
///
///   1. **Tuning**: Adjust bare coupling constants (e.g., the cosmological constant)
///      so that macroscopic observables (e.g., total spacetime volume) reach their
///      target values. In CDT, this drives \f$ N_4 \to \bar{N}_4 \f$.
///
///   2. **Thermalization**: Evolve the system under the Monte Carlo dynamics until
///      it reaches thermal equilibrium, after which configurations are drawn from
///      the stationary distribution \f$ P(\mathcal{T}) \propto e^{-S[\mathcal{T}]} \f$.
///
class Simulation {
  public:
    virtual ~Simulation() = default;

    /// Tune bare coupling constants toward physically meaningful values.
    ///
    /// In CDT this adjusts \f$ k_4 \f$ so the four-volume fluctuates around the target.
    /// In Regge calculus this constructs an initial triangulation. In CST this sets
    /// the sprinkling density.
    virtual void tune(std::function<void(int,int)> progress = nullptr);

    /// Evolve the system to thermal equilibrium.
    ///
    /// In CDT this runs Monte Carlo sweeps until the action stabilizes. In CST this
    /// applies a Poisson sprinkling to break Lorentz-invariance-violating lattice
    /// artifacts. In Regge calculus this applies random edge-length perturbations.
    virtual void thermalize();
};

} // tessera

#endif //TESSERA_SIMULATION_H
