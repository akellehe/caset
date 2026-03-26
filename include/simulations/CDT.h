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

#ifndef CASET_CDT_H
#define CASET_CDT_H

#include "simulations/Simulation.h"
#include "spacetime/Spacetime.h"
#include <memory>
#include <random>
#include <vector>
#include <map>

namespace caset {

/// CDT implements the Metropolis Monte Carlo algorithm for Causal Dynamical Triangulations
/// in 4 dimensions. The algorithm samples triangulated spacetimes weighted by the
/// Regge action, using local Pachner moves that preserve the causal structure.
///
/// The Regge action for 4D CDT is:
/// \f[
///   S = -(k_0 + 6\Delta) N_0 + (k_4 + \Delta) N_{41} + k_4 N_{32}
///       + \varepsilon (N_4 - \bar{N}_4)^2
/// \f]
class CDT : public Simulation {
  public:
    /// @param spacetime The spacetime to simulate. Must be built before passing.
    /// @param k0 Coupling constant related to inverse Newton constant
    /// @param k4 Coupling constant related to cosmological constant
    /// @param delta Asymmetry parameter between timelike and spacelike edges
    /// @param epsilon Volume fixing strength
    /// @param targetN4 Target total number of 4-simplices
    CDT(std::shared_ptr<Spacetime> spacetime, double k0, double k4, double delta,
        double epsilon, std::size_t targetN4);

    /// (2,8) move: Insert a vertex into a (4,1) or (1,4) simplex pair,
    /// creating 8 new simplices from 2.
    bool add();

    /// (8,2) move: Remove an order-8 vertex, merging 8 simplices into 2.
    bool remove();

    /// (4,6)/(6,4) move: Flip simplices around a shared triangle.
    bool flip();

    /// (2,4) move: Rearrange simplices around a timelike edge.
    bool shift();

    /// (4,2) move: Inverse of shift.
    bool ishift();

    /// Adjust k4 to drive N4 toward targetN4.
    void tune() override;

    /// Run sweeps until the system is thermalized.
    void thermalize() override;

    /// One Monte Carlo sweep: N4 random move attempts with Metropolis acceptance.
    /// @return Number of accepted moves in this sweep.
    int sweep();

    /// Compute the full Regge action for the current configuration.
    [[nodiscard]] double computeAction() const;

    /// Compute the volume profile: number of simplices straddling each time slice.
    [[nodiscard]] std::vector<int> getVolumeProfile() const;

    /// @return Acceptance rates as {moveName: acceptedCount/attemptCount}
    [[nodiscard]] std::map<std::string, double> getAcceptanceRates() const;

    [[nodiscard]] std::shared_ptr<Spacetime> getSpacetime() const noexcept;
    [[nodiscard]] double getK0() const noexcept;
    [[nodiscard]] double getK4() const noexcept;
    [[nodiscard]] double getDelta() const noexcept;

  private:
    std::shared_ptr<Spacetime> spacetime;
    double k0, k4, delta, epsilon;
    std::size_t targetN4;
    std::mt19937 rng{std::random_device{}()};

    // Metropolis acceptance test
    bool accept(double deltaS);

    // Incremental action change
    double computeDeltaAction(int dN0, int dN41, int dN32) const;

    // Statistics
    int addAttempts = 0, addAccepted = 0;
    int removeAttempts = 0, removeAccepted = 0;
    int flipAttempts = 0, flipAccepted = 0;
    int shiftAttempts = 0, shiftAccepted = 0;
    int ishiftAttempts = 0, ishiftAccepted = 0;
};

} // caset

#endif //CASET_CDT_H
