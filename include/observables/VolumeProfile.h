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

#ifndef TESSERA_VOLUMEPROFILE_H
#define TESSERA_VOLUMEPROFILE_H

#include "observables/Observable.h"
#include "spacetime/Spacetime.h"
#include <vector>
#include <map>

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::observables {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;

/// # Volume Profile Observable
///
/// Measures the spatial volume profile \f$ N_3(t) \f$, defined as the number
/// of top-dimensional simplices whose initial vertex lies at time slice \f$ t \f$.
/// This is the primary observable for comparing CDT simulations to continuum
/// cosmology.
///
/// ## Physical Significance
///
/// In the de Sitter phase \f$(C_{dS})\f$ of 4D CDT, the ensemble-averaged
/// volume profile matches the metric of the Euclidean four-sphere \f$ S^4 \f$:
///
/// \f[
///   \langle N_3(t) \rangle \;\propto\; \cos^4\!\left(\frac{\pi\, t}{T}\right)
/// \f]
///
/// where \f$ T \f$ is the total time extent. This is the discrete analogue of the
/// scale factor in Friedmann-Lemaitre-Robertson-Walker (FLRW) cosmology:
///
/// \f[
///   ds^2 = -dt^2 + a(t)^2\, d\Omega_3^2
///   \qquad\text{with}\quad
///   a(t) = \cos\!\left(\frac{\pi\, t}{T}\right)
/// \f]
///
/// In the crumpled phase (B), the volume concentrates on a single time slice.
/// In the branched-polymer phase (A), the profile becomes thin and elongated.
///
/// ## References
///
///   - Ambjorn, Jurkiewicz, Loll, *Reconstructing the Universe*, Phys. Rev. D 72 (2005)
///   - Gorlich, *Introduction to Causal Dynamical Triangulations* (2013), Section 3.3
///
class VolumeProfile : public Observable {
  public:
    /// Compute the current volume profile and return the peak volume.
    ///
    /// @param spacetime The spacetime to measure
    /// @return The maximum value of \f$ N_3(t) \f$ across all time slices
    double compute(const std::shared_ptr<Spacetime> &spacetime) override;

    /// Recompute the profile (equivalent to compute for this observable).
    double update(const std::shared_ptr<Spacetime> &spacetime) override;

    /// @return The most recently computed volume profile as a vector indexed by time slice.
    [[nodiscard]] const std::vector<int> &getProfile() const;

    /// @return The average volume profile over all recorded measurements.
    [[nodiscard]] std::vector<double> getAverageProfile() const;

    /// Peak-centered average of the recorded measurements.
    ///
    /// Equivalent to ::centeredAverage applied to the accumulated
    /// measurements (see ::measure).  See that overload for the algorithm.
    [[nodiscard]] std::vector<double> getCenteredAverageProfile(
        bool subtractStalk = false, bool normalizePeak = false) const;

    /// Peak-centered average of a set of volume profiles.
    ///
    /// On a torus the de Sitter blob can sit at any time slice and its
    /// position diffuses along the Markov chain, so naive bin-by-bin
    /// averaging smears it into uniform noise.  Each profile is therefore
    /// zero-padded to the longest length and circularly rolled so its peak
    /// aligns at @c T/2 before the bin-wise mean is taken (Ambjorn,
    /// Jurkiewicz, Loll, *Reconstructing the Universe*, 2005).
    ///
    /// @param profiles     The per-configuration volume profiles to average
    /// @param subtractStalk Subtract each (padded) profile's minimum before
    ///                       centering, removing the constant stalk volume
    /// @param normalizePeak Rescale the result so its peak equals 1
    /// @return The peak-centered average profile (empty if @p profiles is)
    [[nodiscard]] static std::vector<double> centeredAverage(
        const std::vector<std::vector<double>> &profiles,
        bool subtractStalk = false, bool normalizePeak = false);

    /// Record the current volume profile for later averaging.
    /// Call this after each decorrelated Monte Carlo measurement to build statistics.
    ///
    /// @param spacetime The spacetime to measure
    void measure(const std::shared_ptr<Spacetime> &spacetime);

    /// Reset all accumulated measurements.
    void reset();

  private:
    std::vector<int> currentProfile;          ///< Most recent \f$ N_3(t) \f$
    std::vector<std::vector<int>> measurements; ///< History of profiles for averaging
};

} // namespace tessera::observables

#endif //TESSERA_VOLUMEPROFILE_H
