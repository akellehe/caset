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

#ifndef CASET_VOLUMEPROFILE_H
#define CASET_VOLUMEPROFILE_H

#include "observables/Observable.h"
#include "spacetime/Spacetime.h"
#include <vector>
#include <map>

namespace caset {

/// Measures the volume profile N3(t): the number of simplices straddling each time slice.
/// This is the primary observable for comparing CDT simulations to de Sitter cosmology.
class VolumeProfile : public Observable {
  public:
    double compute(std::shared_ptr<Spacetime> &spacetime) override;
    double update(std::shared_ptr<Spacetime> &spacetime) override;

    /// @return The volume profile as a vector indexed by time slice offset from tMin.
    [[nodiscard]] std::vector<int> getProfile() const;

    /// @return The accumulated average volume profile over multiple measurements.
    [[nodiscard]] std::vector<double> getAverageProfile() const;

    /// Record a measurement of the current profile for averaging.
    void measure(std::shared_ptr<Spacetime> &spacetime);

    /// Reset accumulated measurements.
    void reset();

  private:
    std::vector<int> currentProfile;
    std::vector<std::vector<int>> measurements;
};

} // caset

#endif //CASET_VOLUMEPROFILE_H
