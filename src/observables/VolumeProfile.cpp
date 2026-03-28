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

#include "observables/VolumeProfile.h"
#include <algorithm>

namespace caset {

double VolumeProfile::compute(std::shared_ptr<Spacetime> &spacetime) {
  int topSize = spacetime->getMetric()->getSignature()->getDimensions() + 1;
  std::map<int, int> profile;
  for (const auto &s : spacetime->getSimplices()) {
    if (static_cast<int>(s->size()) != topSize) continue;
    int tMin = static_cast<int>(s->getTi());
    profile[tMin]++;
  }
  if (profile.empty()) {
    currentProfile.clear();
    return 0.0;
  }
  int tMinKey = profile.begin()->first;
  int tMaxKey = profile.rbegin()->first;
  currentProfile.assign(tMaxKey - tMinKey + 1, 0);
  for (const auto &[t, count] : profile) {
    currentProfile[t - tMinKey] = count;
  }
  // Return the peak volume
  return static_cast<double>(*std::max_element(currentProfile.begin(), currentProfile.end()));
}

double VolumeProfile::update(std::shared_ptr<Spacetime> &spacetime) {
  return compute(spacetime);
}

std::vector<int> VolumeProfile::getProfile() const {
  return currentProfile;
}

void VolumeProfile::measure(std::shared_ptr<Spacetime> &spacetime) {
  compute(spacetime);
  measurements.push_back(currentProfile);
}

std::vector<double> VolumeProfile::getAverageProfile() const {
  if (measurements.empty()) return {};
  // Find max length
  std::size_t maxLen = 0;
  for (const auto &m : measurements) maxLen = std::max(maxLen, m.size());
  std::vector<double> avg(maxLen, 0.0);
  for (const auto &m : measurements) {
    for (std::size_t i = 0; i < m.size(); ++i) {
      avg[i] += m[i];
    }
  }
  for (auto &v : avg) v /= static_cast<double>(measurements.size());
  return avg;
}

void VolumeProfile::reset() {
  currentProfile.clear();
  measurements.clear();
}

} // caset
