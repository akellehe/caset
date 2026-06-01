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

#include "spacetime/topologies/ComplexProjectivePlane.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <set>
#include <vector>

namespace tessera::spacetime {

void ComplexProjectivePlane::build(Spacetime *spacetime, int /*numSimplices*/) {
  // Kühnel's 9-vertex CP^2. The twelve base 4-simplices below (vertices
  // 0..8) are the literature list of Kühnel & Banchoff with vertex labels
  // shifted down by one. The full set of 36 is closed under the order-three
  // symmetry S = (0 3 6)(1 4 7)(2 5 8) — applying S twice to each base
  // simplex yields its three-member orbit, and 12 * 3 = 36.
  const std::array<std::array<std::uint64_t, 5>, 12> base{{
      {0, 4, 1, 7, 8}, {0, 1, 2, 7, 8}, {0, 2, 5, 7, 8}, {3, 4, 1, 7, 8},
      {3, 1, 2, 7, 8}, {3, 2, 5, 7, 8}, {0, 3, 1, 4, 5}, {0, 3, 2, 4, 5},
      {0, 3, 1, 4, 8}, {0, 3, 2, 5, 7}, {0, 3, 6, 1, 5}, {0, 3, 6, 5, 7}}};

  // S as a lookup: vertex -> image. The three 3-cycles (0 3 6), (1 4 7),
  // (2 5 8).
  const std::array<std::uint64_t, 9> permute{3, 4, 5, 6, 7, 8, 0, 1, 2};

  // Generate the orbit of every base simplex and collect the distinct ones
  // (a std::set keyed on the sorted vertex tuple deduplicates and orders them).
  std::set<std::vector<std::uint64_t>> simplices;
  for (const auto &simplex : base) {
    std::vector<std::uint64_t> current(simplex.begin(), simplex.end());
    for (int step = 0; step < 3; ++step) {
      std::vector<std::uint64_t> sorted = current;
      std::sort(sorted.begin(), sorted.end());
      simplices.insert(std::move(sorted));
      for (auto &vertex : current) vertex = permute[vertex];
    }
  }

  std::vector<std::vector<std::uint64_t>> tops(simplices.begin(), simplices.end());
  buildExplicit(spacetime, 9, tops);
}

} // namespace tessera::spacetime
