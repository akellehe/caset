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

#include "spacetime/topologies/LensSpace.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace tessera::spacetime {

LensSpace::LensSpace(int p, int q) : p_(p), q_(q) {
  // Validate eagerly: an unsupported (p,q) is an error at construction, not a
  // deferred surprise at build() time.
  std::size_t numVertices = 0;
  (void)triangulation(p, q, numVertices);
}

void LensSpace::build(Spacetime *spacetime, int /*numSimplices*/) {
  std::size_t numVertices = 0;
  const std::vector<std::vector<std::uint64_t>> tops =
      triangulation(p_, q_, numVertices);
  buildExplicit(spacetime, numVertices, tops);
}

std::vector<std::vector<std::uint64_t>> LensSpace::triangulation(
    int p, int q, std::size_t &numVertices) {
  if (p == 3 && q == 1) {
    // Lutz vertex-minimal L(3,1): f-vector (12, ...), 54 tetrahedra,
    // H_1 = Z_3. Vertices 0..11; every triangle lies in exactly two
    // tetrahedra (closed 3-manifold).
    numVertices = 12;
    return {
        {0, 1, 2, 3}, {0, 1, 2, 4}, {0, 1, 3, 5}, {0, 1, 4, 5},
        {0, 2, 3, 6}, {0, 2, 4, 6}, {0, 3, 5, 7}, {0, 3, 6, 8},
        {0, 3, 7, 8}, {0, 4, 5, 9}, {0, 4, 6, 10}, {0, 4, 9, 10},
        {0, 5, 7, 9}, {0, 6, 8, 10}, {0, 7, 8, 11}, {0, 7, 9, 11},
        {0, 8, 10, 11}, {0, 9, 10, 11}, {1, 2, 3, 9}, {1, 2, 4, 8},
        {1, 2, 7, 9}, {1, 2, 7, 10}, {1, 2, 8, 10}, {1, 3, 5, 11},
        {1, 3, 9, 11}, {1, 4, 5, 8}, {1, 5, 6, 8}, {1, 5, 6, 11},
        {1, 6, 7, 10}, {1, 6, 7, 11}, {1, 6, 8, 10}, {1, 7, 9, 11},
        {2, 3, 6, 9}, {2, 4, 6, 11}, {2, 4, 8, 11}, {2, 5, 6, 9},
        {2, 5, 6, 11}, {2, 5, 7, 9}, {2, 5, 7, 10}, {2, 5, 10, 11},
        {2, 8, 10, 11}, {3, 4, 7, 8}, {3, 4, 7, 10}, {3, 4, 8, 9},
        {3, 4, 9, 10}, {3, 5, 7, 10}, {3, 5, 10, 11}, {3, 6, 8, 9},
        {3, 9, 10, 11}, {4, 5, 8, 9}, {4, 6, 7, 10}, {4, 6, 7, 11},
        {4, 7, 8, 11}, {5, 6, 8, 9}};
  }
  if (p == 4 && q == 1) {
    // Lutz vertex-minimal L(4,1): f-vector (14, ...), 70 tetrahedra,
    // H_1 = Z_4. Vertices 0..13; every triangle lies in exactly two
    // tetrahedra (closed 3-manifold).
    numVertices = 14;
    return {
        {0, 1, 2, 3}, {0, 1, 2, 4}, {0, 1, 3, 5}, {0, 1, 4, 5},
        {0, 2, 3, 6}, {0, 2, 4, 6}, {0, 3, 5, 7}, {0, 3, 6, 8},
        {0, 3, 7, 8}, {0, 4, 5, 9}, {0, 4, 6, 10}, {0, 4, 9, 10},
        {0, 5, 7, 9}, {0, 6, 8, 10}, {0, 7, 8, 11}, {0, 7, 9, 11},
        {0, 8, 10, 11}, {0, 9, 10, 11}, {1, 2, 3, 9}, {1, 2, 4, 8},
        {1, 2, 8, 9}, {1, 3, 5, 10}, {1, 3, 9, 10}, {1, 4, 5, 12},
        {1, 4, 8, 13}, {1, 4, 12, 13}, {1, 5, 10, 11}, {1, 5, 11, 12},
        {1, 8, 9, 13}, {1, 9, 10, 11}, {1, 9, 11, 13}, {1, 11, 12, 13},
        {2, 3, 6, 13}, {2, 3, 9, 12}, {2, 3, 12, 13}, {2, 4, 6, 11},
        {2, 4, 8, 11}, {2, 5, 6, 11}, {2, 5, 6, 13}, {2, 5, 7, 10},
        {2, 5, 7, 13}, {2, 5, 10, 11}, {2, 7, 10, 12}, {2, 7, 12, 13},
        {2, 8, 9, 12}, {2, 8, 10, 11}, {2, 8, 10, 12}, {3, 4, 7, 8},
        {3, 4, 7, 10}, {3, 4, 8, 13}, {3, 4, 9, 10}, {3, 4, 9, 12},
        {3, 4, 12, 13}, {3, 5, 7, 10}, {3, 6, 8, 13}, {4, 5, 9, 12},
        {4, 6, 7, 10}, {4, 6, 7, 11}, {4, 7, 8, 11}, {5, 6, 9, 12},
        {5, 6, 9, 13}, {5, 6, 11, 12}, {5, 7, 9, 13}, {6, 7, 10, 12},
        {6, 7, 11, 12}, {6, 8, 9, 12}, {6, 8, 9, 13}, {6, 8, 10, 12},
        {7, 9, 11, 13}, {7, 11, 12, 13}};
  }
  if (p == 5 && q == 2) {
    // Lutz vertex-minimal L(5,2): f-vector (14, ...), 72 tetrahedra,
    // H_1 = Z_5. Vertices 0..13; every triangle lies in exactly two
    // tetrahedra (closed 3-manifold).
    numVertices = 14;
    return {
        {0, 1, 2, 3}, {0, 1, 2, 4}, {0, 1, 3, 5}, {0, 1, 4, 5},
        {0, 2, 3, 6}, {0, 2, 4, 6}, {0, 3, 5, 7}, {0, 3, 6, 8},
        {0, 3, 7, 8}, {0, 4, 5, 9}, {0, 4, 6, 10}, {0, 4, 9, 11},
        {0, 4, 10, 11}, {0, 5, 7, 9}, {0, 6, 8, 10}, {0, 7, 8, 11},
        {0, 7, 9, 11}, {0, 8, 10, 11}, {1, 2, 3, 12}, {1, 2, 4, 7},
        {1, 2, 7, 9}, {1, 2, 9, 10}, {1, 2, 10, 12}, {1, 3, 5, 13},
        {1, 3, 12, 13}, {1, 4, 5, 8}, {1, 4, 7, 8}, {1, 5, 6, 11},
        {1, 5, 6, 13}, {1, 5, 8, 11}, {1, 6, 9, 10}, {1, 6, 9, 11},
        {1, 6, 10, 12}, {1, 6, 12, 13}, {1, 7, 8, 11}, {1, 7, 9, 11},
        {2, 3, 6, 11}, {2, 3, 11, 12}, {2, 4, 6, 7}, {2, 5, 6, 11},
        {2, 5, 6, 13}, {2, 5, 10, 12}, {2, 5, 10, 13}, {2, 5, 11, 12},
        {2, 6, 7, 13}, {2, 7, 9, 13}, {2, 9, 10, 13}, {3, 4, 7, 8},
        {3, 4, 7, 10}, {3, 4, 8, 9}, {3, 4, 9, 11}, {3, 4, 10, 11},
        {3, 5, 7, 10}, {3, 5, 10, 13}, {3, 6, 8, 9}, {3, 6, 9, 11},
        {3, 10, 11, 13}, {3, 11, 12, 13}, {4, 5, 8, 9}, {4, 6, 7, 10},
        {5, 7, 9, 12}, {5, 7, 10, 12}, {5, 8, 9, 12}, {5, 8, 11, 12},
        {6, 7, 10, 12}, {6, 7, 12, 13}, {6, 8, 9, 10}, {7, 9, 12, 13},
        {8, 9, 10, 13}, {8, 9, 12, 13}, {8, 10, 11, 13}, {8, 11, 12, 13}};
  }
  throw std::invalid_argument(
      "LensSpace: unsupported (p,q)=(" + std::to_string(p) + "," +
      std::to_string(q) +
      "); available: (3,1), (4,1), (5,2). Note L(2,1)=RP^3 is "
      "RealProjectiveSpace.");
}

} // namespace tessera::spacetime
