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

#include "spacetime/topologies/RealProjectiveSpace.h"

#include <cstdint>
#include <vector>

namespace tessera::spacetime {

void RealProjectiveSpace::build(Spacetime *spacetime, int /*numSimplices*/) {
  // Walkup's minimal 11-vertex RP^3 (f-vector (11, 51, 80, 40)): the 40
  // tetrahedra tabulated by Lutz and shipped by SageMath's
  // RealProjectiveSpace(3), relabeled from the literature's 1..11 down to
  // 0..10. Every triangle lies in exactly two tetrahedra (closed 3-manifold),
  // and the complex is orientable with H_1 = Z/2 — the Dijkgraaf–Witten sign
  // cocycle's positive control.
  const std::vector<std::vector<std::uint64_t>> tops{
      {0, 1, 2, 6}, {0, 1, 2, 10}, {0, 1, 5, 8}, {0, 1, 5, 10},
      {0, 1, 6, 8}, {0, 2, 4, 9}, {0, 2, 4, 10}, {0, 2, 6, 9},
      {0, 3, 6, 8}, {0, 3, 6, 9}, {0, 3, 7, 8}, {0, 3, 7, 9},
      {0, 4, 5, 7}, {0, 4, 5, 10}, {0, 4, 7, 9}, {0, 5, 7, 8},
      {1, 2, 3, 7}, {1, 2, 3, 10}, {1, 2, 6, 7}, {1, 3, 5, 9},
      {1, 3, 5, 10}, {1, 3, 7, 9}, {1, 4, 6, 7}, {1, 4, 6, 8},
      {1, 4, 7, 9}, {1, 4, 8, 9}, {1, 5, 8, 9}, {2, 3, 4, 8},
      {2, 3, 4, 10}, {2, 3, 7, 8}, {2, 4, 8, 9}, {2, 5, 6, 7},
      {2, 5, 6, 9}, {2, 5, 7, 8}, {2, 5, 8, 9}, {3, 4, 5, 6},
      {3, 4, 5, 10}, {3, 4, 6, 8}, {3, 5, 6, 9}, {4, 5, 6, 7}};
  buildExplicit(spacetime, 11, tops);
}

} // namespace tessera::spacetime
