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

#include "spacetime/topologies/SimplexBoundarySphere.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace tessera::spacetime {

void SimplexBoundarySphere::build(Spacetime *spacetime, int /*numSimplices*/) {
  if (n_ < 1) throw std::invalid_argument("SimplexBoundarySphere requires n >= 1");
  // S^n = ∂Δ^{n+1}: n+2 vertices; each top n-simplex omits exactly one vertex.
  const std::size_t numV = static_cast<std::size_t>(n_) + 2;
  std::vector<std::vector<std::uint64_t>> tops;
  tops.reserve(numV);
  for (std::size_t omit = 0; omit < numV; ++omit) {
    std::vector<std::uint64_t> s;
    s.reserve(numV - 1);
    for (std::size_t v = 0; v < numV; ++v)
      if (v != omit) s.push_back(static_cast<std::uint64_t>(v));
    tops.push_back(std::move(s));
  }
  buildExplicit(spacetime, numV, tops);
}

} // namespace tessera::spacetime
