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

#include "SimplexOrientation.h"
#include "spacetime/topologies/Toroid.h"
#include "spacetime/Spacetime.h"
#include "utils.h"
#include <deque>
#include <cmath>
#include <vector>

namespace caset {

/// Build a CDT triangulation with proper bi-directional spatial face structure.
///
/// Constructs time slabs between adjacent layers of (d+1) vertices each.
/// Each slab contains 2*(d+1) simplices: (d+1) of type (d,1) and (d+1) of
/// type (1,d). The spatial (d-1)-faces at interior time slices are shared
/// by simplices from adjacent slabs, enabling the (2,2d) vertex insertion
/// move (Brunekreef Sec. 2.3.1).
///
/// The spatial topology at each time slice is the boundary of the d-simplex
/// (S^{d-1}), which is the minimal triangulation of the spatial sphere.
/// For d=4 this gives S^3 with 5 vertices and 5 tetrahedra per slice,
/// matching the initial configuration described in "Reconstructing the
/// Universe" (Ambjorn et al. 2005, Sec. 3).
void Toroid::build(Spacetime *spacetime, int nSimplices) {
  auto d = spacetime->getMetric()->getSignature()->getDimensions();
  int dPlus1 = d + 1;
  int simplicesPerSlab = 2 * dPlus1;
  int numSlabs = std::max(2, nSimplices / simplicesPerSlab);
  int numLayers = numSlabs + 1;

  spacetime->reserve(nSimplices);

  // Create vertices: (d+1) per time layer
  std::vector<std::vector<VertexPtr>> layers(numLayers);
  for (int t = 0; t < numLayers; ++t) {
    for (int i = 0; i < dPlus1; ++i) {
      layers[t].push_back(
        spacetime->createVertex(std::vector<double>{static_cast<double>(t)}));
    }
    if (t > 0) spacetime->incrementTime();
  }

  // Create simplices for each time slab.
  // Each slab from layer t to t+1 gets:
  //   (d,1) type: for each i, take all layer-t vertices except [i], plus layer-(t+1) vertex [i]
  //   (1,d) type: for each i, take layer-t vertex [i], plus all layer-(t+1) vertices except [i]
  for (int slab = 0; slab < numSlabs; ++slab) {
    auto &S = layers[slab];       // spatial vertices at time t
    auto &N = layers[slab + 1];   // next-time vertices at time t+1

    for (int i = 0; i < dPlus1; ++i) {
      // (d,1) simplex: {S[0],...,S[d]} \ {S[i]} ∪ {N[i]}
      VertexPtrs verts_d1;
      verts_d1.reserve(dPlus1);
      for (int j = 0; j < dPlus1; ++j) {
        if (j != i) verts_d1.push_back(S[j]);
      }
      verts_d1.push_back(N[i]);
      auto [s1, c1] = spacetime->createSimplex(verts_d1);

      // (1,d) simplex: {S[i]} ∪ {N[0],...,N[d]} \ {N[i]}
      VertexPtrs verts_1d;
      verts_1d.reserve(dPlus1);
      verts_1d.push_back(S[i]);
      for (int j = 0; j < dPlus1; ++j) {
        if (j != i) verts_1d.push_back(N[j]);
      }
      auto [s2, c2] = spacetime->createSimplex(verts_1d);
    }
  }

  // Force facet computation on all top simplices so that coface
  // relationships are established. This is needed for the add move
  // to find spatial-face partners via getCofaces().
  for (const auto &s : spacetime->getSimplices()) {
    if (s->size() == static_cast<std::size_t>(dPlus1)) {
      s->getFacets();
    }
  }
}

} // caset
