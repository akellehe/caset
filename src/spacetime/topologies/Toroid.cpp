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

/// Build a CDT triangulation using the staircase product triangulation.
///
/// Constructs time slabs between adjacent layers of (d+1) vertices each.
/// The spatial topology at each time slice is the boundary of the d-simplex
/// (S^{d-1}): for d=4 this gives S^3 with 5 vertices and 5 tetrahedra.
///
/// Each slab is triangulated using the staircase decomposition of the
/// product S^{d-1} × [t, t+1]. For each spatial (d-1)-simplex (face),
/// the staircase produces d top-simplices covering all CDT orientation
/// types: (d,1), (d-1,2), ..., (2,d-1), (1,d). This yields d*(d+1)
/// simplices per slab (20 for d=4), including (3,2)/(2,3) types that
/// enable flip and shift moves ([BGL] Sec. 2.3.2–2.3.3).
///
/// Ref: Ambjorn et al. "Reconstructing the Universe" (2005), Sec. 3;
///      Brunekreef et al. "Simulating CDT quantum gravity" (2023), Sec. 2.3.
void Toroid::build(Spacetime *spacetime, int nSimplices) {
  auto d = spacetime->getMetric()->getSignature()->getDimensions();
  int dPlus1 = d + 1;
  int simplicesPerSlab = d * dPlus1;  // staircase: d simplices per face, (d+1) faces
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

  // Create simplices for each time slab using staircase triangulation.
  // For each spatial face F_i (skip vertex i), the face has d vertices.
  // The staircase produces d simplices with orientations (d,1) down to (1,d):
  //   k=d-1: {v0,...,v_{d-1}, w_{d-1}}             — (d,1)
  //   k=d-2: {v0,...,v_{d-2}, w_{d-2}, w_{d-1}}    — (d-1,2)
  //   ...
  //   k=0:   {v0, w0, w1, ..., w_{d-1}}            — (1,d)
  // where v_j are the d lower-layer vertices of F_i and w_j are their
  // upper-layer counterparts.
  for (int slab = 0; slab < numSlabs; ++slab) {
    auto &S = layers[slab];       // spatial vertices at time t
    auto &N = layers[slab + 1];   // next-time vertices at time t+1

    // For each spatial face F_i (skip vertex i from the boundary of Δ^d)
    for (int i = 0; i < dPlus1; ++i) {
      // Collect the d face vertices and their upper counterparts
      std::vector<VertexPtr> faceS, faceN;
      faceS.reserve(d);
      faceN.reserve(d);
      for (int j = 0; j < dPlus1; ++j) {
        if (j != i) {
          faceS.push_back(S[j]);
          faceN.push_back(N[j]);
        }
      }

      // Staircase: for k = d-1 down to 0, create one simplex
      // with (k+1) lower vertices and (d-k) upper vertices
      for (int k = d - 1; k >= 0; --k) {
        VertexPtrs verts;
        verts.reserve(dPlus1);
        for (int j = 0; j <= k; ++j) verts.push_back(faceS[j]);
        for (int j = k; j < d; ++j) verts.push_back(faceN[j]);
        spacetime->createSimplex(verts);
      }
    }
  }

  // Force facet computation on all top simplices so that coface
  // relationships are established. This is needed for the add move
  // to find spatial-face partners via getCofaces().
  // Iterate by index: getFacets() may register new sub-simplices,
  // growing simplicesVec.  We only process the original top-simplices.
  auto nBefore = spacetime->getSimplices().size();
  for (std::size_t i = 0; i < nBefore; ++i) {
    auto s = spacetime->getSimplices()[i];
    if (s->size() == static_cast<std::size_t>(dPlus1)) {
      s->getFacets();
    }
  }
}

} // caset
