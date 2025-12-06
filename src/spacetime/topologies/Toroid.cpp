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

#include "spacetime/topologies/Toroid.h"
#include "spacetime/Spacetime.h"
#include <iostream>

namespace caset {
void Toroid::build(Spacetime *spacetime, int numSimplices) {
  if (numSimplices % 2 != 0) {
    throw std::invalid_argument("numSimplices must be an even number");
  }

  std::vector<std::tuple<uint8_t, uint8_t> > orientations = {{1, 2}, {2, 1}};
  spacetime->createSimplex(orientations[1]);
  for (int i = 0; i < numSimplices; i++) {
    SimplexPtr rightSimplex = spacetime->createSimplex(orientations[i % 2]);
    OptionalSimplexPair leftFaceRightFace = spacetime->chooseSimplexFacesToGlue(rightSimplex);
    if (!leftFaceRightFace.has_value()) return;
    auto [leftFace, rightFace] = leftFaceRightFace.value();
    auto [left, succeeded] = spacetime->causallyAttachFaces(leftFace, rightFace);
  }
}
}
