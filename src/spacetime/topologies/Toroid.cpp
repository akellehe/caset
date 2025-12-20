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

#include "spacetime/Spacetime.h"
#include "spacetime/topologies/Toroid.h"
#include <iostream>
#include <vector>
#include <memory>


namespace caset {


void Toroid::build(Spacetime *spacetime, int numSimplices) {
  int dimensions = spacetime->getMetric()->getSignature()->getDimensions();
  std::vector<std::tuple<std::uint8_t, std::uint8_t> > orientations{};
  if (dimensions == 3) {
    orientations = {{1, 2}, {2, 1}};
  } else if (dimensions == 4) {
    orientations = {{1, 4}, {2, 3}};
  }
  spacetime->createSimplex(orientations[1]);
  for (int i = 0; i < numSimplices; i++) {
    const auto [rightSimplex, created] = spacetime->createSimplex(orientations[i % 2]);
    OptionalSimplexPtrPair leftFaceRightFace = spacetime->chooseSimplexFacesToGlue(rightSimplex);
    if (!leftFaceRightFace.has_value()) return;
    auto [leftFace, rightFace] = leftFaceRightFace.value();
    [[maybe_unused]] auto [left, succeeded] = spacetime->causallyAttachFaces(leftFace, rightFace);
  }
}
}
