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

//
// Created by andrew on 12/14/25.
//


// (was: #include <pybind11/pybind11.h> — removed; unreferenced.)

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include "mesh/ForwardDeclarations.h"
#include "mesh/SimplexOrientation.h"
#include "mesh/Vertex.h"


namespace caset {
SimplexOrientation::SimplexOrientation(uint8_t ti_, uint8_t tf_)
    : ti(ti_), tf(tf_), k(ti_ + tf_ - 1), fingerprint({ti_, tf_}) {
}

SimplexOrientation::SimplexOrientation()
    : ti(0), tf(0), k(0), fingerprint({0, 0}) {
}

[[nodiscard]] std::pair<uint8_t, uint8_t> SimplexOrientation::numeric() const {
  return {ti, tf};
}

[[nodiscard]] size_t SimplexOrientation::hash() const {
  return fingerprint.fingerprint();
}


[[nodiscard]] SimplexOrientation SimplexOrientation::flip() const {
  SimplexOrientation o{tf, ti};
  return o;
}

[[nodiscard]]
SimplexOrientation SimplexOrientation::decTi() const {
  if (ti == 0) return {0, tf};
  SimplexOrientation o{static_cast<uint8_t>(ti - 1), tf};
  return o;
}

[[nodiscard]]
SimplexOrientation SimplexOrientation::decTf() const {
  if (tf == 0) return {ti, 0};
  SimplexOrientation o{ti, static_cast<uint8_t>(tf - 1)};
  return o;
}

[[nodiscard]] std::string SimplexOrientation::toString() const noexcept {
  return "<SimplexOrientation: (" + std::to_string(ti) + ", " + std::to_string(tf) + ")>";
}

bool SimplexOrientation::operator==(const SimplexOrientation &other) const noexcept {
  return ti == other.ti && tf == other.tf;
}

[[nodiscard]] TimeOrientation SimplexOrientation::getOrientation() const {
  if (ti == tf) return TimeOrientation::UNKNOWN;
  if (ti > tf) return TimeOrientation::PRESENT;
  return TimeOrientation::FUTURE;
}

[[nodiscard]] std::vector<SimplexOrientation> SimplexOrientation::getFacialOrientations() const {
  if (ti + tf == 0) return {};
  if (ti == 0) return {decTf()};
  if (tf == 0) return {decTi()};
  std::vector<SimplexOrientation> orientations;
  orientations.reserve(2);
  orientations.push_back(decTi());
  orientations.push_back(decTf());
  return orientations;
}

/// A k-simplex has \f$ k+1 \f$ vertices.
[[nodiscard]] uint8_t SimplexOrientation::getK() const {
  return k;
}

SimplexOrientation SimplexOrientation::orientationOf(const VertexPtrs &vertices) {
  // Two-pass: first find min/max times, then count.
  // Single-pass was buggy when the first vertex was at tf, not ti.
  double tMin = std::numeric_limits<double>::max();
  double tMax = std::numeric_limits<double>::lowest();
  for (const auto &v : vertices) {
    double t = v->getTime();
    tMin = std::min(tMin, t);
    tMax = std::max(tMax, t);
  }
  uint8_t tiVertices = 0;
  uint8_t tfVertices = 0;
  for (const auto &v : vertices) {
    if (v->getTime() == tMin)
      tiVertices++;
    else
      tfVertices++;
  }
  return {tiVertices, tfVertices};
}
}
