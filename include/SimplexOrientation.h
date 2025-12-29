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

#ifndef CASET_SIMPLEXORIENTATION_H
#define CASET_SIMPLEXORIENTATION_H

#include <pybind11/pybind11.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "ForwardDeclarations.h"

namespace caset {

///
///
/// @param timeOrientation
enum class TimeOrientation : uint8_t {
  FUTURE = 0,
  PRESENT = 1,
  UNKNOWN = 2
};

class SimplexOrientation {
  public:
    ///
    /// The orientation of a simplex is determined by how many vertices lie on the initial and final time slice for the
    /// simplex. The orientation is largely only relevant for Lorentzian/CDT complexes where causality is preserved. Those
    /// complexes restrict to allowed orientations that ensure progression forward in time and "fit together" (so they share
    /// faces without gaps in the complex).
    ///
    /// The convention was established in Ambjorn-Loll's "Causal Dynamical Triangulations" paper from 1998-2001. Every
    /// d-simplex must have its vertices split across two adjacent time slices, t and t+1. That means every simplex has
    /// a split
    ///
    /// \f$ (n, d + 1 - n) \f$
    ///
    /// @param ti_ The number of vertices on the initial time slice.
    /// @param tf_ The number of vertices on the final time slice.
    ///
    SimplexOrientation(uint8_t ti_, uint8_t tf_);
    SimplexOrientation();

    [[nodiscard]] SimplexOrientation decTf() const;
    [[nodiscard]] SimplexOrientation decTi() const;
    [[nodiscard]] SimplexOrientation flip() const;
    [[nodiscard]] TimeOrientation getOrientation() const;
    [[nodiscard]] std::pair<uint8_t, uint8_t> numeric() const;
    [[nodiscard]] std::string toString() const noexcept;
    [[nodiscard]] std::vector<SimplexOrientation> getFacialOrientations() const;
    [[nodiscard]] uint8_t getK() const; /// A k-simplex has \f$ k+1 \f$ vertices.
    [[nodiscard]] size_t hash() const;
    bool operator==(const SimplexOrientation &other) const noexcept;
    static SimplexOrientation orientationOf(const VertexPtrs &vertices);
    Fingerprint fingerprint;
  private:
    uint8_t ti{0};
    uint8_t tf{0};
    uint8_t k{0};
};

}

#endif //CASET_SIMPLEXORIENTATION_H