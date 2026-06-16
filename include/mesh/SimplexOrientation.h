// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

//
// Created by andrew on 12/14/25.
//

#ifndef TESSERA_SIMPLEXORIENTATION_H
#define TESSERA_SIMPLEXORIENTATION_H

// Note: an old `#include <pybind11/pybind11.h>` was removed here. It was
// unreferenced inside the file and was dragging Python.h into every TU
// of the core mesh subsystem — including tessera_core, which then forced
// every consumer (test executables, tessera_quantum) to link Python. None
// of mesh / spacetime / observables actually use pybind11 — that's all
// in src/bindings.cpp.

#include <algorithm>
#include <memory>
#include <vector>

#include "mesh/ForwardDeclarations.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::mesh {
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;

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

#endif //TESSERA_SIMPLEXORIENTATION_H