// MIT License -- Copyright (c) 2025 Andrew Kelleher
#pragma once

#include <vector>
#include <utility>

namespace tessera {

/// Spring-electrical force-directed layout in 3D.
///
/// Places \a n nodes in 3D space using Fruchterman-Reingold-style
/// spring attraction along edges and Coulomb repulsion between all
/// node pairs.  Returns a flat row-major vector of size \a n * 3
/// containing the (x, y, z) positions.
///
/// @param n           Number of nodes
/// @param edges       Index pairs (i, j) defining graph edges
/// @param centerIdx   If >= 0, pin this node at the origin
/// @param initPos     Initial positions (flat, n*3).  Random if empty.
/// @param restLengths Per-edge rest lengths (unit if empty)
/// @param springK     Spring constant for edge attraction
/// @param repulsionK  Coulomb constant for node repulsion
/// @param iters       Number of iterations
/// @param cooling     Multiplicative step-size decay per iteration
/// @param repulsionCap Max nodes for all-pairs repulsion (limits O(n^2))
/// @param seed        Random seed for reproducible layouts
/// @return Flat row-major vector of size n*3 with (x, y, z) positions
[[nodiscard]] std::vector<double> forceLayout3D(
    int n,
    const std::vector<std::pair<int, int>> &edges,
    int centerIdx = -1,
    const std::vector<double> &initPos = {},
    const std::vector<double> &restLengths = {},
    double springK = 0.01,
    double repulsionK = 0.5,
    int iters = 300,
    double cooling = 0.995,
    int repulsionCap = 200,
    int seed = 42);

} // namespace tessera
