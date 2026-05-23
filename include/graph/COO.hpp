// MIT License -- Copyright (c) 2025 Andrew Kelleher
//
// Shared COO (coordinate-list) sparse edge representations.
//
// Three places in the codebase emit (rows, cols, [weights], n) edge
// arrays — ``Spacetime::getDualAdjacency`` (unweighted dual graph),
// ``MutualInformationProfile::weightedAdjacency`` (MI graph), and
// ``EmergentGraph::fromWeightedEdges`` (test factory). Each previously
// invented its own return shape; these two structs give the C++ side a
// single canonical type. The CSR builder in ``graph/CSRBuilder.hpp``
// consumes the same field convention.
//
// Note: each undirected edge is expected to appear *twice* in the
// arrays (rows[k]=u, cols[k]=v AND rows[k']=v, cols[k']=u) so the CSR
// builder can lay out the per-row neighbour lists in one pass.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// === tessera subsystem ns fwd-decls ===
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::graph {
using namespace ::tessera::mesh;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;

// Unweighted COO. ``rows.size() == cols.size() == 2 * |E|`` for
// undirected graphs (each edge listed both directions).
template <typename Idx = std::uint32_t>
struct COO {
    std::vector<Idx> rows;
    std::vector<Idx> cols;
    Idx              n{0};

    [[nodiscard]] std::size_t size()  const noexcept { return rows.size(); }
    [[nodiscard]] bool        empty() const noexcept { return rows.empty(); }
};

// Weighted COO. Same convention; ``weights[k]`` is the weight of the
// directed entry (rows[k] → cols[k]). For an undirected weighted graph
// both directions carry the same weight.
template <typename Idx = int, typename W = double>
struct WeightedCOO {
    std::vector<Idx> rows;
    std::vector<Idx> cols;
    std::vector<W>   weights;
    Idx              n{0};

    [[nodiscard]] std::size_t size()  const noexcept { return rows.size(); }
    [[nodiscard]] bool        empty() const noexcept { return rows.empty(); }
};

} // namespace tessera::graph
