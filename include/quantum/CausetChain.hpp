// Spacetime → causet-chain adapter for the quantum subsystem.
// See docs/source/quantum-plan.md §6.
//
// The vanilla Schwinger MPO (include/quantum/SchwingerModel.hpp) lives
// on a regular 1D lattice with N sites and nearest-neighbour hopping
// pairs (n, n+1). This adapter generalises that lattice: replace it with a
// "chain of antichains" sourced from a tessera::Spacetime, where each
// antichain is the set of vertices at a fixed integer time slice and
// hopping follows the timelike causet edges that connect adjacent
// slices.
//
// This header provides the minimal data extraction needed to drive
// that generalisation. It does NOT itself rebuild an MPO on the
// chain — for the simplest case where every antichain has exactly one
// vertex the chain-of-antichains coincides with the existing 1D
// lattice, and `SchwingerHamiltonian::mpoChain(...)` in
// schwinger_model.hpp can run directly with `params.N = chain.nSites`
// and `chain.hoppingPairs` as the hopping graph.
//
// ─── What this provides ───────────────────────────────────────────────
//
// • CausetChain — flattened (lattice site → spacetime vertex ID)
//                 mapping plus the hopping pairs and the inherited
//                 Hasse-cover Poset.
// • Causet — coarse-grained adapter façade (static methods only).

#pragma once

#include "Poset.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace tessera {
class Spacetime;
}

namespace tessera::quantum {

// Spacetime → 1D lattice adapter (data class).
//
// `antichains[s]` is the sorted list of Spacetime vertex IDs at
// `times[s]`, where `times` is ascending-sorted. The flat lattice
// site index of a (slice, position-in-antichain) pair is the
// concatenation:
//
//   flat_idx = (Σ_{r<s} |antichains[r]|) + position
//
// `vertexIds[flat_idx]` is the inverse map: lattice site → spacetime
// vertex ID. `nSites = sum(|antichains[s]|) = vertexIds.size()`.
//
// `hoppingPairs` lists the (i, j) flat-lattice-site pairs coupled
// by adjacent-time-slice timelike edges: this is what would replace
// the "Σ_n (X_n X_{n+1} + Y_n Y_{n+1})" sum in H_hop on the causet.
// Pairs are stored once with i < j; the MPO builder applies σ⁺σ⁻ +
// σ⁻σ⁺ symmetrically per pair. Edges that span non-adjacent slices
// are skipped — they're transitively reduced out by Poset::fromSpacetime
// and would not contribute a physical hopping term anyway.
//
// `partialOrder` is the Hasse-cover Poset on flat-lattice-site IDs,
// inherited from Spacetime via Poset::fromSpacetime. It's one of the
// three orders compared in the causal-comparison machinery
// (the "≼_cs" entry).
struct CausetChain {
    int nSites{0};
    std::vector<int> times;                                 // ascending
    std::vector<std::vector<std::uint64_t>> antichains;     // [tIdx][pos]
    std::vector<std::uint64_t> vertexIds;                  // [flat_idx]
    std::vector<std::pair<int, int>> hoppingPairs;         // [k] = (i, j) i<j
    tessera::Poset partialOrder;
};

// Coarse-grained façade for tessera::Spacetime → causet adapters.
// Stateless; not instantiable.
class Causet {
public:
    Causet() = delete;
    Causet(Causet const&) = delete;
    Causet& operator=(Causet const&) = delete;

    // Walk the Spacetime's vertex list, group by integer time slice
    // (Vertex::getTime() truncated to int), and extract:
    //
    //   • the antichain layering (antichains, times),
    //   • the flat lattice ↔ spacetime ID mapping (vertexIds),
    //   • the adjacent-slice timelike-edge hopping pairs,
    //   • the Hasse cover Poset on flat lattice IDs.
    //
    // All four outputs share the same flat-index labelling, so a
    // caller can interchangeably feed them to Majorization::agreement,
    // an MPO builder, or a visualisation backend.
    //
    // Antichain ordering inside a slice is by ascending Spacetime
    // vertex ID — stable and deterministic. Edges with squaredLength
    // ≥ 0 (spacelike or null) are ignored, as are any timelike edges
    // with src.time == tgt.time.
    [[nodiscard]] static CausetChain
    chainFrom(tessera::Spacetime const& st);
};

} // namespace tessera::quantum
