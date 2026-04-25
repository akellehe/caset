// Phase 6 (docs/source/quantum-plan.md §6) — Spacetime → causet-chain
// adapter for the quantum subsystem.
//
// The vanilla Schwinger MPO (include/quantum/schwinger_model.hpp) lives
// on a regular 1D lattice with N sites and nearest-neighbour hopping
// pairs (n, n+1). Phase 6 generalises that lattice: replace it with a
// "chain of antichains" sourced from a caset::Spacetime, where each
// antichain is the set of vertices at a fixed integer time slice and
// hopping follows the timelike causet edges that connect adjacent
// slices.
//
// This header provides the minimal data extraction needed to drive
// that generalisation. It does NOT itself rebuild an MPO on the
// chain — for the simplest case where every antichain has exactly one
// vertex the chain-of-antichains coincides with the existing 1D
// lattice and `build_schwinger_mpo` in schwinger_model.hpp can run
// directly with `params.N = chain.n_sites` and a remapping of hopping
// pairs. For non-trivial antichains (causet branches), the MPS chain
// layout still works as long as we order sites so that every hopping
// pair (i, j) has |i - j| reasonably small; otherwise ITensor's tree
// tensor network support is the long-term path. That MPO construction
// is left for a future commit; this file delivers the data so callers
// can wire either path.
//
// ─── What this provides ───────────────────────────────────────────────
//
// • CausetChain     — flattened (lattice site → spacetime vertex ID)
//                     mapping plus the hopping pairs and the inherited
//                     Hasse-cover Poset.
// • extract_causet_chain(Spacetime) — the extractor.

#pragma once

#include "Poset.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace caset {
class Spacetime;
}

namespace caset::quantum {

// Spacetime → 1D lattice adapter.
//
// `antichains[s]` is the sorted list of Spacetime vertex IDs at
// `times[s]`, where `times` is ascending-sorted. The flat lattice
// site index of a (slice, position-in-antichain) pair is the
// concatenation:
//
//   flat_idx = (Σ_{r<s} |antichains[r]|) + position
//
// `vertex_ids[flat_idx]` is the inverse map: lattice site → spacetime
// vertex ID. `n_sites = sum(|antichains[s]|) = vertex_ids.size()`.
//
// `hopping_pairs` lists the (i, j) flat-lattice-site pairs coupled
// by adjacent-time-slice timelike edges: this is what would replace
// the "Σ_n (X_n X_{n+1} + Y_n Y_{n+1})" sum in H_hop on the causet.
// Pairs are stored once with i < j (the adjacency is symmetric); the
// MPO builder applies σ⁺σ⁻ + σ⁻σ⁺ symmetrically per pair.
//
// Edges that span non-adjacent slices (skipping a layer) are skipped
// here — they're transitively reduced out by Poset::from_spacetime
// and would not contribute a physical hopping term anyway.
//
// `partial_order` is the Hasse-cover Poset on flat-lattice-site IDs,
// inherited from Spacetime via Poset::from_spacetime. It's one of the
// three orders compare_orders() measures (the "≺_caset" entry from
// Phase 5).
struct CausetChain {
    int n_sites{0};
    std::vector<int> times;                                 // ascending
    std::vector<std::vector<std::uint64_t>> antichains;     // [t_idx][pos]
    std::vector<std::uint64_t> vertex_ids;                  // [flat_idx]
    std::vector<std::pair<int, int>> hopping_pairs;         // [k] = (i, j) i<j
    caset::Poset partial_order;
};

// Walk the Spacetime's vertex list, group by integer time slice
// (Vertex::getTime() truncated to int), and extract:
//
//   • the antichain layering (antichains, times),
//   • the flat lattice ↔ spacetime ID mapping (vertex_ids),
//   • the adjacent-slice timelike-edge hopping pairs,
//   • the Hasse cover Poset on flat lattice IDs.
//
// All four outputs share the same flat-index labelling, so a caller
// can interchangeably feed them to compare_orders, an MPO builder, or
// a visualisation backend.
//
// Antichain ordering inside a slice is by ascending Spacetime vertex
// ID — stable and deterministic.
//
// Edges with squaredLength >= 0 (spacelike or null) are ignored, as
// are any timelike edges with src.time == tgt.time (a metric
// inconsistency — defensively skipped, never expected from a valid
// Spacetime).
CausetChain extract_causet_chain(caset::Spacetime const& st);

} // namespace caset::quantum
