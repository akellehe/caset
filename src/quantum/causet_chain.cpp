// Implementation of tessera::quantum::extractCausetChain — see
// include/quantum/causet_chain.hpp for the design.

#include "quantum/causet_chain.hpp"

#include "Poset.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
#include "spacetime/Spacetime.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <unordered_map>

namespace tessera::quantum {

CausetChain extractCausetChain(tessera::Spacetime const& st) {
    CausetChain out;

    auto const& vlist = st.getVertexList();
    if (!vlist) return out;

    // ── Group live vertices by integer time slice. std::map keeps the
    // slice index in ascending order, which is exactly what the chain
    // layout needs — earliest slice → site 0, next slice → next site
    // block, and so on.
    std::map<int, std::vector<std::uint64_t>> by_time;
    for (auto* v : vlist->liveVector()) {
        if (v == nullptr) continue;
        const int t = static_cast<int>(v->getTime());
        by_time[t].push_back(v->getId());
    }
    if (by_time.empty()) return out;

    // ── Flatten into the chain layout: ascending-ID inside each
    // antichain, ascending-time across antichains. Build the inverse
    // map (spacetime ID → flat lattice index) along the way; we'll
    // need it for the hopping-pair extraction.
    out.times.reserve(by_time.size());
    out.antichains.reserve(by_time.size());
    std::unordered_map<std::uint64_t, int> id_to_flat;
    int flat_idx = 0;
    for (auto& [t, ids] : by_time) {
        std::sort(ids.begin(), ids.end());
        out.times.push_back(t);
        out.antichains.push_back(ids);  // copy of sorted ids
        for (auto id : ids) {
            id_to_flat.emplace(id, flat_idx);
            out.vertexIds.push_back(id);
            ++flat_idx;
        }
    }
    out.nSites = flat_idx;

    // ── For each timelike edge whose endpoints land on adjacent
    // antichains, record a hopping pair in the flat-index space.
    // "Adjacent" here means the antichain indices differ by 1 — i.e.
    // the edge crosses exactly one slice boundary in the time order
    // we just built. Skip edges that span more than one slice (they
    // would be reduced out as non-cover relations anyway).
    std::unordered_map<int, int> time_to_aidx;
    time_to_aidx.reserve(out.times.size());
    for (int i = 0; i < static_cast<int>(out.times.size()); ++i) {
        time_to_aidx.emplace(out.times[static_cast<std::size_t>(i)], i);
    }

    auto const& elist = st.getEdgeList();
    if (elist) {
        for (auto const* e : elist->toVector()) {
            if (e == nullptr) continue;
            if (e->getSquaredLength() >= 0.0) continue;  // spacelike/null
            auto const& src = e->getSource();
            auto const& dst = e->getTarget();
            if (!src || !dst) continue;
            const int t_s = static_cast<int>(src->getTime());
            const int t_d = static_cast<int>(dst->getTime());
            if (t_s == t_d) continue;

            auto it_s = time_to_aidx.find(t_s);
            auto it_d = time_to_aidx.find(t_d);
            if (it_s == time_to_aidx.end() || it_d == time_to_aidx.end())
                continue;
            const int aidx_s = it_s->second;
            const int aidx_d = it_d->second;
            if (std::abs(aidx_s - aidx_d) != 1) continue;  // not adjacent

            auto it_fs = id_to_flat.find(src->getId());
            auto it_fd = id_to_flat.find(dst->getId());
            if (it_fs == id_to_flat.end() || it_fd == id_to_flat.end())
                continue;
            int i = it_fs->second;
            int j = it_fd->second;
            if (i == j) continue;
            if (i > j) std::swap(i, j);  // canonicalise i < j
            out.hoppingPairs.emplace_back(i, j);
        }
    }
    // Dedupe — a CDT mesh can produce parallel edges between the same
    // vertex pair via different simplices.
    std::sort(out.hoppingPairs.begin(), out.hoppingPairs.end());
    out.hoppingPairs.erase(
        std::unique(out.hoppingPairs.begin(), out.hoppingPairs.end()),
        out.hoppingPairs.end());

    // ── Inherit the Hasse cover Poset on the flat-lattice label set.
    // Poset::fromSpacetime uses the same ascending-ID remapping we
    // applied above (sort by Vertex::getId()), so its node IDs
    // coincide with our flat-lattice indices when every Spacetime
    // vertex has a unique time slice. When several vertices share a
    // time slice, the per-slice ascending-ID order also matches, so
    // the two index spaces stay aligned.
    out.partialOrder = tessera::Poset::fromSpacetime(st);
    return out;
}

} // namespace tessera::quantum
