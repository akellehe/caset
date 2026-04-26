// Phase 6 acceptance — CDT-specific structural invariants on the
// Spacetime → Poset adapter.
//
// A foliated CDT (Foliation::PREFERRED) builds its complex out of d-simplices
// each of which has its vertices split between two adjacent time slices t
// and t+1 (Ambjorn-Loll convention; see Ambjorn 2004, Sec. 2). That gives a
// concrete invariant chain:
//
//   1. Every timelike edge in the Spacetime connects vertices in adjacent
//      time slices — never skips a slice.
//   2. Therefore every Hasse cover edge in `Poset::fromSpacetime` also
//      spans exactly one slice (no transitive reduction can collapse a
//      direct adjacent-slice cover). i.e. on a foliated CDT, covers and
//      hoppingPairs of the chain-of-antichains adapter are the same set.
//   3. The longest chain in the Hasse diagram has length
//      (numTimeSlices - 1): pick any vertex at layer 0, follow covers
//      through one vertex per layer up to the top.
//   4. Every vertex in the spacetime appears as a node in the Poset
//      (extraction is total: no orphan vertices dropped).
//
// These are paper-compliance checks: the CDT formalism we inherit from
// guarantees the foliated structure, and Phase 6 is meaningless if it
// silently collapses or duplicates that structure.

#include "Poset.h"
#include "quantum/causet_chain.hpp"
#include "spacetime/Foliation.h"
#include "spacetime/Metric.h"
#include "spacetime/Signature.h"
#include "spacetime/Spacetime.h"
#include "spacetime/topologies/Toroid.h"

#include <algorithm>
#include <iostream>
#include <set>
#include <vector>

namespace {

// Build a small foliated 4D-Lorentzian Toroid CDT for the invariants
// to apply to. Default Toroid produces dPlus1=5 vertices per layer.
tessera::Spacetime build_toroid_cdt(int n_simplices = 40) {
    tessera::Signature sig(4, tessera::SignatureType::Lorentzian);
    auto metric = std::make_shared<tessera::Metric>(true, sig);
    auto topology = std::static_pointer_cast<tessera::Topology>(
        std::make_shared<tessera::Toroid>());
    tessera::Spacetime st(metric,
                        tessera::SpacetimeType::CDT,
                        std::optional<double>{1.0},
                        std::optional<double>{1.0},
                        tessera::Foliation::PREFERRED,
                        std::optional<std::shared_ptr<tessera::Topology>>{topology});
    st.build(n_simplices);
    return st;
}

bool acceptance_covers_span_adjacent_layers(
    tessera::quantum::CausetChain const& chain) {
    std::cout << "Acceptance #1 — every cover spans adjacent time slices\n";

    auto const& poset = chain.partialOrder;
    std::vector<int> layer_of(static_cast<std::size_t>(chain.nSites), -1);
    int idx = 0;
    for (int li = 0; li < static_cast<int>(chain.antichains.size()); ++li) {
        for (std::size_t k = 0; k < chain.antichains[static_cast<std::size_t>(li)].size(); ++k) {
            layer_of[static_cast<std::size_t>(idx)] = li;
            ++idx;
        }
    }

    auto cov = poset.covers();
    int bad = 0;
    int ok_pairs = 0;
    for (auto const& [a, b] : cov) {
        const int la = layer_of[static_cast<std::size_t>(a)];
        const int lb = layer_of[static_cast<std::size_t>(b)];
        if (la == -1 || lb == -1) { ++bad; continue; }
        if (std::abs(la - lb) != 1) ++bad;
        else ++ok_pairs;
    }
    const bool ok = (bad == 0) && (ok_pairs == static_cast<int>(cov.size()));
    std::cout << "  getCoverCount=" << cov.size()
              << "  adjacent_layer_covers=" << ok_pairs
              << "  bad=" << bad
              << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool acceptance_covers_match_hopping_pairs(
    tessera::quantum::CausetChain const& chain) {
    std::cout << "Acceptance #2 — covers == hoppingPairs on a foliated CDT\n";

    auto cov = chain.partialOrder.covers();
    std::set<std::pair<int, int>> covers_set(cov.begin(), cov.end());
    std::set<std::pair<int, int>> hops_set(
        chain.hoppingPairs.begin(), chain.hoppingPairs.end());

    const bool ok = (covers_set == hops_set);
    std::cout << "  |covers|=" << covers_set.size()
              << "  |hops|=" << hops_set.size()
              << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

// Compute the longest chain length (in cover edges) in a Hasse poset by
// dynamic programming on the cover-DAG. Equivalent to the rank function:
// rank(u) = max over predecessors p of rank(p) + 1, with rank of minimal
// elements = 0.
int longest_chain_length(tessera::Poset const& p) {
    const int n = p.getNodeCount();
    if (n == 0) return 0;
    // Adjacency: incoming covers per node.
    std::vector<std::vector<int>> in_edges(static_cast<std::size_t>(n));
    for (auto const& [a, b] : p.covers()) {
        in_edges[static_cast<std::size_t>(b)].push_back(a);
    }
    // Topological-order rank computation. Since the cover-DAG is a partial
    // order on integer indices and we built the Poset with ascending IDs
    // sorted by (time, ID), processing in node-index order is consistent
    // with the topological order of the foliation (predecessors always have
    // lower indices in our convention).
    std::vector<int> rank(static_cast<std::size_t>(n), 0);
    int best = 0;
    for (int u = 0; u < n; ++u) {
        for (int pred : in_edges[static_cast<std::size_t>(u)]) {
            const int candidate = rank[static_cast<std::size_t>(pred)] + 1;
            if (candidate > rank[static_cast<std::size_t>(u)])
                rank[static_cast<std::size_t>(u)] = candidate;
        }
        if (rank[static_cast<std::size_t>(u)] > best)
            best = rank[static_cast<std::size_t>(u)];
    }
    return best;
}

bool acceptance_height_equals_num_slices_minus_one(
    tessera::quantum::CausetChain const& chain) {
    std::cout << "Acceptance #3 — Hasse height = num_time_slices - 1\n";

    const int height = longest_chain_length(chain.partialOrder);
    const int expected = static_cast<int>(chain.times.size()) - 1;
    const bool ok = (height == expected);
    std::cout << "  num_layers=" << chain.times.size()
              << "  longest_chain=" << height
              << "  expected=" << expected
              << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool acceptance_total_extraction(tessera::Spacetime const& st,
                                 tessera::quantum::CausetChain const& chain) {
    std::cout << "Acceptance #4 — every Spacetime vertex appears in the Poset\n";

    auto const& vlist = st.getVertexList();
    int expected = vlist ? static_cast<int>(vlist->liveVector().size()) : 0;
    const bool ok = (chain.partialOrder.getNodeCount() == expected);
    std::cout << "  spacetime_vertices=" << expected
              << "  poset_nodes=" << chain.partialOrder.getNodeCount()
              << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool acceptance_layer_indegree_outdegree(
    tessera::quantum::CausetChain const& chain) {
    std::cout << "Acceptance #5 — top-layer has out-degree 0; bottom in-degree 0\n";

    const int n = chain.nSites;
    std::vector<int> outdeg(static_cast<std::size_t>(n), 0);
    std::vector<int> indeg(static_cast<std::size_t>(n), 0);
    for (auto const& [a, b] : chain.partialOrder.covers()) {
        outdeg[static_cast<std::size_t>(a)]++;
        indeg[static_cast<std::size_t>(b)]++;
    }

    // Layer assignment from chain.
    std::vector<int> layer_of(static_cast<std::size_t>(n), -1);
    int idx = 0;
    for (int li = 0; li < static_cast<int>(chain.antichains.size()); ++li) {
        for (std::size_t k = 0; k < chain.antichains[static_cast<std::size_t>(li)].size(); ++k) {
            layer_of[static_cast<std::size_t>(idx)] = li;
            ++idx;
        }
    }
    const int top_layer = static_cast<int>(chain.antichains.size()) - 1;
    const int bot_layer = 0;

    int top_out = 0, bot_in = 0;
    for (int u = 0; u < n; ++u) {
        if (layer_of[static_cast<std::size_t>(u)] == top_layer) {
            top_out += outdeg[static_cast<std::size_t>(u)];
        }
        if (layer_of[static_cast<std::size_t>(u)] == bot_layer) {
            bot_in  += indeg[static_cast<std::size_t>(u)];
        }
    }
    const bool ok = (top_out == 0) && (bot_in == 0);
    std::cout << "  top_layer_out_degree_total=" << top_out
              << "  bottom_layer_in_degree_total=" << bot_in
              << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

} // namespace

int main() {
    // Build one Toroid CDT and run all acceptance checks against it.
    // The Toroid build is deterministic (no rng), but reusing the same
    // Spacetime instance avoids any subtle move-construction effects on
    // the VertexList / EdgeList / simplexPool ownership semantics.
    auto st = build_toroid_cdt(60);
    auto chain = tessera::quantum::extractCausetChain(st);

    bool ok = true;
    ok &= acceptance_covers_span_adjacent_layers(chain);
    ok &= acceptance_covers_match_hopping_pairs(chain);
    ok &= acceptance_height_equals_num_slices_minus_one(chain);
    ok &= acceptance_total_extraction(st, chain);
    ok &= acceptance_layer_indegree_outdegree(chain);
    std::cout << (ok ? "\nALL PASS\n" : "\nFAILURES\n");
    return ok ? 0 : 1;
}
