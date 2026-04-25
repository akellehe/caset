// Phase 6 acceptance tests for caset::quantum::extract_causet_chain
// (docs/source/quantum-plan.md §6 — caset-embedded Schwinger lattice
// extraction).
//
// These tests exercise:
//
//   (1) Trivial chain — N time slices each with a single vertex; the
//       extractor collapses to a regular 1D lattice with hopping pairs
//       (0,1), (1,2), …, (N-2, N-1). This is the case the existing
//       Schwinger MPO can run on without any modification — it's the
//       baseline we need before exploring branching causets.
//
//   (2) Branching antichain — a single t=0 vertex linked to two t=1
//       vertices. The flat lattice has 3 sites; hopping pairs are
//       (0, 1) and (0, 2). This is the configuration that will
//       eventually need a tree-tensor-network MPO; here we just verify
//       the data extraction is correct.
//
//   (3) Empty Spacetime — empty CausetChain.
//
//   (4) Sparse vertex IDs — non-contiguous Spacetime IDs are densely
//       remapped to flat lattice indices in a way consistent with
//       Poset::from_spacetime, so the inherited partial_order's node
//       IDs match the flat lattice indices used in hopping_pairs.
//
//   (5) Skipping edges — a timelike edge that spans two slices (t=0
//       to t=2 with t=1 also populated) is NOT included in
//       hopping_pairs (would not be a Hasse cover).

#include "Poset.h"
#include "quantum/causet_chain.hpp"
#include "spacetime/Spacetime.h"

#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

namespace {

caset::VertexPtr make_vertex(caset::Spacetime& st, std::uint64_t id, int t) {
    return st.createVertex(id, std::vector<double>{static_cast<double>(t)});
}

bool pairs_equal(std::vector<std::pair<int, int>> got,
                 std::vector<std::pair<int, int>> want) {
    std::sort(got.begin(), got.end());
    std::sort(want.begin(), want.end());
    return got == want;
}

bool acceptance_trivial_chain() {
    std::cout << "Acceptance #1 — N=4 trivial chain → uniform 1D lattice\n";

    caset::Spacetime st;
    auto v0 = make_vertex(st, 0, 0);
    auto v1 = make_vertex(st, 1, 1);
    auto v2 = make_vertex(st, 2, 2);
    auto v3 = make_vertex(st, 3, 3);
    st.createEdge(v0, v1, -1.0);
    st.createEdge(v1, v2, -1.0);
    st.createEdge(v2, v3, -1.0);

    auto chain = caset::quantum::extract_causet_chain(st);

    const std::vector<std::pair<int, int>> want_hops{
        {0, 1}, {1, 2}, {2, 3}
    };
    bool ok = true;
    ok &= (chain.n_sites == 4);
    ok &= (chain.times.size() == 4);
    ok &= (chain.antichains.size() == 4);
    ok &= pairs_equal(chain.hopping_pairs, want_hops);
    ok &= (chain.partial_order.n_nodes() == 4);
    ok &= (chain.partial_order.covers().size() == 3);

    std::cout << "  n_sites=" << chain.n_sites
              << "  hops=" << chain.hopping_pairs.size()
              << "  poset_covers=" << chain.partial_order.covers().size()
              << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool acceptance_branching_antichain() {
    std::cout << "Acceptance #2 — branching causet (1 root, 2 leaves)\n";

    caset::Spacetime st;
    auto v0 = make_vertex(st, 0, 0);
    auto v1 = make_vertex(st, 1, 1);
    auto v2 = make_vertex(st, 2, 1);
    st.createEdge(v0, v1, -1.0);
    st.createEdge(v0, v2, -1.0);

    auto chain = caset::quantum::extract_causet_chain(st);

    bool ok = true;
    ok &= (chain.n_sites == 3);
    ok &= (chain.times.size() == 2);  // t=0, t=1
    ok &= (chain.antichains.size() == 2);
    ok &= (chain.antichains[0].size() == 1);  // {0}
    ok &= (chain.antichains[1].size() == 2);  // {1, 2}
    // Flat layout: site 0 = vid 0, site 1 = vid 1, site 2 = vid 2.
    // Both timelike edges connect site 0 to site 1 / 2 → two hops.
    const std::vector<std::pair<int, int>> want_hops{{0, 1}, {0, 2}};
    ok &= pairs_equal(chain.hopping_pairs, want_hops);
    // partial_order should also see two covers: 0→1 and 0→2.
    const std::vector<std::pair<int, int>> want_covers{{0, 1}, {0, 2}};
    auto got_covers = chain.partial_order.covers();
    ok &= pairs_equal(got_covers, want_covers);

    std::cout << "  n_sites=" << chain.n_sites
              << "  antichain_sizes=[" << chain.antichains[0].size()
              << "," << chain.antichains[1].size() << "]"
              << "  hops=" << chain.hopping_pairs.size()
              << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool acceptance_empty_spacetime() {
    std::cout << "Acceptance #3 — empty Spacetime → empty CausetChain\n";

    caset::Spacetime st;
    auto chain = caset::quantum::extract_causet_chain(st);

    bool ok = (chain.n_sites == 0)
            && chain.times.empty()
            && chain.antichains.empty()
            && chain.vertex_ids.empty()
            && chain.hopping_pairs.empty()
            && (chain.partial_order.n_nodes() == 0);

    std::cout << "  n_sites=" << chain.n_sites
              << "  times=" << chain.times.size()
              << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool acceptance_sparse_ids() {
    std::cout << "Acceptance #4 — sparse Spacetime IDs → dense flat indices\n";

    caset::Spacetime st;
    auto va = make_vertex(st, 7,  0);
    auto vb = make_vertex(st, 11, 1);
    auto vc = make_vertex(st, 19, 2);
    st.createEdge(va, vb, -1.0);
    st.createEdge(vb, vc, -1.0);

    auto chain = caset::quantum::extract_causet_chain(st);

    // Sorted ascending IDs: 7 → site 0, 11 → site 1, 19 → site 2.
    bool ok = true;
    ok &= (chain.n_sites == 3);
    ok &= (chain.vertex_ids.size() == 3);
    ok &= (chain.vertex_ids[0] == 7);
    ok &= (chain.vertex_ids[1] == 11);
    ok &= (chain.vertex_ids[2] == 19);
    const std::vector<std::pair<int, int>> want{{0, 1}, {1, 2}};
    ok &= pairs_equal(chain.hopping_pairs, want);

    std::cout << "  vertex_ids=[" << chain.vertex_ids[0]
              << "," << chain.vertex_ids[1]
              << "," << chain.vertex_ids[2] << "]"
              << "  hops_match=" << pairs_equal(chain.hopping_pairs, want)
              << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool acceptance_skipping_edge_dropped() {
    std::cout << "Acceptance #5 — skipping (t=0 → t=2) timelike edge dropped\n";

    caset::Spacetime st;
    auto v0 = make_vertex(st, 0, 0);
    auto v1 = make_vertex(st, 1, 1);
    auto v2 = make_vertex(st, 2, 2);
    st.createEdge(v0, v1, -1.0);
    st.createEdge(v1, v2, -1.0);
    st.createEdge(v0, v2, -1.0);  // skips t=1 — should NOT be a hop

    auto chain = caset::quantum::extract_causet_chain(st);

    const std::vector<std::pair<int, int>> want_hops{{0, 1}, {1, 2}};
    bool ok = pairs_equal(chain.hopping_pairs, want_hops);
    // partial_order should also reduce 0→2 out as a non-cover.
    const std::vector<std::pair<int, int>> want_covers{{0, 1}, {1, 2}};
    ok &= pairs_equal(chain.partial_order.covers(), want_covers);

    std::cout << "  hops=" << chain.hopping_pairs.size()
              << " (want 2)  poset_covers=" << chain.partial_order.covers().size()
              << " (want 2)  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

} // namespace

int main() {
    bool ok = true;
    ok &= acceptance_trivial_chain();
    ok &= acceptance_branching_antichain();
    ok &= acceptance_empty_spacetime();
    ok &= acceptance_sparse_ids();
    ok &= acceptance_skipping_edge_dropped();
    std::cout << (ok ? "\nALL PASS\n" : "\nFAILURES\n");
    return ok ? 0 : 1;
}
