// Phase 6 acceptance tests (docs/source/quantum-plan.md §6):
//
// caset::Poset::from_spacetime(Spacetime const&) inherits a partial order
// from a caset::Spacetime by treating timelike edges (Edge::getSquaredLength
// < 0) as strict precedes-relations oriented earliest-time → latest-time,
// then transitively reducing the resulting DAG to its Hasse covers.
//
// These tests build small hand-crafted Spacetimes and verify:
//
//   (1) Two-slice ladder — every cross-slice timelike edge is a cover.
//   (2) Three-slice chain with a "skipping" edge — transitive reduction
//       collapses the skipping edge so that only adjacent-slice covers
//       remain.
//   (3) Empty Spacetime — empty Poset.
//   (4) Vertices with only spacelike edges — no covers (nodes still all
//       present).
//   (5) Self-comparison under compare_orders — Kendall-τ = 1, edit
//       distance = 0.
//
// We bypass the topology-driven `Spacetime::build()` path and use the
// public `createVertex(id, coords)` / `createEdge(src, tgt, sq)` API
// directly. That keeps the tests free of any CDT / Toroid topology
// build requirements while still exercising the real VertexList /
// EdgeList machinery.

#include "Poset.h"
#include "spacetime/Spacetime.h"

#include <algorithm>
#include <iostream>
#include <set>
#include <utility>
#include <vector>

namespace {

// Convenience: build a vertex at integer time `t`, ID `id`, with 1D
// coords {static_cast<double>(t)}. Vertex::getTime() returns |x_0| for
// 1D coords, so this gives integer-valued times that match the
// time-slice / antichain pattern Phase 6 needs.
caset::VertexPtr make_vertex(caset::Spacetime& st, std::uint64_t id, int t) {
    return st.createVertex(id, std::vector<double>{static_cast<double>(t)});
}

bool covers_equal(std::vector<std::pair<int, int>> got,
                  std::vector<std::pair<int, int>> want) {
    std::sort(got.begin(), got.end());
    std::sort(want.begin(), want.end());
    return got == want;
}

bool acceptance_two_slice_ladder() {
    std::cout << "Acceptance #1 — 2-slice ladder, every cross-slice edge is a cover\n";

    caset::Spacetime st;
    auto v0 = make_vertex(st, 0, 0);
    auto v1 = make_vertex(st, 1, 0);
    auto v2 = make_vertex(st, 2, 1);
    auto v3 = make_vertex(st, 3, 1);
    // Four cross-slice timelike edges (squaredLength < 0). No same-time
    // edges, so spacelike at the slice level is implicit.
    st.createEdge(v0, v2, -1.0);
    st.createEdge(v0, v3, -1.0);
    st.createEdge(v1, v2, -1.0);
    st.createEdge(v1, v3, -1.0);

    auto p = caset::Poset::from_spacetime(st);
    const std::vector<std::pair<int, int>> want{
        {0, 2}, {0, 3}, {1, 2}, {1, 3}
    };
    const auto got = p.covers();
    const bool ok = (p.n_nodes() == 4) && covers_equal(got, want);

    std::cout << "  n_nodes=" << p.n_nodes()
              << " (want 4)  covers=" << got.size()
              << " (want " << want.size() << ")  "
              << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool acceptance_three_slice_with_skip() {
    std::cout << "Acceptance #2 — 3-slice chain, skipping edge is reduced\n";

    caset::Spacetime st;
    auto v0 = make_vertex(st, 0, 0);
    auto v1 = make_vertex(st, 1, 1);
    auto v2 = make_vertex(st, 2, 2);
    // Adjacent-slice edges + a "skip" edge from t=0 directly to t=2.
    // The strict precedes-DAG has 0→1, 1→2, 0→2. Transitive reduction
    // should drop 0→2 because 0→1→2 already provides that order.
    st.createEdge(v0, v1, -1.0);
    st.createEdge(v1, v2, -1.0);
    st.createEdge(v0, v2, -1.0);

    auto p = caset::Poset::from_spacetime(st);
    const std::vector<std::pair<int, int>> want{{0, 1}, {1, 2}};
    const auto got = p.covers();
    const bool ok = (p.n_nodes() == 3) && covers_equal(got, want);

    std::cout << "  n_nodes=" << p.n_nodes()
              << "  covers={ ";
    for (auto [a, b] : got) std::cout << "(" << a << "," << b << ") ";
    std::cout << "}  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool acceptance_empty_spacetime() {
    std::cout << "Acceptance #3 — empty Spacetime → empty Poset\n";
    caset::Spacetime st;
    auto p = caset::Poset::from_spacetime(st);
    const bool ok = (p.n_nodes() == 0) && p.covers().empty();
    std::cout << "  n_nodes=" << p.n_nodes()
              << "  covers=" << p.covers().size()
              << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool acceptance_only_spacelike_edges() {
    std::cout << "Acceptance #4 — same-slice spacelike-only graph → no covers\n";

    caset::Spacetime st;
    auto v0 = make_vertex(st, 0, 0);
    auto v1 = make_vertex(st, 1, 0);
    auto v2 = make_vertex(st, 2, 0);
    // Spacelike edges only — squaredLength > 0, all at the same time
    // slice. Should produce a Poset with 3 nodes and 0 covers.
    st.createEdge(v0, v1, +1.0);
    st.createEdge(v1, v2, +1.0);

    auto p = caset::Poset::from_spacetime(st);
    const bool ok = (p.n_nodes() == 3) && p.covers().empty();
    std::cout << "  n_nodes=" << p.n_nodes()
              << "  covers=" << p.covers().size()
              << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool acceptance_self_comparison() {
    std::cout << "Acceptance #5 — compare_orders(p, p) = perfect agreement\n";

    caset::Spacetime st;
    auto v0 = make_vertex(st, 0, 0);
    auto v1 = make_vertex(st, 1, 1);
    auto v2 = make_vertex(st, 2, 2);
    auto v3 = make_vertex(st, 3, 1);
    st.createEdge(v0, v1, -1.0);
    st.createEdge(v0, v3, -1.0);
    st.createEdge(v1, v2, -1.0);
    st.createEdge(v3, v2, -1.0);

    auto p = caset::Poset::from_spacetime(st);
    auto agree = caset::compare_orders(p, p, p.n_nodes());

    const bool ok = (agree.kendall_tau == 1.0) &&
                    (agree.discordant_fraction == 0.0) &&
                    (agree.hasse_edit_distance == 0.0) &&
                    (agree.n_discordant == 0);
    std::cout << "  τ=" << agree.kendall_tau
              << "  discordant_frac=" << agree.discordant_fraction
              << "  edit_dist=" << agree.hasse_edit_distance
              << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool acceptance_dense_remap() {
    std::cout << "Acceptance #6 — sparse Spacetime IDs are densely remapped\n";

    // Use non-contiguous spacetime vertex IDs (0, 5, 11, 13). The Poset
    // should remap them densely to 0..3 in ascending-ID order, so
    // ID-monotonic edges produce ID-monotonic covers in Poset space.
    caset::Spacetime st;
    auto v_lo = make_vertex(st, 0,  0);
    auto v_md = make_vertex(st, 5,  1);
    auto v_hi = make_vertex(st, 11, 2);
    auto v_xx = make_vertex(st, 13, 2);
    st.createEdge(v_lo, v_md, -1.0);
    st.createEdge(v_md, v_hi, -1.0);
    st.createEdge(v_md, v_xx, -1.0);

    auto p = caset::Poset::from_spacetime(st);
    // Ascending sort of {0, 5, 11, 13} → indices 0, 1, 2, 3. Covers
    // (0, 5)→(0, 1), (5, 11)→(1, 2), (5, 13)→(1, 3).
    const std::vector<std::pair<int, int>> want{{0, 1}, {1, 2}, {1, 3}};
    const auto got = p.covers();
    const bool ok = (p.n_nodes() == 4) && covers_equal(got, want);

    std::cout << "  n_nodes=" << p.n_nodes()
              << "  covers={ ";
    for (auto [a, b] : got) std::cout << "(" << a << "," << b << ") ";
    std::cout << "}  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool acceptance_to_dot_format() {
    std::cout << "Acceptance #7 — to_dot() emits the cover edges\n";
    caset::Spacetime st;
    auto v0 = make_vertex(st, 0, 0);
    auto v1 = make_vertex(st, 1, 1);
    st.createEdge(v0, v1, -1.0);

    auto p = caset::Poset::from_spacetime(st);
    const auto dot = p.to_dot();
    const bool ok = (dot.find("digraph poset")  != std::string::npos) &&
                    (dot.find("0 -> 1")          != std::string::npos);
    std::cout << "  contains digraph header=" << (dot.find("digraph poset")  != std::string::npos)
              << " contains '0 -> 1'=" << (dot.find("0 -> 1") != std::string::npos)
              << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

} // namespace

int main() {
    bool ok = true;
    ok &= acceptance_two_slice_ladder();
    ok &= acceptance_three_slice_with_skip();
    ok &= acceptance_empty_spacetime();
    ok &= acceptance_only_spacelike_edges();
    ok &= acceptance_self_comparison();
    ok &= acceptance_dense_remap();
    ok &= acceptance_to_dot_format();
    std::cout << (ok ? "\nALL PASS\n" : "\nFAILURES\n");
    return ok ? 0 : 1;
}
