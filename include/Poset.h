// MIT License -- Copyright (c) 2025 Andrew Kelleher
//
// Hasse-cover representation of a finite partial order over integer-
// indexed nodes. Storage uses tessera's standard mesh primitives
// (tessera::VertexList, tessera::EdgeList) so Poset instances interoperate
// with the rest of tessera's graph machinery — rendering, GraphML / dot
// export, and the Phase 6 causet inheritance share the same Vertex /
// Edge types we already use for the simplicial spacetime.
//
// ─── What this provides ───────────────────────────────────────────────────
//
// • Poset itself — the cover-edge container with addCover / covers /
//   toDot / fromSpacetime.
// • OrderAgreement struct — pairwise statistics on two posets that share
//   a label set.
// • compareOrders() — Kendall-τ, discordant fraction, Hasse edit
//   distance.
//
// All three are general-purpose; the quantum subsystem layers
// majorization-specific glue on top of them in
// include/quantum/majorization.hpp.

#pragma once

#include "mesh/EdgeList.h"
#include "mesh/VertexList.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace tessera {

class Spacetime;  // forward decl for Poset::fromSpacetime (Phase 6)

// Hasse-cover representation of a partial order on integer-indexed nodes.
//
// Internally stores a VertexList (one Vertex per node, no coordinates) and
// an EdgeList of cover edges. Cover edges are tessera::Edge objects; their
// `squaredLength` and `disposition` fields are unused by the partial-order
// semantics and stay at their default values. (See note at the bottom of
// this header about the "compiler flag to pare them down" idea — currently
// not implemented; the per-edge cost is acceptable for our problem sizes.)
class Poset {
public:
    // Default-construct an empty Poset.
    Poset() = default;

    // Construct with `getNodeCount` pre-populated as Vertex(id=0..n-1).
    explicit Poset(int getNodeCount);

    // Pool-based VertexList / EdgeList don't trivially copy because of
    // the back-pointer / fingerprint maps; implement value-style copy
    // explicitly so callers can pass Poset by value.
    Poset(Poset const& other);
    Poset& operator=(Poset const& other);
    Poset(Poset&& other) noexcept = default;
    Poset& operator=(Poset&& other) noexcept = default;

    // Number of nodes (0 .. getNodeCount()-1).
    int getNodeCount() const noexcept;

    // Resize: add empty Vertex(id) entries for missing IDs in [0, n).
    // Does NOT remove existing nodes if n is smaller than the current
    // count. Cover edges are preserved across resizes.
    void setNodeCount(int n);

    // Number of cover edges currently registered.
    int getCoverCount() const noexcept;

    // Add a strict cover edge a → b (a strictly precedes b in the partial
    // order with no intermediate). Caller is responsible for ensuring
    // transitivity and acyclicity; Poset itself does not validate.
    //
    // Both endpoints must already exist (setNodeCount() must have been
    // called or the constructor must have been given a sufficient n).
    // No deduplication is performed — adding the same cover twice
    // creates two parallel edges. Standard usage feeds covers from a
    // transitive-reduction algorithm where duplicates can't arise.
    void addCover(int a, int b);

    // Replace the entire cover list with `new_covers`. Equivalent to
    // clearing edges_ and calling addCover() for each pair, but in one
    // pass.
    void setCovers(std::vector<std::pair<int, int>> const& new_covers);

    // Materialize cover edges as (from, to) integer pairs. Order is
    // insertion order (don't rely on it for correctness — the underlying
    // EdgeList may re-order on free-slot reuse).
    std::vector<std::pair<int, int>> covers() const;

    // Underlying mesh primitives — exposed for tessera interop
    // (visualization, GraphML export, future Spacetime-aware analyses).
    // Mutating via these handles bypasses Poset's invariants; treat as
    // read-only.
    VertexList const& vertices() const noexcept { return vertices_; }
    EdgeList const& edges() const noexcept { return edges_; }
    VertexList& vertices() noexcept { return vertices_; }
    EdgeList& edges() noexcept { return edges_; }

    // ─── Factories ─────────────────────────────────────────────────────

    // Build the causet partial order on the vertices of a tessera Spacetime
    // using the directed-edge / timelike-edge subgraph as the strict
    // precedes-relation, then transitively-reducing to cover edges.
    //
    // Phase 6 of the quantum-plan extends this for the (cut, time)
    // label set used by majorization-vs-LR-vs-causet comparison;
    // currently this base version is implemented as a Phase-6 stub.
    static Poset fromSpacetime(Spacetime const& st);

    // ─── Output ────────────────────────────────────────────────────────

    // Graphviz DOT representation of the Hasse diagram. Nodes labelled by
    // their integer id; cover edges drawn from "a -> b" meaning
    // a strictly precedes b. Suitable for `dot -Tsvg` rendering.
    std::string toDot() const;

private:
    VertexList vertices_;
    EdgeList   edges_;
};

// ─── Pairwise agreement between two posets on a shared label set ─────────

// Counted over UNORDERED label pairs (i, j) with i < j:
//   * "comparable in P" — transitive closure of P relates i to j.
//   * "concordant"      — both relate the pair, in the same direction.
//   * "discordant"      — both relate the pair, in opposite directions.
//   * "only_a"          — A relates the pair but B does not.
//   * "only_b"          — B relates the pair but A does not.
//
// The five counts (concordant, discordant, only_a, only_b, neither) form
// a partition of the C(nLabels, 2) unordered pairs. The
// strong-falsification criterion (quantum-methodology.md §1.2 #1) reads
// directly off `only_a` when (A, B) = (≼_maj, ≼_LR): it's the count of
// majorization pairs whose endpoints lie OUTSIDE the LR cone. Symmetric
// for `only_b`.
struct OrderAgreement {
    double kendallTau{0.0};         // (concordant - discordant) / both, in [-1, 1]
    double discordantFraction{0.0}; // discordant / both, in [0, 1]
    double hasseEditDistance{0.0}; // |E_a △ E_b| / |E_a ∪ E_b|, in [0, 1]
    int    nConcordant{0};
    int    nDiscordant{0};
    int    nComparableBoth{0};
    int    nOnlyA{0};              // pairs related by a only
    int    nOnlyB{0};              // pairs related by b only
};

// Pairwise agreement statistics between two posets on the same label set
// of size nLabels.
//
// Complexity: O(nLabels^3) for the Floyd-Warshall transitive closures,
// then O(nLabels^2) to count pairs. Practical on nLabels up to a few
// thousand.
OrderAgreement compareOrders(Poset const& a,
                              Poset const& b,
                              int nLabels);

} // namespace tessera

// ─── Notes ─────────────────────────────────────────────────────────────────
//
// "compiler flags to pare it down" (per architectural discussion):
// tessera::Edge currently carries `squaredLength` (8 B) and `disposition`
// (1 B + padding) which are unused for Poset cover edges. A future
// TESSERA_LIGHTWEIGHT_EDGES build flag could conditionally drop those
// fields (and make the Edge struct ~16 B instead of ~32 B), halving the
// per-cover memory cost in large posets. Not implemented yet — current
// per-cover overhead (~70 B counting EdgeList map entry) is acceptable
// for the Phase 5 problem sizes (hundreds of thousands of cover edges).
