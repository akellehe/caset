// Causal-order comparison data classes plus the
// `CausalOrders::fromSnapshots` factory. PLAN.md §5 / methodology
// page §1, §4.4.
//
// ─── The three orders ────────────────────────────────────────────────────
//
//   1. ≼_maj — majorization order: (A, s) ≼_maj (B, t) iff the Schmidt
//      spectrum λ_A(s) is majorized by λ_B(t). Built by feeding all
//      snapshot spectra (across cuts AND times) into the
//      Majorization::posetOf routine. The hypothesis that this order
//      "sees" the entanglement causal structure is the methodology
//      page's central claim.
//
//   2. ≼_LR — Lieb-Robinson cone: (A, s) ≼_LR (B, t) iff s < t AND the
//      shortest distance between intervals A and B is ≤ vLr · (t − s).
//      Within-cone information transport bound from
//      {LiebRobinson1972, HastingsKoma2006}.
//
//   3. ≼_cs — causet order: on a regular chain this is just the time
//      order: (A, s) ≼_cs (B, t) iff s < t. Replacing the chain with
//      a non-trivial causet (see the Causet adapter) makes ≼_cs
//      informative within time slices too.
//
// Each order is stored as a Hasse-cover Poset over the same shared
// label set. `Majorization::agreement` then computes Kendall-τ, the
// discordant-pair fraction, and the Hasse-graph edit distance between
// any two of the three.
//
// The end-to-end pipeline (config → snapshots → orders → report) lives
// on SchwingerQuench (see tdvp_runner.hpp). This file just defines the
// data types and the orders factory.

#pragma once

#include "quantum/Majorization.hpp"
#include "quantum/Schmidt.hpp"

#include <string>
#include <vector>

namespace tessera::quantum {

// Forward decl — TDVPSnapshot is defined in tdvp_runner.hpp; we only
// need its name to declare CausalOrders::fromSnapshots.
struct TDVPSnapshot;

// One node in the (cut, time) label set.
struct LabelSpacetime {
    int    cutIdx{0};      // index into the snapshot's spectra/intervals
    int    tIdx{0};        // index into TDVPSnapshot list
    int    intervalI{0};   // the contiguous interval [intervalI, intervalJ]
    int    intervalJ{0};
    double time{0.0};       // physical time at this snapshot
};

// All three orders on the same label set, plus the labels themselves.
// Each Poset stores Hasse cover edges (transitive reduction of the
// strict order). A node `k` corresponds to `labels[k]`.
struct CausalOrders {
    std::vector<LabelSpacetime> labels;
    Poset maj;     // strict-majorization
    Poset lr;      // Lieb-Robinson cone
    Poset cs;      // causet (time-only on regular chain)

    // Build the cross-time majorization poset, the Lieb-Robinson cone
    // poset, and the (regular-chain) causet poset from a list of
    // snapshots. Snapshots must have been recorded with
    // `recordSpectra=true`.
    //
    // `predicate` selects the majorization variant for ≼_maj. nullptr
    // means classical {N1999} majorization (StandardMajorization{1e-12}).
    [[nodiscard]] static CausalOrders fromSnapshots(
        std::vector<TDVPSnapshot> const& snapshots,
        double vLr,
        MajorizationPredicate const* predicate = nullptr);
};

// Result struct for SchwingerQuench::compareCausalOrders.
struct CausalComparisonReport {
    OrderAgreement majVsLr;
    OrderAgreement majVsCs;
    OrderAgreement lrVsCs;
    int nLabels{0};
    int nSnapshots{0};
    double      vLr{0.0};                  // LR velocity used to build ≼_LR
    std::string majKind{"standard"};       // MajorizationPredicate::name() for ≼_maj
};

} // namespace tessera::quantum
