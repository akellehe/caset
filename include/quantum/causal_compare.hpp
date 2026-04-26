// Phase 5: causal-order comparison between three partial orders on the
// label set $\{(\text{cut}, t)\}$ produced by a TDVP run with snapshot
// recording enabled. PLAN.md §5 Phase 5 / methodology page §1, §4.4.
//
// ─── The three orders ────────────────────────────────────────────────────
//
//   1. ≼_maj — majorization order: (A, s) ≼_maj (B, t) iff the Schmidt
//      spectrum λ_A(s) is majorized by λ_B(t). Built by feeding all
//      snapshot spectra (across cuts AND times) into the Phase 3
//      majorizationPoset() routine. Per PLAN.md §3 hypothesis: this
//      order is meant to "see" the entanglement causal structure.
//
//   2. ≼_LR — Lieb-Robinson cone: (A, s) ≼_LR (B, t) iff s < t AND the
//      shortest distance between intervals A and B is ≤ vLr · (t − s).
//      The strict order has only cross-time edges; same-time pairs are
//      incomparable in this ordering. Within-cone information transport
//      bound from {cite}`LiebRobinson1972, HastingsKoma2006`.
//
//   3. ≼_cs — causet order: on the regular chain (Phase 5 scope) this
//      is just the time order: (A, s) ≼_cs (B, t) iff s < t. Phase 6
//      replaces the regular chain with a non-trivial causet, at which
//      point ≼_cs becomes informative within time slices too.
//
// Each order is stored as a Hasse-cover Poset over the same shared label
// set. compareOrders() then computes Kendall-τ, the discordant-pair
// fraction, and the Hasse-graph edit distance between any two of the
// three.
//
// ─── Lieb-Robinson velocity ──────────────────────────────────────────────
//
// Plan §7 trap: "OTOC computation: expensive. Compute only at the final
// time and at one or two intermediate times for vLr estimation; do not
// run every step." For v1 we accept vLr as a config input rather than
// extracting it from OTOC fronts. For our Schwinger spin Hamiltonian the
// natural scale is vLr ≈ 1/a (free-fermion group velocity) — that is
// the default in the high-level `computeCausalComparison` entry point,
// and the Phase 5 acceptance test explicitly tries vLr = 2/a too to
// see the cone-tightening effect.

#pragma once

#include "quantum/majorization.hpp"
#include "quantum/schmidt.hpp"
#include "quantum/tdvp_runner.hpp"

#include <vector>

namespace tessera::quantum {

// One node in the (cut, time) label set.
struct LabelSpacetime {
    int      cutIdx{0};      // index into the snapshot's spectra/intervals
    int      tIdx{0};        // index into TDVPSnapshot list
    int      intervalI{0};   // the contiguous interval [intervalI, intervalJ]
    int      intervalJ{0};
    double   time{0.0};       // physical time at this snapshot
};

// All three orders on the same label set, plus the labels themselves.
// Each Poset stores Hasse cover edges (transitive reduction of the strict
// order). A node `k` corresponds to `labels[k]`.
struct CausalOrders {
    std::vector<LabelSpacetime> labels;
    Poset maj;     // strict-majorization
    Poset lr;      // Lieb-Robinson cone
    Poset cs;      // causet (time-only on regular chain)
};

// Pairwise agreement statistics on a single pair of posets.
//
// Counted over UNORDERED pairs (i, j) with i < j:
//   * "comparable in P" = the transitive closure of P relates i to j or
//     j to i (cover edges suffice to determine this).
//   * "concordant" = both posets relate the pair, in the same direction.
//   * "discordant" = both posets relate the pair, in opposite directions.
//
// Kendall-τ is on the (concordant, discordant) subset:
//     τ = (nConcordant − nDiscordant) / nComparableBoth.
//
// hasseEditDistance is the symmetric difference of cover-edge sets,
// normalized by the union size.
// OrderAgreement is defined at top-level tessera (see include/Poset.h).
// The using-alias in include/quantum/majorization.hpp keeps the old
// `tessera::quantum::OrderAgreement` spelling available.

struct CausalComparisonReport {
    OrderAgreement majVsLr;
    OrderAgreement majVsCs;
    OrderAgreement lrVsCs;
    int nLabels{0};
    int nSnapshots{0};
    double vLr{0.0};   // Lieb-Robinson velocity used to build ≼_LR
};

// Build the cross-time majorization poset from a list of snapshots
// (which must have been recorded with recordSpectra=true). The first
// dim labels in the returned CausalOrders are from snapshots[0], the
// next from snapshots[1], etc.
CausalOrders buildCausalOrders(std::vector<TDVPSnapshot> const& snapshots,
                                 double vLr);

// tessera::compareOrders is the canonical implementation; the alias in
// include/quantum/majorization.hpp re-exports it as
// tessera::quantum::compareOrders for back-compat.

// End-to-end pipeline: DMRG ground state → q-qbar quench → TDVP loop with
// per-step Schmidt spectra → build the three orders → compare. Forces
// `tdvp_cfg.recordSpectra = true` regardless of the input.
//
// `vLr` is the Lieb-Robinson velocity in lattice units (sites / time).
// Default 1.0 corresponds to the free-fermion group velocity for our
// hopping coefficient.
CausalComparisonReport
computeCausalComparison(TDVPConfig const& tdvp_cfg, double vLr = 1.0);

} // namespace tessera::quantum
