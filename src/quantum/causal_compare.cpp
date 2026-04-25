// Implementation of the Phase 5 causal-order comparison. See
// include/quantum/causal_compare.hpp for the design.

#include "quantum/causal_compare.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <utility>

namespace caset::quantum {

namespace {

// Closest-pair distance between contiguous intervals A = [i1, j1] and
// B = [i2, j2]. Returns 0 if they overlap, otherwise the integer gap
// between the disjoint ranges.
int interval_distance(int i1, int j1, int i2, int j2) {
    if (j1 < i2) return i2 - j1;     // B is strictly to the right of A
    if (j2 < i1) return i1 - j2;     // A is strictly to the right of B
    return 0;                        // overlap or touch
}

// Build a Poset from a directed boolean adjacency matrix `strict[i][j]`
// of strict-precedes relations. Applies transitive reduction so the
// returned Poset has Hasse cover edges only.
Poset poset_from_strict(std::vector<std::vector<char>> const& strict) {
    const int n = static_cast<int>(strict.size());
    Poset p(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (!strict[static_cast<std::size_t>(i)]
                       [static_cast<std::size_t>(j)]) continue;
            // Cover iff no intermediate k.
            bool has_intermediate = false;
            for (int k = 0; k < n; ++k) {
                if (k == i || k == j) continue;
                if (strict[static_cast<std::size_t>(i)]
                          [static_cast<std::size_t>(k)] &&
                    strict[static_cast<std::size_t>(k)]
                          [static_cast<std::size_t>(j)]) {
                    has_intermediate = true;
                    break;
                }
            }
            if (!has_intermediate) p.add_cover(i, j);
        }
    }
    return p;
}

// Aggregate every (cut, time) into a flat label list, plus collect the
// corresponding spectrum into a flat vector-of-vectors so we can call
// majorization_poset() once.
struct Flattened {
    std::vector<LabelSpacetime>          labels;
    std::vector<std::vector<double>>     spectra;
};

Flattened flatten_snapshots(std::vector<TDVPSnapshot> const& snapshots) {
    Flattened out;
    for (int t_idx = 0;
         t_idx < static_cast<int>(snapshots.size()); ++t_idx) {
        auto const& snap = snapshots[static_cast<std::size_t>(t_idx)];
        if (snap.spectra.spectra.empty()) {
            throw std::runtime_error(
                "build_causal_orders: snapshots must have record_spectra=true");
        }
        for (int k = 0;
             k < static_cast<int>(snap.spectra.spectra.size()); ++k) {
            out.labels.push_back({k, t_idx,
                snap.spectra.intervals[static_cast<std::size_t>(k)].i,
                snap.spectra.intervals[static_cast<std::size_t>(k)].j,
                snap.time});
            out.spectra.push_back(snap.spectra.spectra[static_cast<std::size_t>(k)]);
        }
    }
    return out;
}

// Build the LR-cone Hasse poset:
//   (a, b) is a strict-LR edge iff
//       labels[a].time < labels[b].time AND
//       interval_distance(A, B)  ≤  v_LR · (labels[b].time - labels[a].time)
Poset build_lr_poset(std::vector<LabelSpacetime> const& labels, double v_LR) {
    const int n = static_cast<int>(labels.size());
    std::vector<std::vector<char>> strict(static_cast<std::size_t>(n),
                                          std::vector<char>(
                                              static_cast<std::size_t>(n), 0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            const auto& A = labels[static_cast<std::size_t>(i)];
            const auto& B = labels[static_cast<std::size_t>(j)];
            if (A.time >= B.time) continue;       // strictly cross-time
            const double dt = B.time - A.time;
            const int    d  = interval_distance(A.interval_i, A.interval_j,
                                                B.interval_i, B.interval_j);
            if (static_cast<double>(d) <= v_LR * dt) {
                strict[static_cast<std::size_t>(i)]
                      [static_cast<std::size_t>(j)] = 1;
            }
        }
    }
    return poset_from_strict(strict);
}

// Build the regular-chain causet Hasse poset:
//   (a, b) is strict iff labels[a].t_idx < labels[b].t_idx.
// On the regular chain there's no spatial structure to add — the causet
// reduces to the time-slice ordering. Phase 6 (causet-embedded chain)
// is where this becomes interesting; this stub gives a useful baseline
// for the comparison pipeline.
Poset build_cs_poset_regular_chain(std::vector<LabelSpacetime> const& labels) {
    const int n = static_cast<int>(labels.size());
    std::vector<std::vector<char>> strict(static_cast<std::size_t>(n),
                                          std::vector<char>(
                                              static_cast<std::size_t>(n), 0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (labels[static_cast<std::size_t>(i)].t_idx
              < labels[static_cast<std::size_t>(j)].t_idx) {
                strict[static_cast<std::size_t>(i)]
                      [static_cast<std::size_t>(j)] = 1;
            }
        }
    }
    return poset_from_strict(strict);
}

} // namespace

// compare_orders is implemented at top-level caset (see src/Poset.cpp);
// the using-alias in include/quantum/majorization.hpp re-exports it as
// caset::quantum::compare_orders for back-compat.

CausalOrders build_causal_orders(std::vector<TDVPSnapshot> const& snapshots,
                                 double v_LR) {
    auto flat = flatten_snapshots(snapshots);

    CausalOrders out;
    out.labels = std::move(flat.labels);
    // Majorization across the full flat label set — this is the same
    // routine used by Phase 3, just on a wider input.
    out.maj = majorization_poset(flat.spectra, /*tol=*/1e-12);
    out.lr  = build_lr_poset(out.labels, v_LR);
    out.cs  = build_cs_poset_regular_chain(out.labels);
    return out;
}

CausalComparisonReport
compute_causal_comparison(TDVPConfig const& tdvp_cfg, double v_LR) {
    TDVPConfig cfg = tdvp_cfg;
    cfg.record_spectra = true;     // mandatory for spectra extraction
    cfg.record_poset   = false;    // we build cross-time posets ourselves

    const auto quench = run_qqbar_quench(cfg);
    auto orders = build_causal_orders(quench.snapshots, v_LR);

    CausalComparisonReport report;
    report.n_labels    = static_cast<int>(orders.labels.size());
    report.n_snapshots = static_cast<int>(quench.snapshots.size());
    report.v_LR        = v_LR;
    report.maj_vs_lr   = compare_orders(orders.maj, orders.lr, report.n_labels);
    report.maj_vs_cs   = compare_orders(orders.maj, orders.cs, report.n_labels);
    report.lr_vs_cs    = compare_orders(orders.lr,  orders.cs, report.n_labels);
    return report;
}

} // namespace caset::quantum
