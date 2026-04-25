// Majorization partial order on probability distributions, plus the Hasse-
// diagram (transitive-reduction) construction we use to build the
// majorization poset of Schmidt spectra of an MPS (Phase 3 of
// docs/source/quantum-plan.md).
//
// ─── Majorization recap ───────────────────────────────────────────────────
//
// Given two finite, non-negative sequences μ and λ, both normalized to the
// same total mass, μ *majorizes* λ (written μ ≻ λ or λ ≼ μ) iff
//
//   sum_{i=1..k} μ_i^↓  ≥  sum_{i=1..k} λ_i^↓     for every k = 1, 2, …
//
// where x_i^↓ denotes the entries of x sorted non-increasingly, with the
// shorter vector zero-padded to the longer one's length. Intuitively, μ is
// "more concentrated" than λ. For probability distributions the total-mass
// condition is automatic (both sums are 1).
//
// The relation is
//   • reflexive   — μ ≼ μ trivially;
//   • antisymmetric (modulo permutation/padding) — μ ≼ λ and λ ≼ μ iff
//     μ↓ = λ↓ as zero-padded vectors;
//   • transitive  — μ ≼ ν and ν ≼ λ ⇒ μ ≼ λ.
// So ≼ is a partial order on equivalence classes of zero-padded sorted
// vectors.
//
// In our application the vectors are Schmidt spectra of the same MPS at
// different (interval, time) labels (see PLAN.md §5 Phase 3 and the
// methodology page docs/source/quantum-methodology.md). Nielsen's theorem
// {cite}`Nielsen1999LOCC` gives the LOCC-conversion interpretation:
// ⟨α| can be deterministically converted to ⟨β| iff the Schmidt spectrum of
// ⟨α| is majorized by that of ⟨β|.

#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace caset::quantum {

// Returns true iff μ majorizes λ. Both vectors are sorted non-increasingly
// internally; the shorter is zero-padded to the longer's length.
//
// `tol` slack is applied to each partial-sum comparison and to the total-mass
// equality check at the end. 1e-12 is well below double-precision noise on
// any spectrum we manipulate; tighten only if the caller expects exact
// integer-like values.
bool majorizes(std::vector<double> const& mu,
               std::vector<double> const& lambda,
               double tol = 1e-12);

// μ strictly majorizes λ iff μ ≻ λ but not λ ≻ μ — i.e. the relation is
// proper and not just an equality of sorted-padded vectors.
bool strictly_majorizes(std::vector<double> const& mu,
                        std::vector<double> const& lambda,
                        double tol = 1e-12);

// Hasse / cover representation of a partial order on a finite set of nodes.
//
// Nodes are integers 0 .. n_nodes-1. `covers` lists pairs (a, b) where a
// strictly majorizes b AND no third node c sits between them — so this is
// the transitive reduction of the strict-majorization graph.
//
// Equality classes (μ ≼ λ AND λ ≼ μ) are NOT given cover edges among
// themselves: members of the same equivalence class share a single
// "position" in the partial order.
struct Poset {
    int n_nodes{0};
    std::vector<std::pair<int, int>> covers;  // (a, b) with a ≻ b and no
                                              // intermediate node
};

// Construct the majorization poset on the given list of spectra.
//
// `spectra[k]` becomes node k. The resulting Poset stores Hasse cover
// edges only — the transitive closure is implicit.
//
// Complexity: O(M^3) for M = spectra.size(), dominated by the transitive-
// reduction pass. Each pair-of-spectra majorizes() call is O(L log L) on
// the spectrum lengths L; for our use case L ≤ MPS bond dimension squared.
Poset majorization_poset(std::vector<std::vector<double>> const& spectra,
                         double tol = 1e-12);

} // namespace caset::quantum
