// Concrete implementations of the `MajorizationPredicate` hierarchy
// plus the variant-agnostic Hasse-poset construction. See
// include/quantum/majorization.hpp for the abstract contract, the
// partial-order axioms each variant satisfies, and the bibliographic
// references — we keep the references short here and point back to the
// header rather than duplicating them.
//
// References used in this file: {N1999} = Nielsen, M. A. (1999), Phys.
// Rev. Lett. 83, 436 (arXiv: quant-ph/9811053); {B2015} = Brändén, P.
// (2015), arXiv: 1410.6601; {AN2008} = Aubrun, G. & Nechita, I. (2008),
// arXiv: 0707.0211. Full bibliographic data is in the header.

#include "quantum/majorization.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace tessera::quantum {

namespace {

// Sort `v` non-increasingly and zero-pad to length `n`. Returns a fresh
// vector so the caller's input is unmodified. Used by every variant
// that needs the canonical descending-sort-then-pad form before
// applying its own pointwise inequalities.
std::vector<double> sortedPadded(std::vector<double> const& v,
                                  std::size_t n) {
    std::vector<double> out = v;
    std::sort(out.begin(), out.end(), std::greater<double>{});
    if (out.size() < n) out.resize(n, 0.0);
    return out;
}

// Sort descending and strip trailing zeros (entries below `tol`). Used
// by the log-concavity check, which is only meaningful on the support.
std::vector<double> sortedTrimmed(std::vector<double> const& v, double tol) {
    std::vector<double> out = v;
    std::sort(out.begin(), out.end(), std::greater<double>{});
    while (!out.empty() && out.back() <= tol) out.pop_back();
    return out;
}

} // namespace

// ─── StandardMajorization ────────────────────────────────────────────────
//
// Implements {N1999} eq. (1) directly: walk the partial sums of the
// descending-sorted spectra in lockstep and accept only if the running
// sum of μ never dips below that of λ.

StandardMajorization::StandardMajorization(double tol) noexcept
    : tol_(tol) {}

bool StandardMajorization::majorizes(std::vector<double> const& mu,
                                       std::vector<double> const& lambda) const {
    const std::size_t n = std::max(mu.size(), lambda.size());
    if (n == 0) return true;  // both empty: trivially majorizes

    auto mus  = sortedPadded(mu,     n);
    auto lams = sortedPadded(lambda, n);

    // {N1999} eq. (1): at every k we need
    //     sum(mus[0..k])  ≥  sum(lams[0..k])
    // with equality at k = n (the total-mass condition).
    double sumMu = 0.0;
    double sumLa = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
        sumMu += mus[k];
        sumLa += lams[k];
        if (sumMu + tol_ < sumLa) return false;
    }
    // Total-mass equality is required for the textbook majorization
    // statement; it's automatic for probability distributions but we
    // don't assume the caller normalised.
    return std::abs(sumMu - sumLa) <= tol_;
}

std::string StandardMajorization::name() const {
    return "standard";
}

// ─── LogConcaveMajorization ──────────────────────────────────────────────
//
// {B2015} §1 defines a non-negative sequence (a_0, a_1, …) to be
// log-concave iff a_i² ≥ a_{i-1} · a_{i+1} for every interior i. We
// gate {N1999}-style majorization on both spectra satisfying this
// shape constraint.

LogConcaveMajorization::LogConcaveMajorization(double tol) noexcept
    : StandardMajorization(tol) {}

bool LogConcaveMajorization::majorizes(std::vector<double> const& mu,
                                         std::vector<double> const& lambda) const {
    if (!isLogConcave(mu, tol_) || !isLogConcave(lambda, tol_)) return false;
    return StandardMajorization::majorizes(mu, lambda);
}

std::string LogConcaveMajorization::name() const {
    return "log-concave";
}

bool LogConcaveMajorization::isLogConcave(std::vector<double> const& v,
                                            double tol) {
    auto s = sortedTrimmed(v, tol);
    if (s.size() < 3) return true;  // length 0/1/2: trivially log-concave
    // {B2015} §1: s_i² ≥ s_{i-1} · s_{i+1}, with `tol` slack on the right.
    for (std::size_t i = 1; i + 1 < s.size(); ++i) {
        if (s[i] * s[i] + tol < s[i - 1] * s[i + 1]) return false;
    }
    return true;
}

// ─── PeakRadialMajorization ──────────────────────────────────────────────
//
// μ ≻ λ iff, after sorting both descending and zero-padding,
//     λᵢ · μ₁  ≤  μᵢ · λ₁     for every i,
// which is the cross-multiplied form of `λᵢ / λ₁ ≤ μᵢ / μ₁`. The
// cross-multiplication keeps the comparison numerically clean for
// spectra with very small peaks (e.g. near-product states with
// λ₁ ≈ 1 - ε).
//
// This is structurally analogous to the L^p-norm dominance in
// {AN2008} Theorem 1.1 / Prop. 2.5 (which characterises *catalytic*
// majorization — the *weakening* of the classical order); here we
// instead use entrywise relative-to-peak dominance, which yields a
// *strengthening* of the classical order.

PeakRadialMajorization::PeakRadialMajorization(double tol) noexcept
    : tol_(tol) {}

bool PeakRadialMajorization::majorizes(std::vector<double> const& mu,
                                         std::vector<double> const& lambda) const {
    const std::size_t n = std::max(mu.size(), lambda.size());
    if (n == 0) return true;

    auto mus  = sortedPadded(mu,     n);
    auto lams = sortedPadded(lambda, n);

    const double peakMu = mus[0];
    const double peakLa = lams[0];

    // Degenerate-peak handling. If μ has zero peak it has all-zero
    // mass and majorizes only an equally-zero spectrum. If λ has zero
    // peak, it is "trivially decayed" and any μ peak-radial-majorizes
    // it.
    if (peakMu <= tol_) return peakLa <= tol_;
    if (peakLa <= tol_) return true;

    // Pointwise relative-to-peak dominance (cross-multiplied form):
    //     μᵢ / μ₁  ≤  λᵢ / λ₁     for every i,
    // i.e. μ has SMALLER non-peak entries (relative to its peak) than
    // λ does — μ is at least as peaked as λ. Cross-multiplying:
    //     μᵢ · λ₁  ≤  λᵢ · μ₁,
    // equivalently the test `μᵢ · λ₁ > λᵢ · μ₁` rules μ out.
    for (std::size_t i = 0; i < n; ++i) {
        if (mus[i] * peakLa > lams[i] * peakMu + tol_) return false;
    }
    return true;
}

std::string PeakRadialMajorization::name() const {
    return "peak-radial";
}

// ─── Variant-agnostic Hasse-poset construction ───────────────────────────
//
// Given any `MajorizationPredicate`, construct the Hasse diagram of
// the strict relation. The construction is identical to {N1999}'s
// classical case — it depends on the partial-order axioms in the
// `MajorizationPredicate` contract, not on the specific predicate's
// definition.

Poset majorizationPoset(std::vector<std::vector<double>> const& spectra,
                         MajorizationPredicate const& predicate) {
    const int N = static_cast<int>(spectra.size());
    Poset out(N);
    if (N == 0) return out;

    // Pre-compute the strict adjacency. We pay O(N²) predicate calls
    // here so the transitive-reduction loop below is O(N³) bool ops
    // rather than O(N³) sort+compare ops.
    std::vector<std::vector<char>> strict(
        static_cast<std::size_t>(N),
        std::vector<char>(static_cast<std::size_t>(N), 0));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            if (predicate.strictlyMajorizes(
                    spectra[static_cast<std::size_t>(i)],
                    spectra[static_cast<std::size_t>(j)])) {
                strict[static_cast<std::size_t>(i)]
                      [static_cast<std::size_t>(j)] = 1;
            }
        }
    }

    // Transitive reduction: an edge (i, j) in the strict graph survives
    // iff there is no third node k with i ≻ k ≻ j. Correctness of this
    // reduction depends on transitivity of the underlying predicate,
    // which the `MajorizationPredicate` contract requires.
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (!strict[static_cast<std::size_t>(i)]
                       [static_cast<std::size_t>(j)]) continue;
            bool hasIntermediate = false;
            for (int k = 0; k < N; ++k) {
                if (k == i || k == j) continue;
                if (strict[static_cast<std::size_t>(i)]
                          [static_cast<std::size_t>(k)] &&
                    strict[static_cast<std::size_t>(k)]
                          [static_cast<std::size_t>(j)]) {
                    hasIntermediate = true;
                    break;
                }
            }
            if (!hasIntermediate) {
                out.addCover(i, j);
            }
        }
    }
    return out;
}

// ─── Backward-compatible free-function wrappers ─────────────────────────

bool majorizes(std::vector<double> const& mu,
               std::vector<double> const& lambda,
               double tol) {
    return StandardMajorization{tol}.majorizes(mu, lambda);
}

bool strictlyMajorizes(std::vector<double> const& mu,
                        std::vector<double> const& lambda,
                        double tol) {
    return StandardMajorization{tol}.strictlyMajorizes(mu, lambda);
}

Poset majorizationPoset(std::vector<std::vector<double>> const& spectra,
                         double tol) {
    return majorizationPoset(spectra, StandardMajorization{tol});
}

} // namespace tessera::quantum
