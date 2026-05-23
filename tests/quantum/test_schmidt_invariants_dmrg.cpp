// Mathematical invariants on Schmidt spectra of Schwinger-model
// DMRG ground states. Complements:
//
//   • test_schmidt_spectra.cpp           — spectra of hand-built MPSes
//                                          (product, GHZ, Bell)
//   • test_schwinger_schmidt_cross_check — DMRG GS Schmidt vs. dense ED
//                                          on small N (numeric agreement)
//   • test_phase3_majorization_python    — DMRG GS Σλ = 1 invariant at
//                                          one parameter point (N=6, m=0)
//
// Where this fits: the existing tests verify NUMERICAL CORRECTNESS at
// specific points but don't assert the universal mathematical invariants
// that any reduced-density-matrix spectrum must satisfy across the full
// parameter sweep. Those invariants are:
//
//   (P1)  Σ_α λ_α = 1                    (probability normalization)
//   (P2)  λ_α ∈ [0, 1] for all α          (probability + bounded)
//   (P3)  λ_α monotone non-increasing     (sort convention; the public
//                                          API documents descending order
//                                          via Schmidt::allOf)
//   (P4)  rank ≤ 2^min(w, N-w)            (Schmidt rank bound for a cut
//                                          of width w on a chain of N
//                                          spin-½ sites)
//
// These hold for every contiguous bipartition of any pure-state MPS,
// regardless of the Hamiltonian or DMRG sweep schedule. A failure means
// either Schmidt::of() or DMRG itself produced an unphysical state.
//
// We sweep a small parameter grid (N ∈ {6, 8}, m/g ∈ {0, 0.25, 1.0},
// L₀ ∈ {0, 0.5}) — 12 ground states × O(N²) cuts each = O(100) spectra
// validated. Total runtime ~5–10 s, dominated by DMRG.

#include "quantum/DMRGRunner.hpp"
#include "quantum/Schmidt.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

using namespace tessera::quantum;

namespace {

struct InvariantStats {
    int n_spectra{0};
    int n_failed_normalization{0};
    int n_failed_bounds{0};
    int n_failed_monotone{0};
    int n_failed_rank{0};
    double max_norm_error{0.0};
    double min_lambda{0.0};
    double max_lambda{0.0};
};

bool spectrum_invariants(std::vector<double> const& spec,
                         int N, int i, int j,
                         InvariantStats& out,
                         double tol = 1e-9) {
    out.n_spectra++;

    // (P1) Σ λ = 1.
    const double total = std::accumulate(spec.begin(), spec.end(), 0.0);
    const double norm_err = std::abs(total - 1.0);
    if (norm_err > out.max_norm_error) out.max_norm_error = norm_err;
    bool ok = true;
    if (norm_err > tol) {
        ++out.n_failed_normalization;
        ok = false;
    }

    // (P2) λ_α ∈ [0, 1]. The lower bound -tol allows for tiny negative
    // numerical noise from SVD; the upper bound is hard at 1 + tol.
    for (double lam : spec) {
        if (lam < out.min_lambda) out.min_lambda = lam;
        if (lam > out.max_lambda) out.max_lambda = lam;
        if (lam < -tol || lam > 1.0 + tol) {
            ++out.n_failed_bounds;
            ok = false;
            break;
        }
    }

    // (P3) Monotone non-increasing.
    for (std::size_t k = 1; k < spec.size(); ++k) {
        if (spec[k] > spec[k - 1] + tol) {
            ++out.n_failed_monotone;
            ok = false;
            break;
        }
    }

    // (P4) rank bound: cut [i, j] on N qubits has width w = j-i+1,
    // so the reduced-density-matrix rank is at most 2^min(w, N-w).
    // Past that index the spectrum should be (numerically) zero.
    const int w = j - i + 1;
    const int max_rank_log2 = std::min(w, N - w);
    const std::size_t max_rank = (max_rank_log2 < 30)
        ? (std::size_t{1} << max_rank_log2)
        : spec.size();
    for (std::size_t k = max_rank; k < spec.size(); ++k) {
        if (std::abs(spec[k]) > tol) {
            ++out.n_failed_rank;
            ok = false;
            break;
        }
    }
    return ok;
}

bool sweep_one_config(QuantumConfig const& cfg) {
    auto r = SchwingerModel{cfg}.solveWithMajorization(/*tol=*/1e-12);
    InvariantStats stats;
    bool ok = true;
    for (std::size_t k = 0; k < r.spectra.spectra.size(); ++k) {
        const auto& spec = r.spectra.spectra[k];
        const auto& iv   = r.spectra.intervals[k];
        if (!spectrum_invariants(spec, cfg.N, iv.i, iv.j, stats)) {
            ok = false;
        }
    }
    std::cout
        << "  N=" << cfg.N
        << " m=" << cfg.m
        << " L0=" << cfg.L0
        << "  spectra=" << stats.n_spectra
        << "  Σλ_max_err=" << stats.max_norm_error
        << "  λ_range=[" << stats.min_lambda << ", " << stats.max_lambda << "]"
        << "  fails=(norm:" << stats.n_failed_normalization
        << " bounds:" << stats.n_failed_bounds
        << " monotone:" << stats.n_failed_monotone
        << " rank:" << stats.n_failed_rank << ")"
        << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

QuantumConfig make_config(int N, double m, double L0) {
    QuantumConfig c;
    c.N = N; c.a = 1.0; c.g = 1.0; c.m = m; c.L0 = L0;
    c.maxBondDim = 32;
    c.nSweeps = 8;
    c.cutoff = 1e-12;
    c.krylovDim = 4;
    c.quiet = true;
    c.conserveQns = true;
    return c;
}

} // namespace

int main() {
    std::cout
        << "Schmidt-spectrum mathematical invariants on Schwinger DMRG GS\n"
           "(P1) Σλ=1   (P2) λ∈[0,1]   (P3) descending   (P4) rank ≤ 2^min(w,N-w)\n"
           "--------------------------------------------------------------\n";

    bool ok = true;
    // Parameter grid: representative slice covering massless / intermediate /
    // heavy mass regimes and a non-zero background field.
    for (int N : {6, 8}) {
        for (double m : {0.0, 0.25, 1.0}) {
            for (double L0 : {0.0, 0.5}) {
                if (!sweep_one_config(make_config(N, m, L0))) ok = false;
            }
        }
    }
    std::cout << (ok ? "\nALL PASS\n" : "\nFAILURES\n");
    return ok ? 0 : 1;
}
