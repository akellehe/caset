// Majorization predicate + transitive reduction for the majorization
// poset. See include/quantum/majorization.hpp for the formal definitions
// and the reference to Nielsen 1999.

#include "quantum/majorization.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace tessera::quantum {

namespace {

// Sort `v` non-increasingly and zero-pad to length `n`. Returns a fresh
// vector so the caller's input is unmodified.
std::vector<double> sorted_padded(std::vector<double> const& v,
                                  std::size_t n) {
    std::vector<double> out = v;
    std::sort(out.begin(), out.end(), std::greater<double>{});
    if (out.size() < n) out.resize(n, 0.0);
    return out;
}

} // namespace

bool majorizes(std::vector<double> const& mu,
               std::vector<double> const& lambda,
               double tol) {
    const std::size_t n = std::max(mu.size(), lambda.size());
    if (n == 0) return true;  // both empty: trivially majorizes

    auto mus  = sorted_padded(mu,     n);
    auto lams = sorted_padded(lambda, n);

    // Walk the partial sums in lockstep: at every k we need sum(mus[0..k])
    // ≥ sum(lams[0..k]) — i.e. μ collects at least as much mass on the
    // "biggest" k coordinates as λ does.
    double sum_mu = 0.0, sum_la = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
        sum_mu += mus[k];
        sum_la += lams[k];
        if (sum_mu + tol < sum_la) return false;
    }
    // Total-mass equality is required for the textbook majorization
    // statement; it's automatic for probability distributions but we don't
    // assume the caller normalised.
    return std::abs(sum_mu - sum_la) <= tol;
}

bool strictlyMajorizes(std::vector<double> const& mu,
                        std::vector<double> const& lambda,
                        double tol) {
    return majorizes(mu, lambda, tol) && !majorizes(lambda, mu, tol);
}

Poset majorizationPoset(std::vector<std::vector<double>> const& spectra,
                         double tol) {
    const int N = static_cast<int>(spectra.size());
    Poset out(N);
    if (N == 0) return out;

    // Pre-compute the strict-majorization adjacency. We pay O(N^2) majorizes
    // calls here so the transitive-reduction loop below is O(N^3) bool ops
    // rather than O(N^3) sort+sum ops.
    std::vector<std::vector<char>> strict(N, std::vector<char>(N, 0));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            if (strictlyMajorizes(spectra[i], spectra[j], tol)) {
                strict[i][j] = 1;
            }
        }
    }

    // Transitive reduction: an edge (i, j) in the strict graph survives
    // iff there is no third node k with i ≻ k ≻ j.
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (!strict[i][j]) continue;
            bool has_intermediate = false;
            for (int k = 0; k < N; ++k) {
                if (k == i || k == j) continue;
                if (strict[i][k] && strict[k][j]) {
                    has_intermediate = true;
                    break;
                }
            }
            if (!has_intermediate) {
                out.addCover(i, j);
            }
        }
    }
    return out;
}

} // namespace tessera::quantum
