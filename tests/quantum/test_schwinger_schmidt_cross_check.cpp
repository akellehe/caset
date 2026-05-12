// Cross-check Schmidt::of (MPS path) against an independent dense
// Schmidt decomposition for the Schwinger ground state at small N.
//
// The Phase 3 product / GHZ / Bell tests in test_schmidt_spectra.cpp use
// hand-checkable analytic spectra. This file extends coverage to a
// non-trivial physical state — the Schwinger ground state, where the
// spectrum has no closed form and any sign or coefficient bug in the
// MPS-side Schmidt extraction would diverge from the dense reference.
//
// Pipeline:
//   1. Build the dense Schwinger Hamiltonian (SchwingerHamiltonian::denseMatrix).
//   2. Diagonalize and pick the GS eigenvector in the Sz=0 sector
//      (charge-neutral, our DMRG sector).
//   3. For each contiguous bipartition [i, j] | rest, reshape the GS
//      vector into a (sites_in_A) × (sites_in_bar_A) matrix and SVD.
//      The squared singular values are the reference Schmidt spectrum.
//   4. Compare against Schmidt::of applied to a DMRG-optimized
//      MPS at the same parameters; agreement to ~1e-10 is expected
//      (dense ED is exact, DMRG converges to machine precision at this
//      size).

#include "quantum/dmrg_runner.hpp"
#include "quantum/schmidt.hpp"
#include "quantum/schwinger_model.hpp"

#include <itensor/all.h>

#include <Eigen/Dense>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace tessera::quantum;

namespace {

// Dense Schmidt spectrum across an arbitrary bipartition described by a
// bitmask: site n (1-based) is in A iff bit (N - n) of `mask` is set.
// Same site/bit convention used throughout the dense builder.
//
// We don't need to restrict to contiguous A here — the function works on
// any subset — but Phase 3's cut family is contiguous so the caller
// passes contiguous masks.
std::vector<double>
dense_schmidt_spectrum_subset(Eigen::VectorXd const& psi,
                              int N,
                              std::uint64_t mask) {
    const int sizeA = std::popcount(mask);
    const int sizeB = N - sizeA;
    const int dimA = 1 << sizeA;
    const int dimB = 1 << sizeB;

    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(dimA, dimB);
    // Walk all 2^N basis states. For each state s, decompose into the bits
    // belonging to A vs. B in their respective natural orderings (MSB-
    // first within each subsystem) and write the amplitude into M[a, b].
    for (Eigen::Index s = 0; s < psi.size(); ++s) {
        int a = 0, b = 0;
        int pA = sizeA - 1;  // MSB-first within A
        int pB = sizeB - 1;  // MSB-first within B
        for (int n = 1; n <= N; ++n) {
            const int bit_pos = N - n;
            const int bit = (s >> bit_pos) & 1;
            if ((mask >> bit_pos) & 1) {
                a |= bit << pA;  --pA;
            } else {
                b |= bit << pB;  --pB;
            }
        }
        M(a, b) = psi(static_cast<Eigen::Index>(s));
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M);
    auto sigmas = svd.singularValues();
    std::vector<double> spec;
    spec.reserve(static_cast<std::size_t>(sigmas.size()));
    for (Eigen::Index k = 0; k < sigmas.size(); ++k) {
        spec.push_back(sigmas(k) * sigmas(k));
    }
    std::sort(spec.begin(), spec.end(), std::greater<double>{});
    return spec;
}

// Build the bitmask for a contiguous interval [i, j]: bits at positions
// (N-i), (N-i-1), …, (N-j) are set.
inline std::uint64_t contiguous_mask(int N, int i, int j) {
    std::uint64_t m = 0;
    for (int n = i; n <= j; ++n) m |= (1ull << (N - n));
    return m;
}

// Lowest-energy eigenvector of the dense Schwinger H restricted to the
// Sz = 0 (charge-neutral) sector, embedded back into the full 2^N space.
Eigen::VectorXd dense_gs_sz0(SchwingerDense const& sd) {
    const int N = sd.params.N;
    const int half = N / 2;

    std::vector<Eigen::Index> idx;
    for (Eigen::Index s = 0; s < sd.H.rows(); ++s) {
        if (std::popcount(static_cast<std::uint64_t>(s))
            == static_cast<unsigned>(half)) {
            idx.push_back(s);
        }
    }
    const Eigen::Index k = static_cast<Eigen::Index>(idx.size());
    Eigen::MatrixXd Hsub(k, k);
    for (Eigen::Index a = 0; a < k; ++a) {
        for (Eigen::Index b = 0; b < k; ++b) {
            Hsub(a, b) = sd.H(idx[a], idx[b]);
        }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hsub);
    Eigen::VectorXd psi_full = Eigen::VectorXd::Zero(sd.H.rows());
    for (Eigen::Index a = 0; a < k; ++a) {
        psi_full(idx[a]) = es.eigenvectors()(a, 0);
    }
    return psi_full;
}

bool spectra_close(std::vector<double> const& a,
                   std::vector<double> const& b,
                   double tol) {
    // Pad to common length and compare element-by-element after sorting.
    std::vector<double> as = a, bs = b;
    std::sort(as.rbegin(), as.rend());
    std::sort(bs.rbegin(), bs.rend());
    const std::size_t n = std::max(as.size(), bs.size());
    as.resize(n, 0.0);
    bs.resize(n, 0.0);
    for (std::size_t k = 0; k < n; ++k) {
        if (std::abs(as[k] - bs[k]) > tol) return false;
    }
    return true;
}

bool run_case(int N, double m, double L0) {
    SchwingerParams p;
    p.N = N; p.a = 1.0; p.g = 1.0; p.m = m; p.L0 = L0;

    // Dense reference: full ED of the Schwinger H.
    SchwingerHamiltonian H{p};
    auto sd = H.denseMatrix();
    auto psi_dense = dense_gs_sz0(sd);

    // MPS path: DMRG to convergence, then Schmidt::of().
    QuantumConfig cfg;
    cfg.N = N; cfg.a = p.a; cfg.g = p.g; cfg.m = p.m; cfg.L0 = p.L0;
    cfg.maxBondDim = 64;
    cfg.nSweeps = 12;
    cfg.cutoff = 1e-14;

    auto sm = H.mpo(/*conserveQns=*/true);
    auto state = itensor::InitState(sm.sites);
    for (int i = 1; i <= N; ++i) {
        state.set(i, (i % 2 == 1) ? "Up" : "Dn");
    }
    auto psi0 = itensor::MPS(state);
    auto sweeps = itensor::Sweeps(cfg.nSweeps);
    sweeps.maxdim() = 20, 40, 64, 64;
    sweeps.cutoff() = cfg.cutoff;
    sweeps.niter() = 4;
    sweeps.noise() = 1e-7, 1e-8, 0.0;
    auto [E, psi_mps] = itensor::dmrg(sm.H, psi0, sweeps,
                                       itensor::Args("Silent", true));

    // Compare every contiguous-cut spectrum.
    bool ok = true;
    int n_checked = 0;
    constexpr double tol = 1e-9;
    for (int i = 1; i <= N; ++i) {
        for (int j = i; j <= N; ++j) {
            if (i == 1 && j == N) continue;
            const std::uint64_t mask = contiguous_mask(N, i, j);
            const auto ref = dense_schmidt_spectrum_subset(psi_dense, N, mask);
            const auto mps = Schmidt::of(psi_mps, i, j);
            if (!spectra_close(ref, mps, tol)) {
                ok = false;
                std::cout << "  FAIL [" << i << "," << j << "]"
                          << "  dense=";
                for (double x : ref) std::cout << x << " ";
                std::cout << " mps=";
                for (double x : mps) std::cout << x << " ";
                std::cout << "\n";
            }
            ++n_checked;
        }
    }
    std::cout << "  N=" << N << " m=" << m << " L0=" << L0
              << "  cuts=" << n_checked
              << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

} // namespace

int main() {
    std::cout << "Schwinger ground-state Schmidt spectra: MPS vs dense ED\n";
    std::cout << "-------------------------------------------------------\n";
    bool ok = true;
    // Cover the same parameter range as the Phase 1 acceptance test, so
    // any disagreement here also localises which (m, L0) breaks.
    ok &= run_case(/*N=*/4, /*m=*/0.0,   /*L0=*/0.0);
    ok &= run_case(/*N=*/4, /*m=*/0.125, /*L0=*/0.0);
    ok &= run_case(/*N=*/4, /*m=*/0.25,  /*L0=*/0.0);
    ok &= run_case(/*N=*/4, /*m=*/0.0,   /*L0=*/0.5);
    ok &= run_case(/*N=*/6, /*m=*/0.0,   /*L0=*/0.0);
    ok &= run_case(/*N=*/6, /*m=*/0.125, /*L0=*/0.0);
    ok &= run_case(/*N=*/6, /*m=*/0.25,  /*L0=*/0.5);
    ok &= run_case(/*N=*/8, /*m=*/0.0,   /*L0=*/0.0);
    std::cout << (ok ? "\nALL PASS\n" : "\nFAILURES\n");
    return ok ? 0 : 1;
}
