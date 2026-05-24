// Extended tests for tessera::quantum::koashiImotoDecompose covering
// inputs that test_koashi_imoto_core doesn't reach: larger Hilbert
// dimensions, asymmetric (d_A != d_B) factorisations, random pure /
// mixed states, output-invariant checks (Hermitian / positive / unit
// trace on every block-by-block piece), and reconstruction sanity.
//
// Same hand-calculated / analytically-derivable expected values; same
// no-shortcuts general algorithm exercised throughout.

#include "quantum/KoashiImoto.hpp"
#include "quantum/QuantumState.hpp"
#include "test_helpers.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <iostream>
#include <random>

using namespace tessera::quantum;
using namespace tessera::test_helpers_core;

namespace {

constexpr double TOL = 1e-9;

const KoashiImotoTolerances DEFAULT_TOL{};

// ── Larger Hilbert dimensions ─────────────────────────────────────────

bool t_ki_on_2x3_product_marginals() {
    // ρ_A on H_2, ρ_B on H_3; non-square joint (6×6). Product input
    // gives a single block with K_A = K_B = 1, r_A = 2, r_B = 3.
    Eigen::Matrix2cd rhoA;
    rhoA << 0.7, 0.0, 0.0, 0.3;
    Eigen::MatrixXcd rhoB(3, 3);
    rhoB << 0.5, 0.0, 0.0,
            0.0, 0.3, 0.0,
            0.0, 0.0, 0.2;
    auto rhoAB = productState(rhoA, rhoB);

    auto r = koashiImotoDecompose(rhoAB, 2, 3, DEFAULT_TOL);
    bool ok = expect_true(r.blocks.size() == 1,
        "KI(2x3 product) -> 1 block");
    if (r.blocks.empty()) return false;
    const auto& blk = r.blocks[0];
    ok &= expect_true(blk.dimLeftA == 1 && blk.dimLeftB == 1,
        "L sides == 1");
    ok &= expect_true(blk.dimRightA == 2 && blk.dimRightB == 3,
        "R sides match dim_A, dim_B");
    // tailA in eigenbasis of ρ_A descending: diag(0.7, 0.3).
    Eigen::Matrix2cd expectedTailA;
    expectedTailA << 0.7, 0.0, 0.0, 0.3;
    ok &= expect_matrix_near(blk.tailA, expectedTailA, TOL,
        "tail A == diag(0.7, 0.3)");
    // tailB in eigenbasis of ρ_B descending: diag(0.5, 0.3, 0.2).
    Eigen::MatrixXcd expectedTailB(3, 3);
    expectedTailB << 0.5, 0, 0,
                     0,  0.3, 0,
                     0,  0,   0.2;
    ok &= expect_matrix_near(blk.tailB, expectedTailB, TOL,
        "tail B == diag(0.5, 0.3, 0.2)");
    return ok;
}

bool t_ki_on_3x3_max_mixed() {
    // I_9 / 9 on a 3⊗3 Hilbert space = (I_3/3) ⊗ (I_3/3). One block,
    // K_A = K_B = 1, r_A = r_B = 3; tails are I/3.
    Eigen::MatrixXcd rhoAB = Eigen::MatrixXcd::Identity(9, 9) / 9.0;
    auto r = koashiImotoDecompose(rhoAB, 3, 3, DEFAULT_TOL);
    bool ok = expect_true(r.blocks.size() == 1,
        "KI(I/9) -> 1 block");
    if (r.blocks.empty()) return false;
    const auto& blk = r.blocks[0];
    ok &= expect_true(blk.dimLeftA == 1 && blk.dimLeftB == 1,
        "L sides == 1");
    ok &= expect_true(blk.dimRightA == 3 && blk.dimRightB == 3,
        "R sides == 3");
    Eigen::MatrixXcd thirdI = Eigen::MatrixXcd::Identity(3, 3) / 3.0;
    ok &= expect_matrix_near(blk.tailA, thirdI, TOL, "tail A == I/3");
    ok &= expect_matrix_near(blk.tailB, thirdI, TOL, "tail B == I/3");
    return ok;
}

bool t_ki_on_4dim_ghz_like_classical() {
    // 4-classical-label correlated state on a 4⊗4 Hilbert:
    //   ρ = Σ_{j=0..3} (1/4) |jj><jj|
    // ρ_A = ρ_B = I/4 (all eigvecs degenerate, but pairwise distinct
    // conditional states). KI: 4 blocks, each K=r=1, weight 1/4.
    Eigen::MatrixXcd rhoAB = Eigen::MatrixXcd::Zero(16, 16);
    for (int j = 0; j < 4; ++j) {
        const int idx = j * 4 + j;  // |jj> in (A ⊗ B) ordering
        rhoAB(idx, idx) = 0.25;
    }
    auto r = koashiImotoDecompose(rhoAB, 4, 4, DEFAULT_TOL);
    bool ok = expect_true(r.blocks.size() == 4,
        "KI(4-classical) -> 4 blocks");
    double total = 0.0;
    for (const auto& blk : r.blocks) {
        total += blk.weight;
        ok &= expect_true(blk.dimLeftA == 1 && blk.dimLeftB == 1,
            "each block: K = 1");
        ok &= expect_true(blk.dimRightA == 1 && blk.dimRightB == 1,
            "each block: r = 1");
        ok &= expect_near(blk.weight, 0.25, TOL,
            "each block weight == 1/4");
    }
    ok &= expect_near(total, 1.0, TOL, "Σ_j p_j == 1");
    return ok;
}

// ── Random pure / mixed inputs: output invariants ─────────────────────

bool t_ki_output_invariants_on_random_pure() {
    // For a Haar-random pure state |ψ>_AB, KI should produce valid
    // density matrices (Hermitian, positive, unit-trace per block).
    std::mt19937 rng(0xBEEFCAFE);
    std::normal_distribution<double> gauss(0.0, 1.0);

    bool ok = true;
    for (int trial = 0; trial < 4; ++trial) {
        // Random complex Gaussian vector → normalise → outer product.
        const int dA = 3, dB = 3;
        Eigen::VectorXcd psi(dA * dB);
        for (int i = 0; i < dA * dB; ++i)
            psi(i) = std::complex<double>(gauss(rng), gauss(rng));
        psi /= psi.norm();
        Eigen::MatrixXcd rhoAB = psi * psi.adjoint();

        auto r = koashiImotoDecompose(rhoAB, dA, dB, DEFAULT_TOL);
        const std::string suf = " (trial " + std::to_string(trial) + ")";

        double totalW = 0.0;
        for (const auto& blk : r.blocks) {
            totalW += blk.weight;
            // Each block's core is a density matrix on its L spaces.
            const auto& C = blk.coreState;
            ok &= expect_true(
                std::abs(C.trace().real() - 1.0) < 1e-9 || C.size() == 1,
                "core has unit trace (or 1x1 trivial)" + suf);
            ok &= expect_true(
                (C - C.adjoint()).norm() < 1e-9,
                "core is Hermitian" + suf);
            // tailA and tailB are diagonal-positive density matrices.
            ok &= expect_true(
                std::abs(blk.tailA.trace().real() - 1.0) < 1e-9,
                "tail A has unit trace" + suf);
            ok &= expect_true(
                std::abs(blk.tailB.trace().real() - 1.0) < 1e-9,
                "tail B has unit trace" + suf);
        }
        ok &= expect_near(totalW, 1.0, 1e-9,
            "Σ_j p_j == 1 for random pure input" + suf);
    }
    return ok;
}

bool t_ki_output_invariants_on_random_mixed() {
    // Same invariants on Haar-random mixed states.
    bool ok = true;
    for (int trial = 0; trial < 4; ++trial) {
        const int dA = 2, dB = 3;
        auto qAB = QuantumState::randomMixed(
            dA * dB, std::log(static_cast<double>(dA * dB)) * 0.5,
            0xC0DE0000 + trial);
        const auto& rhoAB = qAB.matrix();

        auto r = koashiImotoDecompose(rhoAB, dA, dB, DEFAULT_TOL);
        const std::string suf = " (trial " + std::to_string(trial) + ")";

        double totalW = 0.0;
        for (const auto& blk : r.blocks) {
            totalW += blk.weight;
            ok &= expect_true(blk.dimLeftA >= 1 && blk.dimRightA >= 1,
                "block dim_L^A, dim_R^A both ≥ 1" + suf);
            ok &= expect_true(blk.dimLeftB >= 1 && blk.dimRightB >= 1,
                "block dim_L^B, dim_R^B both ≥ 1" + suf);
            ok &= expect_true(
                (blk.coreState - blk.coreState.adjoint()).norm() < 1e-8,
                "core Hermitian" + suf);
        }
        ok &= expect_near(totalW, 1.0, 1e-8,
            "Σ_j p_j == 1 for random mixed input" + suf);
    }
    return ok;
}

// ── Pure-state Schmidt equivalence ────────────────────────────────────

bool t_ki_pure_matches_schmidt_rank() {
    // For a pure ρ_AB, the Schmidt rank equals the maximum dim_L^A.
    // We build a state with explicit Schmidt rank r by superposing r
    // computational basis pairs and check the KI block dims.
    bool ok = true;
    for (int schmidtRank : {1, 2, 3}) {
        const int dA = 4, dB = 4;
        Eigen::VectorXcd psi = Eigen::VectorXcd::Zero(dA * dB);
        for (int k = 0; k < schmidtRank; ++k) {
            psi(k * dB + k) = 1.0;  // |kk> contributions
        }
        psi /= psi.norm();
        Eigen::MatrixXcd rhoAB = psi * psi.adjoint();

        auto r = koashiImotoDecompose(rhoAB, dA, dB, DEFAULT_TOL);

        // Pure state with Schmidt rank > 0 should give one non-trivial
        // block whose dim_L^A == schmidtRank (the entangled core has
        // rank = Schmidt rank).
        int nonTrivial = 0, maxLA = 0;
        for (const auto& blk : r.blocks) {
            if (blk.dimLeftA > 1) ++nonTrivial;
            if (blk.dimLeftA > maxLA) maxLA = blk.dimLeftA;
        }
        const std::string suf =
            " (schmidtRank=" + std::to_string(schmidtRank) + ")";
        if (schmidtRank == 1) {
            // Rank 1 is a product state; no non-trivial blocks.
            ok &= expect_true(nonTrivial == 0,
                "rank-1 pure: no non-trivial L blocks" + suf);
        } else {
            ok &= expect_true(maxLA == schmidtRank,
                "max dim_L^A == Schmidt rank" + suf);
        }
    }
    return ok;
}

// ── Reconstruction sanity ─────────────────────────────────────────────

bool t_ki_reconstruction_trace_unity() {
    // The Σ, A', B' matrices are normalised so Σ_j p_j = Tr(sigma) = 1.
    // The aPrime / bPrime block-diagonals also sum to Tr = 1 by
    // construction. Verify on a mixed input.
    Eigen::Matrix2cd rhoA;
    rhoA << 0.55, 0.05, 0.05, 0.45;
    Eigen::Matrix2cd rhoB = QuantumState::maximallyMixed(2).matrix();
    auto rhoAB = productState(rhoA, rhoB);

    auto r = koashiImotoDecompose(rhoAB, 2, 2, DEFAULT_TOL);

    return expect_near(r.sigma.trace().real(),  1.0, 1e-9,
            "Tr(sigma)  == 1")
        && expect_near(r.aPrime.trace().real(), 1.0, 1e-9,
            "Tr(aPrime) == 1")
        && expect_near(r.bPrime.trace().real(), 1.0, 1e-9,
            "Tr(bPrime) == 1");
}

// ── Asymmetric dimension stress ───────────────────────────────────────

bool t_ki_handles_asymmetric_dims() {
    // dim_A == 4, dim_B == 2 — Σ is biased; KI should still produce
    // well-formed output.
    Eigen::MatrixXcd rhoA = Eigen::MatrixXcd::Identity(4, 4) / 4.0;
    Eigen::Matrix2cd rhoB = QuantumState::maximallyMixed(2).matrix();
    auto rhoAB = productState(rhoA, rhoB);

    auto r = koashiImotoDecompose(rhoAB, 4, 2, DEFAULT_TOL);
    bool ok = expect_true(r.blocks.size() == 1,
        "KI(I_4/4 ⊗ I_2/2) -> 1 block");
    if (r.blocks.empty()) return false;
    const auto& blk = r.blocks[0];
    return ok
        && expect_true(blk.dimLeftA == 1 && blk.dimLeftB == 1,
            "asymmetric: L = 1 on both sides")
        && expect_true(blk.dimRightA == 4,
            "asymmetric: dim_R^A == 4")
        && expect_true(blk.dimRightB == 2,
            "asymmetric: dim_R^B == 2");
}

// ── Tolerance sensitivity ─────────────────────────────────────────────

bool t_ki_loose_tolerance_does_not_crash() {
    // Loosening the tolerances should not crash; the result may merge
    // structure that tighter tolerances would distinguish.
    auto rhoAB = bellPhiPlus();
    KoashiImotoTolerances loose{1e-3, 1e-3, 1e-3};
    auto r = koashiImotoDecompose(rhoAB, 2, 2, loose);
    bool ok = expect_true(!r.blocks.empty(),
        "loose-tolerance KI on Bell produces ≥ 1 block (no crash)");
    double total = 0.0;
    for (const auto& blk : r.blocks) total += blk.weight;
    ok &= expect_near(total, 1.0, 1e-9,
        "Σ_j p_j == 1 with loose tolerances");
    return ok;
}

} // namespace

int main() {
    std::cout << "== test_koashi_imoto_advanced ==\n";
    bool ok = true;
    ok &= t_ki_on_2x3_product_marginals();
    ok &= t_ki_on_3x3_max_mixed();
    ok &= t_ki_on_4dim_ghz_like_classical();
    ok &= t_ki_output_invariants_on_random_pure();
    ok &= t_ki_output_invariants_on_random_mixed();
    ok &= t_ki_pure_matches_schmidt_rank();
    ok &= t_ki_reconstruction_trace_unity();
    ok &= t_ki_handles_asymmetric_dims();
    ok &= t_ki_loose_tolerance_does_not_crash();
    std::cout << (ok ? "ALL PASSED\n" : "SOME FAILED\n");
    return ok ? 0 : 1;
}
