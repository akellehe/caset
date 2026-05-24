// Tests for tessera::quantum::KoashiImoto and the partial-trace /
// mutual-information helpers it depends on.
//
// Every expected value is hand-calculated and quoted in a comment
// alongside the assertion. The reference cases — Bell, classically
// correlated, pure product, mixed product, maximally mixed, and a
// constructed K_A = K_B = 2, R_A = R_B = 2 block — cover all four
// canonical KI "shapes" plus one mixed L/R case to exercise the
// unified algorithm at every limit.

#include "quantum/KoashiImoto.hpp"
#include "test_helpers.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <iostream>

using namespace tessera::quantum;
using namespace tessera::test_helpers_core;

namespace {

constexpr double TOL = 1e-10;

const KoashiImotoTolerances DEFAULT_TOL{};

// ── Partial traces against hand-calculated targets ────────────────────

bool t_partial_trace_b_of_bell() {
    // |Φ+⟩ = (|00⟩ + |11⟩)/√2 → ρ_A = (|0⟩⟨0| + |1⟩⟨1|)/2 = I/2.
    auto rhoAB = bellPhiPlus();
    auto rhoA = partialTraceB(rhoAB, 2, 2);
    Eigen::Matrix2cd expected = 0.5 * Eigen::Matrix2cd::Identity();
    return expect_matrix_near(rhoA, expected, TOL,
        "partialTraceB(|Φ+⟩⟨Φ+|) == I/2");
}

bool t_partial_trace_a_of_bell() {
    auto rhoAB = bellPhiPlus();
    auto rhoB = partialTraceA(rhoAB, 2, 2);
    Eigen::Matrix2cd expected = 0.5 * Eigen::Matrix2cd::Identity();
    return expect_matrix_near(rhoB, expected, TOL,
        "partialTraceA(|Φ+⟩⟨Φ+|) == I/2");
}

bool t_partial_trace_of_product_recovers_marginals() {
    // ρ_A ⊗ ρ_B with ρ_A = diag(0.6, 0.4), ρ_B = diag(0.3, 0.7).
    Eigen::Matrix2cd rhoA;
    rhoA << 0.6, 0.0, 0.0, 0.4;
    Eigen::Matrix2cd rhoB;
    rhoB << 0.3, 0.0, 0.0, 0.7;
    auto rhoAB = productState(rhoA, rhoB);
    return expect_matrix_near(partialTraceB(rhoAB, 2, 2), rhoA, TOL,
            "partialTraceB(ρ_A ⊗ ρ_B) == ρ_A")
        && expect_matrix_near(partialTraceA(rhoAB, 2, 2), rhoB, TOL,
            "partialTraceA(ρ_A ⊗ ρ_B) == ρ_B");
}

bool t_partial_trace_dimensions() {
    // 2 × 3 = 6-dim joint. ρ_A ⊗ ρ_B with non-square ρ_A, ρ_B.
    Eigen::Matrix2cd rhoA = 0.5 * Eigen::Matrix2cd::Identity();
    Eigen::MatrixXcd rhoB = Eigen::MatrixXcd::Identity(3, 3) / 3.0;
    auto rhoAB = productState(rhoA, rhoB);
    return expect_matrix_near(partialTraceB(rhoAB, 2, 3), rhoA, TOL,
            "partialTraceB on 2×3 joint recovers ρ_A (2×2)")
        && expect_matrix_near(partialTraceA(rhoAB, 2, 3), rhoB, TOL,
            "partialTraceA on 2×3 joint recovers ρ_B (3×3)");
}

// ── Mutual information against hand-calculated targets ────────────────

bool t_mutual_information_of_product_is_zero() {
    // I(A:B) = S(A) + S(B) - S(AB). For ρ_A ⊗ ρ_B, S(AB) = S(A) + S(B),
    // so I = 0.
    Eigen::Matrix2cd rhoA;
    rhoA << 0.6, 0.0, 0.0, 0.4;
    Eigen::Matrix2cd rhoB;
    rhoB << 0.3, 0.0, 0.0, 0.7;
    auto rhoAB = productState(rhoA, rhoB);
    return expect_near(mutualInformation(rhoAB, 2, 2), 0.0, TOL,
        "I(ρ_A ⊗ ρ_B) == 0");
}

bool t_mutual_information_of_bell_is_2_log_2() {
    // |Φ+⟩ is pure → S(AB) = 0. Marginals I/2 → S(A) = S(B) = log 2.
    // I = 2 log 2.
    auto rhoAB = bellPhiPlus();
    return expect_near(mutualInformation(rhoAB, 2, 2),
                       2.0 * std::log(2.0), TOL,
        "I(|Φ+⟩⟨Φ+|) == 2 log 2 nats");
}

bool t_mutual_information_of_classical_is_log_2() {
    // ρ = ½(|00⟩⟨00| + |11⟩⟨11|) → S(AB) = log 2, S(A) = S(B) = log 2.
    // I = log 2 + log 2 − log 2 = log 2.
    auto rhoAB = classicallyCorrelated();
    return expect_near(mutualInformation(rhoAB, 2, 2), std::log(2.0), TOL,
        "I(classical) == log 2 nats");
}

bool t_mutual_information_of_maximally_mixed_is_zero() {
    // I_4/4 = (I_2/2) ⊗ (I_2/2). Product → I = 0.
    Eigen::MatrixXcd rhoAB = Eigen::MatrixXcd::Identity(4, 4) / 4.0;
    return expect_near(mutualInformation(rhoAB, 2, 2), 0.0, TOL,
        "I(I/4) == 0");
}

// ── Conditional states against hand-calculated targets ────────────────

bool t_conditional_b_of_bell_given_a_eigvecs() {
    // |Φ+⟩⟨Φ+| projected onto |0⟩_A is |0⟩⟨0| ⊗ ½|0⟩⟨0|; trace_A normalises
    // to σ^B = |0⟩⟨0|. Symmetric for |1⟩_A.
    auto rhoAB = bellPhiPlus();
    Eigen::Vector2cd a0(1, 0), a1(0, 1);
    Eigen::Matrix2cd b00, b11;
    b00 << 1, 0, 0, 0;
    b11 << 0, 0, 0, 1;
    Eigen::VectorXcd a0x = a0, a1x = a1;
    return expect_matrix_near(conditionalB(rhoAB, a0x, 2, 2), b00, TOL,
            "σ^B_{|0⟩_A}(Bell) == |0⟩⟨0|")
        && expect_matrix_near(conditionalB(rhoAB, a1x, 2, 2), b11, TOL,
            "σ^B_{|1⟩_A}(Bell) == |1⟩⟨1|");
}

bool t_conditional_b_of_product_is_marginal_for_any_a() {
    // For ρ_A ⊗ ρ_B, σ^B for any |a⟩ is just ρ_B.
    Eigen::Matrix2cd rhoA;
    rhoA << 0.6, 0.0, 0.0, 0.4;
    Eigen::Matrix2cd rhoB;
    rhoB << 0.3, 0.0, 0.0, 0.7;
    auto rhoAB = productState(rhoA, rhoB);
    Eigen::VectorXcd a0(2), a1(2);
    a0 << 1, 0;
    a1 << 0, 1;
    return expect_matrix_near(conditionalB(rhoAB, a0, 2, 2), rhoB, TOL,
            "σ^B_{|0⟩}(ρ_A ⊗ ρ_B) == ρ_B")
        && expect_matrix_near(conditionalB(rhoAB, a1, 2, 2), rhoB, TOL,
            "σ^B_{|1⟩}(ρ_A ⊗ ρ_B) == ρ_B");
}

bool t_conditional_b_handles_zero_weight() {
    // For |0⟩⟨0|_A ⊗ ρ_B, projecting on |1⟩_A has zero weight; the
    // conditional should fall back to the maximally-mixed sentinel.
    Eigen::Matrix2cd rhoA;
    rhoA << 1.0, 0.0, 0.0, 0.0;
    Eigen::Matrix2cd rhoB;
    rhoB << 0.7, 0.0, 0.0, 0.3;
    auto rhoAB = productState(rhoA, rhoB);
    Eigen::VectorXcd a1(2);
    a1 << 0, 1;
    Eigen::Matrix2cd halfI = 0.5 * Eigen::Matrix2cd::Identity();
    return expect_matrix_near(conditionalB(rhoAB, a1, 2, 2), halfI, TOL,
        "σ^B at zero-weight projection falls back to I/d");
}

// ── KI decomposition against hand-calculated targets ──────────────────

// Helper: tr-distance ‖A - B‖₁ / 2, used to compare density matrices.
// For Hermitian inputs this reduces to ½ Σ |λ_i(A - B)|.
double traceDistance(const Eigen::MatrixXcd& a, const Eigen::MatrixXcd& b) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(a - b);
    double s = 0.0;
    for (Eigen::Index i = 0; i < es.eigenvalues().size(); ++i) {
        s += std::abs(es.eigenvalues()[i]);
    }
    return 0.5 * s;
}

bool t_ki_on_pure_product() {
    // ρ_AB = |0⟩⟨0|_A ⊗ |0⟩⟨0|_B. ρ_A has one nonzero eigval (1.0)
    // and one zero eigval; B same. Conditional B-states for the two
    // A-eigvecs are |0⟩⟨0| and I/2 (sentinel — the |1⟩_A projection is
    // zero-weight). With the sentinel they're distinct, so the
    // connected-components step puts them in different j-blocks (or
    // the zero-weight one drops out).
    // Expectation: one well-defined block carrying the |00⟩ structure.
    Eigen::Matrix2cd rhoA = Eigen::Matrix2cd::Zero();
    rhoA(0, 0) = 1.0;
    Eigen::Matrix2cd rhoB = rhoA;
    auto rhoAB = productState(rhoA, rhoB);

    auto r = koashiImotoDecompose(rhoAB, 2, 2, DEFAULT_TOL);
    bool ok = expect_true(!r.blocks.empty(),
        "KI(|00⟩) produces at least one block");
    // Σ_j p_j should sum to 1 (the input is normalised).
    double totalWeight = 0.0;
    for (const auto& blk : r.blocks) totalWeight += blk.weight;
    ok &= expect_near(totalWeight, 1.0, TOL,
        "Σ_j p_j == 1 for KI(|00⟩)");
    return ok;
}

bool t_ki_on_mixed_product_is_trivial_l_full_r() {
    // ρ_AB = ρ_A ⊗ ρ_B with ρ_A = diag(0.6, 0.4), ρ_B = diag(0.3, 0.7).
    // All A-eigvecs have the same conditional B-state (= ρ_B). So:
    //   single j-block, K_A = K_B = 1, r_A = r_B = 2.
    //   dim_L^A = dim_L^B = 1 → Σ = 1×1 trivial.
    //   tail A = diag(0.6, 0.4) = ρ_A in A-eigenbasis.
    //   tail B = diag(0.3, 0.7) = ρ_B in B-eigenbasis.
    Eigen::Matrix2cd rhoA;
    rhoA << 0.6, 0.0, 0.0, 0.4;
    Eigen::Matrix2cd rhoB;
    rhoB << 0.3, 0.0, 0.0, 0.7;
    auto rhoAB = productState(rhoA, rhoB);

    auto r = koashiImotoDecompose(rhoAB, 2, 2, DEFAULT_TOL);
    bool ok = expect_true(r.blocks.size() == 1,
        "KI(mixed product) → exactly 1 block");
    if (r.blocks.empty()) return false;

    const auto& blk = r.blocks[0];
    ok &= expect_true(blk.dimLeftA == 1,  "block.dim_L^A == 1");
    ok &= expect_true(blk.dimLeftB == 1,  "block.dim_L^B == 1");
    ok &= expect_true(blk.dimRightA == 2, "block.dim_R^A == 2");
    ok &= expect_true(blk.dimRightB == 2, "block.dim_R^B == 2");
    ok &= expect_near(blk.weight, 1.0, TOL, "block.weight == 1");

    // A's eigvals descending are (0.6, 0.4); the tail should be diag(0.6, 0.4).
    Eigen::Matrix2cd expectedTailA;
    expectedTailA << 0.6, 0.0, 0.0, 0.4;
    ok &= expect_matrix_near(blk.tailA, expectedTailA, TOL,
        "tail A == diag(0.6, 0.4)");

    // B's eigvals descending are (0.7, 0.3); tail diag(0.7, 0.3).
    Eigen::Matrix2cd expectedTailB;
    expectedTailB << 0.7, 0.0, 0.0, 0.3;
    ok &= expect_matrix_near(blk.tailB, expectedTailB, TOL,
        "tail B == diag(0.7, 0.3)");
    return ok;
}

bool t_ki_on_bell_is_full_l_trivial_r() {
    // |Φ+⟩⟨Φ+|. ρ_A = I/2 (degenerate eigvals); the eigvecs |0⟩, |1⟩
    // have distinct conditional B-states |0⟩⟨0|, |1⟩⟨1| AND coherent
    // coupling (off-diagonal ½). Single connected component → 1 block.
    // K_A = K_B = 2 (two distinct cond states), r_A = r_B = 1.
    // dim_L^A = dim_L^B = 2, dim_R^A = dim_R^B = 1.
    // Σ_block (after weight normalisation) = |Φ+⟩⟨Φ+| itself (4×4).
    auto rhoAB = bellPhiPlus();

    auto r = koashiImotoDecompose(rhoAB, 2, 2, DEFAULT_TOL);
    bool ok = expect_true(r.blocks.size() == 1,
        "KI(Bell) → exactly 1 block");
    if (r.blocks.empty()) return false;

    const auto& blk = r.blocks[0];
    ok &= expect_true(blk.dimLeftA == 2,  "block.dim_L^A == 2");
    ok &= expect_true(blk.dimLeftB == 2,  "block.dim_L^B == 2");
    ok &= expect_true(blk.dimRightA == 1, "block.dim_R^A == 1");
    ok &= expect_true(blk.dimRightB == 1, "block.dim_R^B == 1");
    ok &= expect_near(blk.weight, 1.0, TOL, "block.weight == 1");
    // tails are 1×1 trivial identity.
    ok &= expect_matrix_near(blk.tailA,
            Eigen::MatrixXcd::Identity(1, 1), TOL,
            "tail A == [[1]]");
    ok &= expect_matrix_near(blk.tailB,
            Eigen::MatrixXcd::Identity(1, 1), TOL,
            "tail B == [[1]]");
    // Core should be the Bell state (4×4) — equal trace distance to it.
    ok &= expect_near(traceDistance(blk.coreState, bellPhiPlus()), 0.0, TOL,
        "block.core == |Φ+⟩⟨Φ+| (tr-distance)");
    return ok;
}

bool t_ki_on_classical_correlated() {
    // ρ = ½(|00⟩⟨00| + |11⟩⟨11|). ρ_A = ρ_B = I/2 (degenerate).
    // Conditional B-states: |0⟩_A → |0⟩⟨0|, |1⟩_A → |1⟩⟨1| (distinct).
    // No off-diagonal coupling (the joint is block-diagonal).
    // → two separate connected components, two j-blocks.
    // Each block: 1 A-eigvec, 1 B-eigvec, K = 1, r = 1.
    // dim_L^A = dim_L^B = dim_R^A = dim_R^B = 1. Each weight = ½.
    auto rhoAB = classicallyCorrelated();

    auto r = koashiImotoDecompose(rhoAB, 2, 2, DEFAULT_TOL);
    bool ok = expect_true(r.blocks.size() == 2,
        "KI(classical) → exactly 2 blocks");
    if (r.blocks.size() != 2) return false;

    for (size_t k = 0; k < r.blocks.size(); ++k) {
        const auto& blk = r.blocks[k];
        const std::string suf = " (block " + std::to_string(k) + ")";
        ok &= expect_true(blk.dimLeftA == 1,  "dim_L^A == 1" + suf);
        ok &= expect_true(blk.dimLeftB == 1,  "dim_L^B == 1" + suf);
        ok &= expect_true(blk.dimRightA == 1, "dim_R^A == 1" + suf);
        ok &= expect_true(blk.dimRightB == 1, "dim_R^B == 1" + suf);
        ok &= expect_near(blk.weight, 0.5, TOL, "weight == 1/2" + suf);
    }
    // Total weight is 1.
    double total = 0.0;
    for (const auto& b : r.blocks) total += b.weight;
    ok &= expect_near(total, 1.0, TOL, "Σ_j p_j == 1");
    return ok;
}

bool t_ki_on_maximally_mixed() {
    // I/4 on a 2⊗2 Hilbert space = (I/2) ⊗ (I/2). Product → 1 block.
    // All conditional states equal I/2 → K_A = K_B = 1, r_A = r_B = 2.
    // tails are I/2 (block ω_R is the eigval spread within the single group).
    Eigen::MatrixXcd rhoAB = Eigen::MatrixXcd::Identity(4, 4) / 4.0;

    auto r = koashiImotoDecompose(rhoAB, 2, 2, DEFAULT_TOL);
    bool ok = expect_true(r.blocks.size() == 1,
        "KI(I/4) → exactly 1 block");
    if (r.blocks.empty()) return false;

    const auto& blk = r.blocks[0];
    ok &= expect_true(blk.dimLeftA == 1,  "block.dim_L^A == 1");
    ok &= expect_true(blk.dimLeftB == 1,  "block.dim_L^B == 1");
    ok &= expect_true(blk.dimRightA == 2, "block.dim_R^A == 2");
    ok &= expect_true(blk.dimRightB == 2, "block.dim_R^B == 2");
    Eigen::Matrix2cd halfI = 0.5 * Eigen::Matrix2cd::Identity();
    ok &= expect_matrix_near(blk.tailA, halfI, TOL,
        "tail A == I/2");
    ok &= expect_matrix_near(blk.tailB, halfI, TOL,
        "tail B == I/2");
    return ok;
}

bool t_ki_mixed_block_two_groups_of_two() {
    // Construct a 4-dim A-side block with K_A = r_A = 2 by combining
    // two Bell pairs on disjoint A-subspaces with a classical mixture.
    //
    // ρ = ½(|Φ+⟩_{01}⟨Φ+| + |Φ+⟩_{23}⟨Φ+|) on H_A (dim 4) ⊗ H_B (dim 2)
    // is the wrong shape (joint must be dim 8 not 4). Instead use:
    //
    //   ρ = ½(|Φ+_AB⟩⟨Φ+_AB| ⊗ |0⟩⟨0|_X) + ½(|Φ+_AB⟩⟨Φ+_AB| ⊗ |1⟩⟨1|_X)
    //
    // where (A, X) is one composite system of dim 4 and B is dim 2.
    // Then:
    //   ρ_{(AX), B} = |Φ+_AB⟩⟨Φ+_AB| ⊗ (I_X / 2)
    //
    // — a product on the combined A side: |Φ+_AB⟩⟨Φ+_AB| (correlated)
    // tensored with I_X/2 (uncorrelated). The composite A-side has dim 4;
    // KI should find 1 block with K_A = 2 (from the AB Bell structure)
    // and r_A = 2 (from the trivial X factor).
    //
    // Construction in matrix form: with the row order (a_A, a_X, b)
    // = (0..1, 0..1, 0..1) flattened as ((a_A * 2 + a_X) * 2 + b):
    //
    //   ρ[(a_A * 2 + a_X) * 2 + b, (a'_A * 2 + a'_X) * 2 + b']
    //     = ½ · |Φ+⟩_AB[a_A, b ; a'_A, b'] · δ_{a_X, a'_X}
    //
    // ρ_{(AX), B} is 8×8 (dim_AX = 4, dim_B = 2). KI should give
    // 1 block, K_A = 2 (the Bell L-structure on the AB factor),
    // r_A = 2 (the trivial X factor multiplicity).
    Eigen::MatrixXcd rho = Eigen::MatrixXcd::Zero(8, 8);
    auto bell = bellPhiPlus();
    auto rowIdx = [](int aA, int aX, int b) {
        return (aA * 2 + aX) * 2 + b;
    };
    for (int aA = 0; aA < 2; ++aA) {
        for (int ap = 0; ap < 2; ++ap) {
            for (int b = 0; b < 2; ++b) {
                for (int bp = 0; bp < 2; ++bp) {
                    const auto val = 0.5 * bell(aA * 2 + b, ap * 2 + bp);
                    for (int aX = 0; aX < 2; ++aX) {
                        rho(rowIdx(aA, aX, b), rowIdx(ap, aX, bp)) = val;
                    }
                }
            }
        }
    }

    auto r = koashiImotoDecompose(rho, 4, 2, DEFAULT_TOL);
    bool ok = expect_true(r.blocks.size() == 1,
        "KI(Bell ⊗ I_X) → 1 block");
    if (r.blocks.empty()) return false;
    const auto& blk = r.blocks[0];
    // The Bell structure (K_A = 2) lives in the AB factor; the X factor
    // contributes r_A = 2. K_B = 2 (Bell on B side), r_B = 1.
    ok &= expect_true(blk.dimLeftA  == 2, "block.dim_L^A == 2");
    ok &= expect_true(blk.dimRightA == 2, "block.dim_R^A == 2");
    ok &= expect_true(blk.dimLeftB  == 2, "block.dim_L^B == 2");
    ok &= expect_true(blk.dimRightB == 1, "block.dim_R^B == 1");
    return ok;
}

bool t_ki_reproducibility() {
    // Same input → same output bitwise across two calls (canonicalization
    // conventions are deterministic).
    auto rhoAB = bellPhiPlus();
    auto r1 = koashiImotoDecompose(rhoAB, 2, 2, DEFAULT_TOL);
    auto r2 = koashiImotoDecompose(rhoAB, 2, 2, DEFAULT_TOL);
    return expect_matrix_near(r1.sigma,  r2.sigma,  0.0, "sigma  reproducible bitwise")
        && expect_matrix_near(r1.aPrime, r2.aPrime, 0.0, "aPrime reproducible bitwise")
        && expect_matrix_near(r1.bPrime, r2.bPrime, 0.0, "bPrime reproducible bitwise");
}

bool t_ki_block_weights_sum_to_unity() {
    // For any normalised ρ_AB, Σ_j p_j = 1.
    bool ok = true;
    Eigen::Matrix2cd rhoA;
    rhoA << 0.6, 0.0, 0.0, 0.4;
    Eigen::Matrix2cd rhoB;
    rhoB << 0.3, 0.0, 0.0, 0.7;
    const std::vector<Eigen::MatrixXcd> inputs = {
        bellPhiPlus(),
        classicallyCorrelated(),
        productState(rhoA, rhoB),
        Eigen::MatrixXcd::Identity(4, 4) / 4.0,
    };
    const std::vector<std::string> names = {
        "Bell", "classical", "product(0.6/0.4, 0.3/0.7)", "I/4",
    };
    for (size_t k = 0; k < inputs.size(); ++k) {
        auto r = koashiImotoDecompose(inputs[k], 2, 2, DEFAULT_TOL);
        double total = 0.0;
        for (const auto& b : r.blocks) total += b.weight;
        ok &= expect_near(total, 1.0, TOL,
            "Σ_j p_j == 1 for " + names[k]);
    }
    return ok;
}

bool t_ki_rejects_malformed_input() {
    // Non-square rhoAB.
    Eigen::MatrixXcd nonSquare(4, 3);
    nonSquare.setZero();
    bool ok = expect_throws("KI rejects non-square rhoAB",
        [&]{ (void)koashiImotoDecompose(nonSquare, 2, 2, DEFAULT_TOL); });

    // Dimension product mismatch.
    Eigen::MatrixXcd shaped = Eigen::MatrixXcd::Zero(6, 6);
    ok &= expect_throws("KI rejects rhoAB.rows() != dimA·dimB",
        [&]{ (void)koashiImotoDecompose(shaped, 2, 2, DEFAULT_TOL); });

    // Non-positive dimensions.
    ok &= expect_throws("KI rejects dimA <= 0",
        [&]{ (void)koashiImotoDecompose(bellPhiPlus(), 0, 2, DEFAULT_TOL); });
    ok &= expect_throws("KI rejects dimB <= 0",
        [&]{ (void)koashiImotoDecompose(bellPhiPlus(), 2, -1, DEFAULT_TOL); });
    return ok;
}

bool t_ki_blocks_descending_weight() {
    // For a classical mixture with weights 0.7 / 0.3, the two j-blocks
    // should appear in descending p_j order.
    Eigen::MatrixXcd rho = Eigen::MatrixXcd::Zero(4, 4);
    rho(0, 0) = 0.7;
    rho(3, 3) = 0.3;
    auto r = koashiImotoDecompose(rho, 2, 2, DEFAULT_TOL);
    bool ok = expect_true(r.blocks.size() == 2,
        "weighted classical → 2 blocks");
    if (r.blocks.size() != 2) return false;
    ok &= expect_near(r.blocks[0].weight, 0.7, TOL,
        "blocks[0].weight == 0.7 (descending)");
    ok &= expect_near(r.blocks[1].weight, 0.3, TOL,
        "blocks[1].weight == 0.3");
    return ok;
}

} // namespace

int main() {
    std::cout << "== test_koashi_imoto_core ==\n";
    bool ok = true;
    ok &= t_partial_trace_b_of_bell();
    ok &= t_partial_trace_a_of_bell();
    ok &= t_partial_trace_of_product_recovers_marginals();
    ok &= t_partial_trace_dimensions();
    ok &= t_mutual_information_of_product_is_zero();
    ok &= t_mutual_information_of_bell_is_2_log_2();
    ok &= t_mutual_information_of_classical_is_log_2();
    ok &= t_mutual_information_of_maximally_mixed_is_zero();
    ok &= t_conditional_b_of_bell_given_a_eigvecs();
    ok &= t_conditional_b_of_product_is_marginal_for_any_a();
    ok &= t_conditional_b_handles_zero_weight();
    ok &= t_ki_on_pure_product();
    ok &= t_ki_on_mixed_product_is_trivial_l_full_r();
    ok &= t_ki_on_bell_is_full_l_trivial_r();
    ok &= t_ki_on_classical_correlated();
    ok &= t_ki_on_maximally_mixed();
    ok &= t_ki_mixed_block_two_groups_of_two();
    ok &= t_ki_reproducibility();
    ok &= t_ki_block_weights_sum_to_unity();
    ok &= t_ki_rejects_malformed_input();
    ok &= t_ki_blocks_descending_weight();
    std::cout << (ok ? "ALL PASSED\n" : "SOME FAILED\n");
    return ok ? 0 : 1;
}
