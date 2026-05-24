// Tests for tessera::quantum::QuantumState.
//
// Every assertion is checked against a hand-calculated reference value.
// The reference values themselves are documented inline so the
// expectations are auditable without external sources.

#include "quantum/QuantumState.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <iostream>

using namespace tessera::quantum;
using namespace tessera::test_helpers_core;

namespace {

constexpr double TOL  = 1e-10;
constexpr double TOL5 = 1e-5;  // looser for randomMixed's bisection target

// ── Constructors ──────────────────────────────────────────────────────

bool t_default_constructor_is_one_dim() {
    QuantumState s;
    return expect_true(s.dim() == 1,
        "QuantumState() has dim == 1")
        && expect_matrix_near(s.matrix(),
            Eigen::MatrixXcd::Identity(1, 1), TOL,
            "QuantumState() matrix == [[1]]");
}

bool t_explicit_dim_is_maximally_mixed() {
    // QuantumState(d) constructs I/d on a d-dim Hilbert space.
    QuantumState s(4);
    Eigen::MatrixXcd expected = Eigen::MatrixXcd::Identity(4, 4) / 4.0;
    return expect_true(s.dim() == 4, "QuantumState(4).dim() == 4")
        && expect_matrix_near(s.matrix(), expected, TOL,
            "QuantumState(4).matrix() == I/4");
}

bool t_explicit_dim_rejects_nonpositive() {
    bool ok = true;
    ok &= expect_throws("QuantumState(0) throws", []{ QuantumState s(0); });
    ok &= expect_throws("QuantumState(-1) throws", []{ QuantumState s(-1); });
    return ok;
}

bool t_matrix_constructor_validates() {
    // Valid: |0⟩⟨0|
    Eigen::Matrix2cd valid;
    valid << 1.0, 0.0,
             0.0, 0.0;
    QuantumState s(valid);
    bool ok = expect_matrix_near(s.matrix(), valid, TOL,
        "QuantumState(valid) preserves matrix");

    // Non-Hermitian: throws
    Eigen::Matrix2cd nonHerm;
    nonHerm << 0.5, std::complex<double>(0.0, 1.0),
               std::complex<double>(0.0, 1.0), 0.5;
    ok &= expect_throws("QuantumState(non-Hermitian) throws",
        [&]{ QuantumState s(nonHerm); });

    // Negative eigenvalue: throws
    Eigen::Matrix2cd negEig;
    negEig << 1.5, 0.0,
              0.0, -0.5;
    ok &= expect_throws("QuantumState(negative eigval) throws",
        [&]{ QuantumState s(negEig); });

    // Trace != 1: throws
    Eigen::Matrix2cd badTrace;
    badTrace << 0.5, 0.0,
                0.0, 0.4;
    ok &= expect_throws("QuantumState(trace != 1) throws",
        [&]{ QuantumState s(badTrace); });
    return ok;
}

// ── Factories ─────────────────────────────────────────────────────────

bool t_maximally_mixed_entropy_and_purity() {
    // For ρ = I_d/d:
    //   S(ρ) = -Σ (1/d) log(1/d) = log d nats.
    //   Tr(ρ²) = d · (1/d)² = 1/d.
    auto q2 = QuantumState::maximallyMixed(2);
    auto q4 = QuantumState::maximallyMixed(4);
    auto q8 = QuantumState::maximallyMixed(8);
    return expect_near(q2.entropy(), std::log(2.0), TOL,
            "S(I/2) == log 2")
        && expect_near(q4.entropy(), std::log(4.0), TOL,
            "S(I/4) == log 4")
        && expect_near(q8.entropy(), std::log(8.0), TOL,
            "S(I/8) == log 8")
        && expect_near(q2.purity(), 0.5, TOL,   "Tr((I/2)²) == 1/2")
        && expect_near(q4.purity(), 0.25, TOL,  "Tr((I/4)²) == 1/4")
        && expect_near(q8.purity(), 0.125, TOL, "Tr((I/8)²) == 1/8");
}

bool t_computational_basis_is_pure() {
    // |i⟩⟨i| is pure (rank-1 projector). S = 0, Tr(ρ²) = 1.
    auto q = QuantumState::computationalBasis(4, 2);
    Eigen::MatrixXcd expected = Eigen::MatrixXcd::Zero(4, 4);
    expected(2, 2) = 1.0;
    return expect_matrix_near(q.matrix(), expected, TOL,
            "computationalBasis(4, 2) == |2⟩⟨2|")
        && expect_near(q.entropy(), 0.0, TOL,
            "S(|2⟩⟨2|) == 0")
        && expect_near(q.purity(), 1.0, TOL,
            "Tr((|2⟩⟨2|)²) == 1");
}

bool t_computational_basis_index_validation() {
    return expect_throws("computationalBasis(4, 4) throws (out of range)",
            []{ QuantumState::computationalBasis(4, 4); })
        && expect_throws("computationalBasis(4, -1) throws (out of range)",
            []{ QuantumState::computationalBasis(4, -1); })
        && expect_throws("computationalBasis(0, 0) throws (dim <= 0)",
            []{ QuantumState::computationalBasis(0, 0); });
}

bool t_random_mixed_invariants_and_target_entropy() {
    // randomMixed should produce a Hermitian, positive, trace-1 state
    // whose entropy matches the requested target within bisection tol.
    bool ok = true;
    for (int dim : {2, 3, 4, 6}) {
        const double maxS = std::log(static_cast<double>(dim));
        for (double frac : {0.0, 0.3, 0.7, 1.0}) {
            const double target = frac * maxS;
            auto q = QuantumState::randomMixed(dim, target, 0xC0FFEE);
            const std::string suffix =
                " (dim=" + std::to_string(dim)
                + ", S=" + std::to_string(target) + ")";
            ok &= expect_true(q.dim() == dim,    "randomMixed dim" + suffix);
            ok &= expect_true(q.isHermitian(TOL),
                "randomMixed Hermitian" + suffix);
            ok &= expect_true(q.isPositiveSemidefinite(TOL),
                "randomMixed positive" + suffix);
            ok &= expect_true(q.hasUnitTrace(TOL),
                "randomMixed trace 1" + suffix);
            ok &= expect_near(q.entropy(), target, TOL5,
                "randomMixed entropy ≈ target" + suffix);
        }
    }
    return ok;
}

bool t_random_mixed_is_reproducible() {
    // Same (dim, target, seed) → same matrix bitwise.
    auto q1 = QuantumState::randomMixed(5, 1.2, 42);
    auto q2 = QuantumState::randomMixed(5, 1.2, 42);
    return expect_matrix_near(q1.matrix(), q2.matrix(), 0.0,
        "randomMixed with same seed is bit-identical");
}

bool t_random_mixed_rejects_out_of_range_entropy() {
    return expect_throws("randomMixed(4, -0.1, ...) throws",
            []{ QuantumState::randomMixed(4, -0.1, 0); })
        && expect_throws("randomMixed(4, log 4 + 1, ...) throws",
            []{ QuantumState::randomMixed(4, std::log(4.0) + 1.0, 0); });
}

// ── Observables ───────────────────────────────────────────────────────

bool t_entropy_of_bell_marginal_is_log_2() {
    // The Bell state |Φ+⟩ has marginal ρ_A = I/2 → S = log 2 nats.
    // We feed the marginal directly to QuantumState here; the partial
    // trace itself is tested in the KI suite.
    Eigen::Matrix2cd halfI = 0.5 * Eigen::Matrix2cd::Identity();
    QuantumState s(halfI);
    return expect_near(s.entropy(), std::log(2.0), TOL,
        "S(Bell marginal) == log 2");
}

bool t_is_locally_pure_boundaries() {
    // Pure state should pass any reasonable eps.
    auto pure = QuantumState::computationalBasis(3, 0);
    bool ok = expect_true(pure.isLocallyPure(1e-12),
        "|0⟩⟨0| isLocallyPure(1e-12)");
    ok &= expect_true(pure.isLocallyPure(1e-3),
        "|0⟩⟨0| isLocallyPure(1e-3)");

    // Maximally mixed I/2 has purity 0.5 → not pure under any tight eps.
    auto mixed = QuantumState::maximallyMixed(2);
    ok &= expect_false(mixed.isLocallyPure(1e-10),
        "I/2 not isLocallyPure(1e-10)");
    ok &= expect_false(mixed.isLocallyPure(0.4),
        "I/2 not isLocallyPure(0.4)");
    // But eps so large it swallows the purity gap accepts everything.
    ok &= expect_true(mixed.isLocallyPure(0.6),
        "I/2 isLocallyPure(0.6) — eps swallows gap");
    return ok;
}

// ── Validators (called explicitly) ────────────────────────────────────

bool t_validators_accept_valid_states() {
    auto q = QuantumState::maximallyMixed(3);
    return expect_true(q.isHermitian(TOL),    "I/3 isHermitian")
        && expect_true(q.isPositiveSemidefinite(TOL),
                                              "I/3 isPositiveSemidefinite")
        && expect_true(q.hasUnitTrace(TOL),   "I/3 hasUnitTrace");
}

bool t_set_matrix_validates_set_matrix_unchecked_does_not() {
    QuantumState s;
    Eigen::Matrix2cd valid = 0.5 * Eigen::Matrix2cd::Identity();
    s.setMatrix(valid);
    bool ok = expect_matrix_near(s.matrix(), valid, TOL,
        "setMatrix on valid accepted");

    Eigen::Matrix2cd bad;
    bad << 2.0, 0.0,
           0.0, 0.0;  // trace = 2, not 1
    ok &= expect_throws("setMatrix on invalid throws",
        [&]{ s.setMatrix(bad); });

    // setMatrixUnchecked bypasses the validators — useful when callers
    // have already established the invariants (e.g. KI construction).
    s.setMatrixUnchecked(bad);
    ok &= expect_matrix_near(s.matrix(), bad, TOL,
        "setMatrixUnchecked installs even invalid matrix");
    return ok;
}

} // namespace

int main() {
    std::cout << "== test_quantum_state_core ==\n";
    bool ok = true;
    ok &= t_default_constructor_is_one_dim();
    ok &= t_explicit_dim_is_maximally_mixed();
    ok &= t_explicit_dim_rejects_nonpositive();
    ok &= t_matrix_constructor_validates();
    ok &= t_maximally_mixed_entropy_and_purity();
    ok &= t_computational_basis_is_pure();
    ok &= t_computational_basis_index_validation();
    ok &= t_random_mixed_invariants_and_target_entropy();
    ok &= t_random_mixed_is_reproducible();
    ok &= t_random_mixed_rejects_out_of_range_entropy();
    ok &= t_entropy_of_bell_marginal_is_log_2();
    ok &= t_is_locally_pure_boundaries();
    ok &= t_validators_accept_valid_states();
    ok &= t_set_matrix_validates_set_matrix_unchecked_does_not();
    std::cout << (ok ? "ALL PASSED\n" : "SOME FAILED\n");
    return ok ? 0 : 1;
}
