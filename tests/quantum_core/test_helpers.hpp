// Lightweight test harness shared by the tests/quantum_core/ suite.
//
// No gtest dependency — the rest of the project's quantum tests use
// stdout-driven PASS/FAIL assertions, so we match that convention. Each
// test function returns `bool` (true == passed); main() ANDs them all
// and returns 0 on success, 1 on any failure.

#pragma once

#include <Eigen/Dense>

#include <cmath>
#include <complex>
#include <iostream>
#include <string>

namespace tessera::test_helpers_core {

inline bool report(bool cond, const std::string& desc) {
    std::cout << "  " << desc << " ... " << (cond ? "PASS" : "FAIL") << "\n";
    return cond;
}

inline bool expect_true(bool cond, const std::string& desc) {
    return report(cond, desc);
}

inline bool expect_false(bool cond, const std::string& desc) {
    return report(!cond, desc);
}

inline bool expect_near(double actual, double expected, double tol,
                        const std::string& desc) {
    const bool ok = std::abs(actual - expected) <= tol;
    if (!ok) {
        std::cout << "    [expected " << expected
                  << ", got " << actual
                  << ", diff " << (actual - expected)
                  << ", tol " << tol << "]\n";
    }
    return report(ok, desc);
}

inline bool expect_matrix_near(const Eigen::MatrixXcd& actual,
                                const Eigen::MatrixXcd& expected,
                                double tol,
                                const std::string& desc) {
    if (actual.rows() != expected.rows() ||
        actual.cols() != expected.cols()) {
        std::cout << "    [shape mismatch: actual "
                  << actual.rows() << "x" << actual.cols()
                  << ", expected "
                  << expected.rows() << "x" << expected.cols() << "]\n";
        return report(false, desc);
    }
    const double diff = (actual - expected).norm();
    const bool ok = diff <= tol;
    if (!ok) {
        std::cout << "    [Frobenius diff " << diff
                  << ", tol " << tol << "]\n";
    }
    return report(ok, desc);
}

inline bool expect_throws(const std::string& desc, auto&& callable) {
    try {
        callable();
        return report(false, desc + " (expected exception, got none)");
    } catch (...) {
        return report(true, desc);
    }
}

// ── Reference states (in matrix form, computational basis) ──────────

// Bell state |Φ+⟩ = (|00⟩ + |11⟩) / √2, density matrix
//   ρ = ½ ( |00⟩⟨00| + |00⟩⟨11| + |11⟩⟨00| + |11⟩⟨11| )
// In the 4×4 (A ⊗ B) ordering (row idx = a·2 + b):
//   ρ[0,0] = ρ[0,3] = ρ[3,0] = ρ[3,3] = ½, rest 0.
inline Eigen::Matrix4cd bellPhiPlus() {
    Eigen::Matrix4cd r = Eigen::Matrix4cd::Zero();
    r(0, 0) = 0.5; r(0, 3) = 0.5;
    r(3, 0) = 0.5; r(3, 3) = 0.5;
    return r;
}

// Classical correlated state ½(|00⟩⟨00| + |11⟩⟨11|): the diagonal of
// the Bell state, no coherence. Same marginals as Bell, but I = log 2
// nats (not 2 log 2).
inline Eigen::Matrix4cd classicallyCorrelated() {
    Eigen::Matrix4cd r = Eigen::Matrix4cd::Zero();
    r(0, 0) = 0.5; r(3, 3) = 0.5;
    return r;
}

// Product state ρ_A ⊗ ρ_B with given (Hermitian, trace-1) marginals.
// Both must be size dimA, dimB respectively.
inline Eigen::MatrixXcd
productState(const Eigen::MatrixXcd& rhoA, const Eigen::MatrixXcd& rhoB) {
    const int dA = static_cast<int>(rhoA.rows());
    const int dB = static_cast<int>(rhoB.rows());
    Eigen::MatrixXcd p(dA * dB, dA * dB);
    for (int i = 0; i < dA; ++i) {
        for (int j = 0; j < dA; ++j) {
            for (int a = 0; a < dB; ++a) {
                for (int b = 0; b < dB; ++b) {
                    p(i * dB + a, j * dB + b) = rhoA(i, j) * rhoB(a, b);
                }
            }
        }
    }
    return p;
}

} // namespace tessera::test_helpers_core
