#include "quantum/QuantumState.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace tessera::quantum {

namespace {

void requireSquareNonEmpty(const Eigen::MatrixXcd& m) {
    if (m.rows() == 0 || m.rows() != m.cols()) {
        throw std::invalid_argument(
            "QuantumState: density matrix must be square and non-empty");
    }
}

bool isHermitianImpl(const Eigen::MatrixXcd& m, double tol) {
    return (m - m.adjoint()).norm() <= tol * std::max(1.0, m.norm());
}

bool isPositiveSemidefiniteImpl(const Eigen::MatrixXcd& m, double tol) {
    // Hermitian-only test; assume caller has already checked Hermiticity
    // or doesn't care about the asymmetric component for this check.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(m);
    if (es.info() != Eigen::Success) return false;
    return es.eigenvalues().minCoeff() >= -tol;
}

bool hasUnitTraceImpl(const Eigen::MatrixXcd& m, double tol) {
    const auto t = m.trace();
    return std::abs(t.real() - 1.0) <= tol && std::abs(t.imag()) <= tol;
}

void validateOrThrow(const Eigen::MatrixXcd& m) {
    requireSquareNonEmpty(m);
    const double tol = 1e-10;
    if (!isHermitianImpl(m, tol)) {
        throw std::invalid_argument("QuantumState: matrix is not Hermitian");
    }
    if (!isPositiveSemidefiniteImpl(m, tol)) {
        throw std::invalid_argument(
            "QuantumState: matrix is not positive semidefinite");
    }
    if (!hasUnitTraceImpl(m, tol)) {
        throw std::invalid_argument("QuantumState: matrix does not have unit trace");
    }
}

} // namespace

QuantumState::QuantumState() noexcept
    : rho_(Eigen::MatrixXcd::Identity(1, 1)) {}

QuantumState::QuantumState(int dim) {
    if (dim <= 0) {
        throw std::invalid_argument("QuantumState: dim must be positive");
    }
    rho_ = Eigen::MatrixXcd::Identity(dim, dim) / static_cast<double>(dim);
}

QuantumState::QuantumState(Eigen::MatrixXcd rho) {
    validateOrThrow(rho);
    rho_ = std::move(rho);
}

int QuantumState::dim() const noexcept {
    return static_cast<int>(rho_.rows());
}

const Eigen::MatrixXcd& QuantumState::matrix() const noexcept { return rho_; }
Eigen::MatrixXcd&       QuantumState::matrix()       noexcept { return rho_; }

void QuantumState::setMatrix(Eigen::MatrixXcd rho) {
    validateOrThrow(rho);
    rho_ = std::move(rho);
}

void QuantumState::setMatrixUnchecked(Eigen::MatrixXcd rho) noexcept {
    rho_ = std::move(rho);
}

double QuantumState::purity() const noexcept {
    return (rho_ * rho_).trace().real();
}

double QuantumState::entropy(double tol) const noexcept {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(rho_);
    if (es.info() != Eigen::Success) return 0.0;
    double s = 0.0;
    const auto& evs = es.eigenvalues();
    for (Eigen::Index i = 0; i < evs.size(); ++i) {
        const double p = evs[i];
        if (p > tol) s -= p * std::log(p);
    }
    return s;
}

bool QuantumState::isLocallyPure(double eps) const noexcept {
    return purity() >= 1.0 - eps;
}

QuantumState QuantumState::maximallyMixed(int dim) {
    return QuantumState{dim};
}

QuantumState QuantumState::computationalBasis(int dim, int index) {
    if (dim <= 0) {
        throw std::invalid_argument("QuantumState: dim must be positive");
    }
    if (index < 0 || index >= dim) {
        throw std::out_of_range("QuantumState: index out of range");
    }
    Eigen::MatrixXcd m = Eigen::MatrixXcd::Zero(dim, dim);
    m(index, index) = 1.0;
    QuantumState q;
    q.setMatrixUnchecked(std::move(m));
    return q;
}

QuantumState QuantumState::randomMixed(int dim, double targetEntropy,
                                       std::uint64_t seed) {
    if (dim <= 0) {
        throw std::invalid_argument("QuantumState: dim must be positive");
    }
    const double maxS = std::log(static_cast<double>(dim));
    if (targetEntropy < 0.0 || targetEntropy > maxS + 1e-12) {
        throw std::invalid_argument(
            "QuantumState::randomMixed: targetEntropy outside [0, log dim]");
    }

    // Construct a target spectrum that achieves the requested entropy
    // closely, then conjugate by a Haar-random unitary to remove basis
    // bias. Spectrum: a single "peak" eigenvalue p plus (dim - 1)
    // copies of (1 - p) / (dim - 1). The von Neumann entropy of this
    // spectrum as a function of p ∈ [1/dim, 1] is monotone in p and
    // covers [0, log dim], so we bisect.

    auto spectrumEntropy = [dim](double p) {
        const double q = (1.0 - p) / static_cast<double>(dim - 1);
        double s = 0.0;
        if (p > 0.0) s -= p * std::log(p);
        if (q > 0.0) s -= (dim - 1) * q * std::log(q);
        return s;
    };

    double lo = 1.0 / static_cast<double>(dim);  // maximally mixed -> S = log dim
    double hi = 1.0;                             // pure -> S = 0
    // Note: spectrumEntropy is DECREASING in p over [1/dim, 1].
    for (int iter = 0; iter < 100; ++iter) {
        const double mid = 0.5 * (lo + hi);
        const double s = spectrumEntropy(mid);
        if (std::abs(s - targetEntropy) < 1e-12) { lo = hi = mid; break; }
        if (s > targetEntropy) lo = mid; else hi = mid;
    }
    const double p = 0.5 * (lo + hi);
    const double q = (1.0 - p) / static_cast<double>(dim - 1);

    Eigen::VectorXd spectrum(dim);
    spectrum(0) = p;
    for (int i = 1; i < dim; ++i) spectrum(i) = q;

    // Haar-random unitary via QR on a Ginibre matrix.
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> N(0.0, 1.0);
    Eigen::MatrixXcd G(dim, dim);
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            G(i, j) = std::complex<double>(N(rng), N(rng));
        }
    }
    Eigen::HouseholderQR<Eigen::MatrixXcd> qr(G);
    Eigen::MatrixXcd Q = qr.householderQ();
    // Fix the phase ambiguity so Q is Haar (Mezzadri 2007).
    Eigen::MatrixXcd R = qr.matrixQR().triangularView<Eigen::Upper>();
    Eigen::VectorXcd diag = R.diagonal();
    for (int i = 0; i < dim; ++i) {
        const auto d = diag(i);
        const std::complex<double> ph = (d == std::complex<double>(0, 0))
            ? std::complex<double>(1, 0)
            : d / std::abs(d);
        Q.col(i) *= ph;
    }

    Eigen::MatrixXcd D = Eigen::MatrixXcd::Zero(dim, dim);
    for (int i = 0; i < dim; ++i) D(i, i) = spectrum(i);
    Eigen::MatrixXcd rho = Q * D * Q.adjoint();
    // Force exact Hermiticity (numerical hygiene).
    rho = 0.5 * (rho + rho.adjoint());

    QuantumState s;
    s.setMatrixUnchecked(std::move(rho));
    return s;
}

bool QuantumState::isHermitian(double tol) const noexcept {
    return isHermitianImpl(rho_, tol);
}
bool QuantumState::isPositiveSemidefinite(double tol) const noexcept {
    return isPositiveSemidefiniteImpl(rho_, tol);
}
bool QuantumState::hasUnitTrace(double tol) const noexcept {
    return hasUnitTraceImpl(rho_, tol);
}

} // namespace tessera::quantum
