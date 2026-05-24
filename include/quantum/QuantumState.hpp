// Density matrix on a single quantum system.
//
// Every Vertex in the simulation carries one of these. Vertex
// dimensions vary (Σ vertices may have higher dimension than A/B
// vertices) and Σ vs A/B is only a state-shape distinction, not a
// type tag.
//
// Backing storage is Eigen::MatrixXcd with runtime dimension. All
// entropies returned by helpers here are in nats (base e), matching
// the project-wide convention.

#pragma once

#include <Eigen/Dense>

#include <cstdint>

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::quantum {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;

class QuantumState {
  public:
    // 1-dim trivial state (the I/1 "placeholder" for empty Σ vertices,
    // null tails, etc.). Sized this small so a Vertex that has not been
    // initialized still has a well-defined trace-1 state.
    QuantumState() noexcept;

    // Maximally mixed state on a Hilbert space of dimension d: I/d.
    explicit QuantumState(int dim);

    // From an explicit density matrix. Validates Hermiticity, positivity
    // and unit trace within tolerance; throws std::invalid_argument when
    // the input is malformed. Use setMatrixUnchecked when you've already
    // validated upstream and need the constructor to skip the checks.
    explicit QuantumState(Eigen::MatrixXcd rho);

    // Dimension of the carrier Hilbert space.
    [[nodiscard]] int dim() const noexcept;

    // The density matrix, const and mutable. Mutable access bypasses the
    // construction-time validators; the caller is responsible for keeping
    // the matrix Hermitian, positive, and trace-one.
    [[nodiscard]] const Eigen::MatrixXcd& matrix() const noexcept;
    [[nodiscard]] Eigen::MatrixXcd&       matrix()       noexcept;

    // Replace the matrix with explicit validation (same semantics as the
    // matrix constructor). Throws on a malformed input.
    void setMatrix(Eigen::MatrixXcd rho);

    // Replace the matrix without validation. For callers that have
    // already guaranteed Hermiticity, positivity, and unit trace, e.g.
    // KI block construction or partial trace.
    void setMatrixUnchecked(Eigen::MatrixXcd rho) noexcept;

    // Tr(ρ²): 1.0 for pure states, 1/d for the maximally mixed state,
    // and anywhere in [1/d, 1] in general.
    [[nodiscard]] double purity() const noexcept;

    // Von Neumann entropy S(ρ) = -Σ λ log λ in nats. Eigenvalues below
    // `tol` are treated as zero, avoiding the log(0) singularity.
    [[nodiscard]] double entropy(double tol = 1e-12) const noexcept;

    // True iff Tr(ρ²) ≥ 1 - eps. The simulation's "is this vertex still
    // pure?" predicate; default eps matches InteractionConfig::epsLocalPure.
    [[nodiscard]] bool isLocallyPure(double eps = 1e-10) const noexcept;

    // ── Static factories ───────────────────────────────────────────────

    // I_d / d on a d-dim Hilbert space. Entropy = log d, purity = 1/d.
    [[nodiscard]] static QuantumState maximallyMixed(int dim);

    // |i⟩⟨i| on a d-dim Hilbert space. Pure (entropy 0, purity 1).
    // Throws std::out_of_range if index is not in [0, dim).
    [[nodiscard]] static QuantumState computationalBasis(int dim, int index);

    // A Haar-random mixed state on H_d with target von Neumann entropy
    // `targetEntropy` (nats). Implementation: sample a Haar-random pure
    // state on H_d ⊗ H_k where k is chosen so the marginal has the
    // requested entropy band, then trace out the ancilla. Reproducible
    // given `seed`.
    //
    // Throws std::invalid_argument when targetEntropy is outside
    // [0, log dim].
    [[nodiscard]] static QuantumState
    randomMixed(int dim, double targetEntropy, std::uint64_t seed);

    // ── Validation (used by tests and ASSERTIONS builds) ───────────────

    [[nodiscard]] bool isHermitian(double tol = 1e-10) const noexcept;
    [[nodiscard]] bool isPositiveSemidefinite(double tol = 1e-10) const noexcept;
    [[nodiscard]] bool hasUnitTrace(double tol = 1e-10) const noexcept;

  private:
    Eigen::MatrixXcd rho_;
};

} // namespace tessera::quantum
