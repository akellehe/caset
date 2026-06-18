// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_MERGECOBORDISM_H
#define TESSERA_COBORDISM_MERGECOBORDISM_H

#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime { class Spacetime; }
namespace tessera::cobordism {
using namespace ::tessera::spacetime;

/// # MergeCobordism
///
/// The canonical merge primitive (#363): build an **emergent operator** from
/// starting and ending states through a pair-of-pants cobordism bulk, mediated
/// by the Regge action on the dual complex while keeping a valid simplicial
/// manifold in both spaces. Implements `docs/design/cobordism.md`.
///
/// Given two qubit inputs \f$ \psi_A, \psi_B \f$ and an expected output
/// \f$ \psi_{AB} \f$, it assembles the boundary
/// \f$ \partial W = \operatorname{geo}(\psi_A) \sqcup \operatorname{geo}(\psi_B)
/// \sqcup \operatorname{geo}(\psi_{AB}) \f$ (each a register surface whose
/// \f$ \ker L_1 \f$ carries its state), seeds the connecting bulk with an
/// icosahedron, opens \f$ \dim(U^{Choi})+1 = 4 \f$ vertex-disjoint holes by
/// boundary-fixed surgery so the bulk's \f$ b_1 = 3 \f$ (the \f$ \sum = 0 \f$
/// charge-conserving operator dimension), then relaxes the interior edge lengths
/// to a stationary point of the dual Regge action — minimizing
/// \f$ r = \beta\|\nabla S_{Regge}\|^2 + \sum_i \|(I-P_{\ker L_1^i})\psi_i\|^2 \f$
/// by a Gauss-Newton / Levenberg-Marquardt descent on the **exact analytic**
/// Jacobian (the action Hessian `ReggeSolver::actionHessianExact` as the
/// stationarity block, `EigenstateSynthesis::residualForPeriodsGradient` as the
/// state block — no finite differences). The converge → `RemoveMove` /
/// fail → `AddMove` loop (boundary-fixed Pachner moves, topology-preserving)
/// finds the minimal realizing metric. The merge operator is then read off the
/// relaxed bulk, \f$ U_{AB} = \operatorname{unvec}(\ker L_1(W - \partial W)) \f$.
///
/// ## Modes
///
/// * **Emergent** (`U` empty): `outputStates` is required and the operator
///   emerges from the relaxation.
/// * **U-supplied** (`U` non-empty): `outputStates` is *computed* from `U` and
///   the inputs; `U` is then ignored except for that calculation, and the
///   algorithm runs unchanged (a consistency check — the emergent operator
///   should reproduce the supplied `U`).
class MergeCobordism {
  public:
    /// Convergence + topology statistics of the relaxation, for introspection.
    struct Stats {
      bool converged{false};        ///< the final residual fell below epsilon
      double residual{0.0};         ///< final total residual r
      double statActionResidual{0.0};  ///< final \f$ \beta\|\nabla S_{Regge}\|^2 \f$
      double stateResidual{0.0};    ///< final \f$ \sum_i \|(I-P_i)\psi_i\|^2 \f$
      std::complex<double> dualAction{0.0, 0.0};  ///< final `dualReggeAction()`
      int attempts{0};              ///< Add/Remove iterations used (<= maxAttempts)
      int addMoves{0};              ///< accepted AddMove count
      int removeMoves{0};           ///< accepted RemoveMove count
      int flipMoves{0};             ///< accepted flip / iflip / shift moves
      int relaxIterations{0};       ///< total inner LM iterations across attempts
      // === observed topology, called out per spec ===
      std::vector<int> bettiCobordism{};  ///< Betti numbers of \f$ W \f$
      int b1Bulk{0};                ///< \f$ b_1(W - \partial W) = \dim \ker L_1 \f$
      int kerL1Bulk{0};             ///< \f$ \dim \ker L_1(W - \partial W) \f$ (== b1Bulk)
      int interiorVertices{0};      ///< interior (non-\f$ \partial W \f$) vertex count
      std::string topology{};       ///< human-readable topology call-out
    };

    /// Build and run the merge.
    ///
    /// @param inputStates  the input qubit states (e.g. \f$ \{\psi_A, \psi_B\} \f$),
    ///   each a length-\f$ d \f$ complex amplitude vector.
    /// @param outputStates the expected output state(s) (e.g. \f$ \{\psi_{AB}\} \f$);
    ///   **required** when `U` is empty, **ignored** (recomputed from `U`) when
    ///   `U` is supplied.
    /// @param U            optional operator, flat row-major
    ///   (\f$ U_{ij} = U[i\,d + j] \f$). When non-empty the output states are
    ///   computed by applying `U` to the inputs and `U` is otherwise ignored.
    /// @param beta         weight \f$ \beta \f$ on the stationary-action residual.
    /// @param epsilon      convergence tolerance on the total residual \f$ r \f$.
    /// @param maxAttempts  cap on Add/Remove loop iterations (default 100).
    /// @param seed         RNG seed for the Pachner moves / LM restarts.
    /// @throws std::invalid_argument if `inputStates` is empty, the (effective)
    ///   `outputStates` is empty, or a state length is not a power of two; if `U`
    ///   is supplied with a size that is not \f$ d \times d \f$ for the input
    ///   dimension \f$ d \f$.
    MergeCobordism(
        const std::vector<std::vector<std::complex<double>>> &inputStates,
        const std::vector<std::vector<std::complex<double>>> &outputStates,
        const std::vector<std::complex<double>> &U = {},
        double beta = 1.0, double epsilon = 1e-6, int maxAttempts = 100,
        std::uint64_t seed = 0, bool verbose = false);

    // === Introspection members (docs/design/cobordism.md) ===

    /// The input states \f$ \{\psi_i\}_{\text{in}} \f$ as supplied.
    [[nodiscard]] const std::vector<std::vector<std::complex<double>>> &
    inputStates() const noexcept { return inputStates_; }

    /// The output states \f$ \{\psi_{AB}\} \f$ — as supplied (emergent mode) or
    /// as computed from `U` (U-supplied mode).
    [[nodiscard]] const std::vector<std::vector<std::complex<double>>> &
    outputStates() const noexcept { return outputStates_; }

    /// The cobordism \f$ W \f$ — the full relaxed simplicial complex
    /// (\f$ \partial W \f$ + bulk).
    [[nodiscard]] std::shared_ptr<Spacetime> cobordism() const noexcept {
      return cobordism_;
    }

    /// The boundary \f$ \partial W \f$: the boundary top cells (sorted vertex-id
    /// tuples) — the union of the three register surfaces.
    [[nodiscard]] const std::vector<std::vector<std::uint64_t>> &boundary()
        const noexcept { return boundaryCells_; }

    /// The bulk \f$ W - \partial W \f$: the interior 1-cells (the edges both of
    /// whose endpoints are interior), in `ker L_1` column order — the geometry
    /// the operator is read from (`EigenstateSynthesis::bulkMinusBoundaryCells`).
    [[nodiscard]] const std::vector<std::vector<std::uint64_t>> &bulk()
        const noexcept { return bulkCells_; }

    /// The merge operator \f$ U_{AB} = \operatorname{unvec}(\ker L_1(W -
    /// \partial W)) \f$, flat row-major (\f$ d \times d \f$). The emergent
    /// operator (emergent mode) or the operator read back from the cobordism
    /// built for the supplied `U` (U-supplied mode). Empty if the bulk carried
    /// no operator (\f$ \ker L_1 = 0 \f$ — the relaxation never opened the
    /// holes).
    [[nodiscard]] const std::vector<std::complex<double>> &operatorU()
        const noexcept { return operatorU_; }

    /// The Choi state of the merge operator — the carried harmonic of the bulk,
    /// the \f$ \sum = 0 \f$ charge-conserving period vector `unvec`'d into
    /// `operatorU()`. Flat, length \f$ d^2 \f$.
    [[nodiscard]] const std::vector<std::complex<double>> &choiState()
        const noexcept { return choiState_; }

    /// The emergent final state \f$ \psi_{AB} \f$: the metric \f$ L_1(W) \f$
    /// harmonic carrying the inputs, read over the output torus's cycles — the
    /// L_1(W) read-out's output (required to be a harmonic of the whole system).
    [[nodiscard]] const std::vector<std::complex<double>> &outputState()
        const noexcept { return outputState_; }

    /// Convergence + topology statistics.
    [[nodiscard]] const Stats &stats() const noexcept { return stats_; }

  private:
    // === inputs / configuration ===
    std::vector<std::vector<std::complex<double>>> inputStates_{};
    std::vector<std::vector<std::complex<double>>> outputStates_{};
    double beta_{1.0};
    double epsilon_{1e-12};   // converged only when r <= ~1e-12
    int maxAttempts_{200};    // boundary-fixed Pachner moves to search before giving up
    std::uint64_t seed_{0};
    bool verbose_{false};   // emit per-phase progress to stderr
    std::size_t stateDim_{0};  // d (qubit => 2)
    // The qubit boundary holes (∂W = 3 tori) and the S^1 layer stride, set by
    // buildSeed and read by computeStateTargets to build the register cycles.
    std::vector<std::vector<std::uint64_t>> holes_{};
    std::uint64_t layerStride_{0};
    // The r_psi term: the states pinned as period targets over the boundary tori.
    // Each qubit (a0, a1) -> its torus's two cycles (hole-circle, S^1):
    // stateLoops_[q] is a signed edge-loop, stateTargets_[q] its target period.
    // The JOINT readout (all 6 cycles) over-determines the bulk's b_1 harmonics
    // (m = nd + 1, the +1 a charge-conservation relation), so residualForLoops
    // bites and its gradient is non-singular; a single torus's 2 cycles would not.
    std::vector<std::vector<std::pair<std::uint64_t, std::uint64_t>>> stateLoops_{};
    std::vector<std::complex<double>> stateTargets_{};

    // === results ===
    std::shared_ptr<Spacetime> cobordism_{};                 // W
    std::vector<std::vector<std::uint64_t>> boundaryCells_{};  // ∂W top cells
    std::vector<std::vector<std::uint64_t>> bulkCells_{};      // W − ∂W (interior 1-cells)
    std::vector<std::complex<double>> operatorU_{};           // U_AB (row-major d×d)
    std::vector<std::complex<double>> choiState_{};           // ker L₁ carried Choi state
    std::vector<std::complex<double>> outputState_{};         // emergent ψ_AB (L₁(W) read-out)
    Stats stats_{};

    // === pipeline (docs/design/cobordism.md) ===

    // Compute the output state(s) by applying the supplied operator U to the
    // inputs (U-supplied mode), so the boundary pins the U-predicted final state.
    void computeOutputsFromOperator(const std::vector<std::complex<double>> &U);

    // Build the boundary register cycles (stateLoops_) and their target periods
    // (stateTargets_) from the states: each qubit -> its torus's hole-circle and
    // S^1 cycle, the joint r_psi readout.
    void computeStateTargets();

    // Assemble ∂W = geo(ψ_A) ⊔ geo(ψ_B) ⊔ geo(ψ_AB) and the icosahedron-seeded
    // connecting bulk into one valid manifold; open dim(U^Choi)+1 vertex-disjoint
    // holes by boundary-fixed surgery so b₁(W − ∂W) = dim(U^Choi). Sets
    // cobordism_, boundaryCells_.
    void buildSeed();

    // The converge → RemoveMove / fail → AddMove loop: relax the interior edge
    // lengths to a stationary point of the dual Regge action under the state
    // constraints, growing/pruning the interior by boundary-fixed Pachner moves
    // until r < epsilon or maxAttempts is reached. Fills stats_.
    void optimize();

    // Read U_AB = unvec(ker L₁(W − ∂W)) off the relaxed bulk. Sets operatorU_,
    // choiState_, bulkCells_, and the topology fields of stats_.
    void extractOperator();
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_MERGECOBORDISM_H
