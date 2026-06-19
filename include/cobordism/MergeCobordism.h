// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_MERGECOBORDISM_H
#define TESSERA_COBORDISM_MERGECOBORDISM_H

#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "cobordism/TopologyBuilder.h"

namespace tessera::spacetime { class Spacetime; }

namespace tessera::cobordism {
using ::tessera::spacetime::Spacetime;

/// # MergeCobordism
///
/// The canonical merge primitive (#363): an **emergent operator** built from
/// input/output qubit states through a cobordism bulk, mediated by the dual
/// Lorentzian Regge action while keeping a valid simplicial manifold.
///
/// The topology is supplied by a `TopologyBuilder` (#378) — defaulting to the
/// \#353-style color `RegisterTopology` (select `TorusOperatorTopology` for the
/// \f$ (T^2 - 3\,\text{holes}) \times S^1 \f$ qubit operator) — which builds
/// \f$ W \f$ and supplies the per-state read-out cycles. The interior edge
/// lengths are relaxed to a stationary point of the dual Regge action under the
/// state-pinning residual (`r = \beta\|\nabla S\|^2 + r_\psi`) by a Gauss-Newton
/// / Levenberg-Marquardt descent on the **exact analytic** Jacobian, using the
/// **sparse** action Hessian (`actionHessianExactSparse`). The operator is then
/// read off the relaxed bulk.
///
/// ## Modes
/// * **Emergent** (`U` empty): `outputStates` is required; the operator emerges.
/// * **U-supplied**: `outputStates` is computed from `U` and the inputs; `U` is
///   then ignored (a consistency check — the emergent operator should reproduce
///   the supplied `U`).
class MergeCobordism {
  public:
    /// Convergence + topology statistics of the relaxation.
    struct Stats {
      bool converged{false};            ///< the final residual fell below epsilon
      double residual{0.0};             ///< final total residual r
      double statActionResidual{0.0};   ///< final \f$ \beta\|\nabla S\|^2 \f$
      double stateResidual{0.0};        ///< final \f$ r_\psi \f$
      std::complex<double> dualAction{0.0, 0.0};  ///< final `dualReggeAction()`
      int relaxIterations{0};           ///< inner LM iterations used
      std::vector<int> bettiCobordism{};  ///< Betti numbers of \f$ W \f$
      int b1Bulk{0};                    ///< \f$ b_1(W - \partial W) \f$
      int kerL1Bulk{0};                 ///< \f$ \dim \ker L_1(W - \partial W) \f$
      int interiorVertices{0};          ///< interior (non-\f$ \partial W \f$) vertices
      std::string topology{};           ///< the TopologyBuilder's name
    };

    /// Build and run the merge.
    /// @param inputStates  the input qubit states, each a length-\f$ d \f$ vector.
    /// @param outputStates the expected output state(s); required when `U` empty.
    /// @param U            optional operator, flat row-major; when set the outputs
    ///   are computed from it and `U` is otherwise ignored.
    /// @param beta         weight on the stationary-action residual.
    /// @param epsilon      convergence tolerance on the total residual.
    /// @param maxIters     interior-relaxation iteration budget (LM steps). The
    ///   production default is 400; tests pass a small value for speed.
    /// @param seed         RNG seed for the metric jitter.
    /// @param topology     the cobordism topology; null => RegisterTopology (the
    ///   \#353-style color register default). The admissible state dimension is
    ///   topology-specific (register: \f$ d = 3 \f$; operator: a power of two).
    /// @param verbose      emit per-iteration relax progress to stderr.
    MergeCobordism(
        const std::vector<std::vector<std::complex<double>>> &inputStates,
        const std::vector<std::vector<std::complex<double>>> &outputStates,
        const std::vector<std::complex<double>> &U = {}, double beta = 1.0,
        double epsilon = 1e-6, int maxIters = 400, std::uint64_t seed = 0,
        std::shared_ptr<TopologyBuilder> topology = nullptr, bool verbose = false);

    [[nodiscard]] const std::vector<std::vector<std::complex<double>>> &
    inputStates() const noexcept { return inputStates_; }
    [[nodiscard]] const std::vector<std::vector<std::complex<double>>> &
    outputStates() const noexcept { return outputStates_; }

    /// The cobordism \f$ W \f$ (relaxed boundary + bulk).
    [[nodiscard]] std::shared_ptr<Spacetime> cobordism() const noexcept {
      return cobordism_;
    }
    /// The boundary \f$ \partial W \f$ top-cells.
    [[nodiscard]] const std::vector<std::vector<std::uint64_t>> &boundary()
        const noexcept { return boundaryCells_; }
    /// The bulk \f$ W - \partial W \f$ interior 1-cells (ker L_1 column order).
    [[nodiscard]] const std::vector<std::vector<std::uint64_t>> &bulk()
        const noexcept { return bulkCells_; }
    /// The merge operator \f$ U_{AB} = \operatorname{unvec}(\ker L_1(W-\partial W)) \f$,
    /// flat row-major \f$ d\times d \f$.
    [[nodiscard]] const std::vector<std::complex<double>> &operatorU()
        const noexcept { return operatorU_; }
    /// The Choi state of the merge operator (the carried \f$ \sum=0 \f$ period
    /// vector), flat length \f$ d^2 \f$.
    [[nodiscard]] const std::vector<std::complex<double>> &choiState()
        const noexcept { return choiState_; }

    [[nodiscard]] const Stats &stats() const noexcept { return stats_; }

  private:
    // === inputs / configuration ===
    std::vector<std::vector<std::complex<double>>> inputStates_{};
    std::vector<std::vector<std::complex<double>>> outputStates_{};
    double beta_{1.0};
    double epsilon_{1e-6};
    int maxIters_{400};
    std::uint64_t seed_{0};
    std::size_t stateDim_{0};
    bool verbose_{false};
    std::shared_ptr<TopologyBuilder> topology_{};

    // r_psi: the states pinned as period targets over the boundary cycles the
    // topology's readout() supplies.
    std::vector<TopologyBuilder::EdgeLoop> stateLoops_{};
    std::vector<std::complex<double>> stateTargets_{};

    // === results ===
    std::shared_ptr<Spacetime> cobordism_{};
    std::vector<std::vector<std::uint64_t>> boundaryCells_{};
    std::vector<std::vector<std::uint64_t>> bulkCells_{};
    std::vector<std::complex<double>> operatorU_{};
    std::vector<std::complex<double>> choiState_{};
    Stats stats_{};

    // === pipeline ===
    // Compute the output state(s) by applying U to the inputs (U-supplied mode).
    void computeOutputsFromOperator(const std::vector<std::complex<double>> &U);
    // The boundary read-out cycles + target periods, from topology_->readout().
    void computeStateTargets();
    // Build W via topology_->build(); sets cobordism_, boundaryCells_.
    void buildSeed();
    // Relax the interior edge lengths to a stationary point of the dual Regge
    // action under the state constraints (GN/LM, sparse Hessian). Fills stats_.
    void optimize();
    // Read U_AB off the relaxed bulk. Sets operatorU_, choiState_, bulkCells_,
    // and the topology fields of stats_.
    void extractOperator();
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_MERGECOBORDISM_H
