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
/// The canonical merge primitive (#363): an **emergent operator / output state**
/// built from input/output qubit states through a cobordism bulk, mediated by the
/// dual Lorentzian Regge action while keeping a valid simplicial manifold.
///
/// The topology is supplied by a `TopologyBuilder` (#378) — defaulting to the
/// \#353-style color `RegisterTopology` (select `TorusOperatorTopology` for the
/// \f$ (T^2 - 3\,\text{holes}) \times S^1 \f$ qubit operator) — which builds
/// \f$ W \f$ and supplies the per-state read-out cycles. The interior edge
/// lengths are relaxed to a stationary point of the dual Regge action under the
/// state-pinning residual (`r = \beta\|\nabla S\|^2 + r_state`) by a Gauss-Newton
/// / Levenberg-Marquardt descent on the **exact analytic** Jacobian, using the
/// **sparse** action Hessian (`actionHessianExactSparse`). The emergent quantities
/// are then read off the relaxed bulk.
///
/// ## Modes — the primary emergent quantity is the one the caller did NOT supply
///
/// * **Output-supplied** (`U` empty): `outputStates` is pinned *together with* the
///   inputs as \f$ \partial W \f$, the interior relaxes, and the **operator** is
///   the primary emergent quantity (`operatorU()` — see the deferral note below).
/// * **U-supplied** (`outputStates` omitted): the expected output is computed by
///   applying `U` to the inputs and pinned as the output target, and the **output
///   state** is the primary emergent quantity — read back over the output cycles
///   by transporting the inputs through the relaxed geometry (the #353
///   `inputs → emergent output` flow), via `outputState()`. (Making the transport
///   itself realize `U` — "`U` as the bulk constraint" — is deferred; see below.)
///
/// `outputState()` is populated in **both** modes — primary when `U` was supplied,
/// a consistency read when the output was supplied. It is genuinely emergent: the
/// read carries only the **inputs** (`carriedRepresentativeOverLoops` on the input
/// cycles) and reads their periods over the output cycles, so it is input-dominated
/// and independent of the pinned target, not an echo of the seed. NB on the current
/// topology the input→output transport is `≈ identity`, so `outputState()` tracks
/// the (transported) inputs and does **not** yet reflect `U`'s action — reflecting
/// `U` is the operator-as-bulk-constraint that awaits the same operator-topology
/// rework as the operator read-out below.
///
/// `operatorU()` / `choiState()` (the
/// \f$ \operatorname{unvec}(\ker L_1(W-\partial W)) \f$ operator read-out) are
/// **deferred** pending the interior-handle operator-topology rework: on the
/// current topology \f$ \ker L_1(W-\partial W) \f$ is a \f$ d^2-1 \f$-dim
/// interior-cochain subspace with no basis-independent map to the
/// \f$ d\times d \f$ operator (it needs distinguished interior Choi-cycles the
/// topology does not yet supply), so they stay **empty** rather than report a
/// frame-dependent value. (#376)
///
/// ## State-residual term \f$ r_\text{state} \f$ (#377, selectable)
/// The matter/state term defaults to the #353 **realizability** residual
/// \f$ r_U \f$ (`EigenstateSynthesis::residualForLoops`): it holds the state's
/// target periods **exactly** (a minimal leak) and scores the non-harmonicity
/// \f$ \|L\psi-\lambda\psi\|^2 \f$ — \f$ \to 0 \f$ for a realizable (e.g.
/// \f$ \sum = 0 \f$) state, floored otherwise (confinement). The **hard
/// period-pin** \f$ r_\psi \f$ (`EigenstateSynthesis::periodGapForLoops`, the
/// carried-vs-target period gap \f$ \|P^{\top}c-\text{target}\|^2 \f$ with the
/// carried object a pure harmonic) is selectable via `StateResidualMode` for
/// comparison. Both share the realizable zero set; they differ off-zero.
class MergeCobordism {
  public:
    /// The matter/state term \f$ r_\text{state} \f$ the relaxation pins against.
    enum class StateResidualMode {
      Realizability,  ///< \f$ r_U \f$ (default): hold the periods exact, score the
                      ///< state's non-harmonicity (`residualForLoops`).
      PeriodPin,      ///< \f$ r_\psi \f$: pure-harmonic carried-vs-target period
                      ///< gap (`periodGapForLoops`).
    };

    /// Convergence + topology statistics of the relaxation.
    struct Stats {
      bool converged{false};            ///< the final residual fell below epsilon
      double residual{0.0};             ///< final total residual r
      double statActionResidual{0.0};   ///< final \f$ \beta\|\nabla S\|^2 \f$
      double stateResidual{0.0};        ///< final \f$ r_\text{state} \f$ (the
                                        ///< selected term: r_U or r_psi)
      std::complex<double> dualAction{0.0, 0.0};  ///< final `dualReggeAction()`
      int relaxIterations{0};           ///< inner LM iterations used
      std::vector<int> bettiCobordism{};  ///< Betti numbers of \f$ W \f$
      int b1Bulk{0};                    ///< \f$ b_1(W - \partial W) \f$
      int kerL1Bulk{0};                 ///< \f$ \dim \ker L_1(W - \partial W) \f$
      int interiorVertices{0};          ///< interior (non-\f$ \partial W \f$) vertices
      std::string topology{};           ///< the TopologyBuilder's name
      std::string stateMode{};          ///< the selected r_state term ("r_U" / "r_psi")
    };

    /// Build and run the merge.
    /// @param inputStates  the input qubit states, each a length-\f$ d \f$ vector.
    /// @param outputStates the expected output state(s). Required when `U` is
    ///   empty; **must be omitted** when `U` is supplied (then computed from `U` —
    ///   the two modes are mutually exclusive). Throws if both are supplied.
    /// @param U            optional operator, flat row-major; when set the outputs
    ///   are computed by applying it to the inputs and the output state is the
    ///   primary emergent quantity (`U`-supplied mode).
    /// @param beta         weight on the stationary-action residual.
    /// @param epsilon      convergence tolerance on the total residual.
    /// @param maxIters     interior-relaxation iteration budget (LM steps). The
    ///   production default is 400; tests pass a small value for speed.
    /// @param seed         RNG seed for the metric jitter.
    /// @param topology     the cobordism topology; null => RegisterTopology (the
    ///   \#353-style color register default). The admissible state dimension is
    ///   topology-specific (register: \f$ d = 3 \f$; operator: a power of two).
    /// @param verbose      emit per-iteration relax progress to stderr.
    /// @param stateMode    the \f$ r_\text{state} \f$ term — `Realizability`
    ///   (\f$ r_U \f$, default) or `PeriodPin` (\f$ r_\psi \f$, for comparison).
    MergeCobordism(
        const std::vector<std::vector<std::complex<double>>> &inputStates,
        const std::vector<std::vector<std::complex<double>>> &outputStates = {},
        const std::vector<std::complex<double>> &U = {}, double beta = 1.0,
        double epsilon = 1e-6, int maxIters = 400, std::uint64_t seed = 0,
        std::shared_ptr<TopologyBuilder> topology = nullptr, bool verbose = false,
        StateResidualMode stateMode = StateResidualMode::Realizability);

    /// The input states \f$ \{\psi_i\} \f$ as supplied.
    [[nodiscard]] const std::vector<std::vector<std::complex<double>>> &
    inputStates() const noexcept { return inputStates_; }
    /// The output state(s) — as supplied (output-supplied mode), or computed by
    /// applying `U` to each input (`U`-supplied mode). Distinct from the *emergent*
    /// single `outputState()` read off the relaxed geometry.
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
    /// flat row-major \f$ d\times d \f$. **Deferred** (always empty for now): the
    /// principled read-out awaits the interior-handle operator-topology rework —
    /// see the Modes note. Empty also signals the bulk carried no operator.
    [[nodiscard]] const std::vector<std::complex<double>> &operatorU()
        const noexcept { return operatorU_; }
    /// The Choi state of the merge operator (the carried \f$ \sum=0 \f$ period
    /// vector), flat length \f$ d^2 \f$. **Deferred** with `operatorU()` (empty).
    [[nodiscard]] const std::vector<std::complex<double>> &choiState()
        const noexcept { return choiState_; }
    /// The emergent final state \f$ \psi_{AB} \f$: the periods of the metric
    /// \f$ L_1(W) \f$ harmonic carrying the **inputs** read over the output
    /// cycles — the output the relaxed geometry produces from the inputs alone.
    /// The primary emergent quantity in `U`-supplied mode; a consistency read
    /// (emergent vs. the pinned target) in output-supplied mode. Flat; length is
    /// `loopsPerState` \f$ \times \f$ (number of output states) (\f$ d \f$ for a
    /// single qubit output). Empty when the read is not determinate — when some state
    /// went unpinned (more states than the topology's capacity), so the
    /// input/output cycle split cannot be trusted.
    [[nodiscard]] const std::vector<std::complex<double>> &outputState()
        const noexcept { return outputState_; }

    /// Convergence + topology statistics of the relaxation.
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
    StateResidualMode stateMode_{StateResidualMode::Realizability};

    // r_state: the states pinned as period targets over the boundary cycles the
    // topology's readout() supplies (scored by r_U or r_psi per stateMode_).
    std::vector<TopologyBuilder::EdgeLoop> stateLoops_{};
    std::vector<std::complex<double>> stateTargets_{};

    // === results ===
    std::shared_ptr<Spacetime> cobordism_{};
    std::vector<std::vector<std::uint64_t>> boundaryCells_{};
    std::vector<std::vector<std::uint64_t>> bulkCells_{};
    std::vector<std::complex<double>> operatorU_{};
    std::vector<std::complex<double>> choiState_{};
    std::vector<std::complex<double>> outputState_{};  // emergent psi_AB (#376)
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
