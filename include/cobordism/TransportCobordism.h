// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_TRANSPORTCOBORDISM_H
#define TESSERA_COBORDISM_TRANSPORTCOBORDISM_H

#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "cobordism/TopologyBuilder.h"

namespace tessera::spacetime { class Spacetime; }

namespace tessera::cobordism {
using ::tessera::spacetime::Spacetime;

/// # TransportCobordism
///
/// The **transport** primitive: pin boundary color states, relax the bulk to a
/// stationary point of the dual Lorentzian Regge action, and read the **emergent
/// result** by carrying the pinned inputs through the bulk to the result block.
/// This is the \#353 color register and the \#396 tripartite proton junction --- a
/// *transport* of the boundary states, not a *merge* (there is no pair-of-pants
/// and no operator extracted; the result is the inputs carried to another window).
/// Contrast `MergeCobordism`, which pins inputs **and** outputs (or `U`) and
/// extracts the merge operator or final state via the operator topology's S¹ loops.
///
/// The topology travels with the construction (`TopologyBuilder`): the default
/// `RegisterTopology` (the \#353 color register), or `TripartiteRegisterTopology`
/// (the trivalent proton junction). The topology supplies the EXACT triangle-hole
/// read-out (`readoutHoles`): the inputs are pinned over `residualForPeriods` and
/// the first unpinned window (the result block) is read off the relaxed geometry.
/// Only **inputs** are supplied --- the result EMERGES.
class TransportCobordism {
  public:
    /// The state-pinning residual: r_U realizability (the carried state's exact
    /// period non-harmonicity) or r_psi a hard carried-vs-target period gap.
    enum class StateResidualMode { Realizability, PeriodPin };

    /// Convergence + topology statistics of the relaxation.
    struct Stats {
      int relaxIterations{0};
      double residual{0.0};            ///< beta||grad_I S||^2 + r_state at the relaxed metric
      double statActionResidual{0.0};  ///< beta||grad_I S||^2 over the interior edges
      double stateResidual{0.0};       ///< r_state (r_U or r_psi) of the pinned inputs
      std::complex<double> dualAction{0.0, 0.0};  ///< final dualReggeAction()
      std::vector<int> bettiCobordism{};          ///< Betti numbers of W
      int b1Bulk{0};                              ///< b_1 of the bulk
      int interiorVertices{0};
      bool converged{false};
      std::string topology{};
      std::string stateMode{};
    };

    /// Build the transport and run the relaxation.
    /// @param inputStates the color states pinned on the input windows (d=3 each).
    /// @param topology    the transport topology (default: the \#353 register).
    TransportCobordism(
        const std::vector<std::vector<std::complex<double>>> &inputStates,
        double beta = 1.0, double epsilon = 1e-6, int maxIters = 400,
        std::uint64_t seed = 0,
        std::shared_ptr<TopologyBuilder> topology = nullptr, bool verbose = false,
        StateResidualMode stateMode = StateResidualMode::Realizability);

    [[nodiscard]] const std::vector<std::vector<std::complex<double>>> &
    inputStates() const noexcept { return inputStates_; }

    [[nodiscard]] std::shared_ptr<Spacetime> cobordism() const noexcept {
      return cobordism_;
    }

    [[nodiscard]] const std::vector<std::vector<std::uint64_t>> &boundary()
        const noexcept { return boundaryCells_; }

    [[nodiscard]] const std::vector<std::vector<std::uint64_t>> &bulk()
        const noexcept { return bulkCells_; }

    /// The EMERGENT result: the carried inputs read over the result block's color
    /// holes off the relaxed geometry (flat, length d). The transport's primary
    /// output --- the inputs carried through the bulk to the result window.
    [[nodiscard]] const std::vector<std::complex<double>> &result()
        const noexcept { return result_; }

    /// The pinned INPUT triangle holes, their induced-orientation-signed target
    /// periods, and the EMERGENT result block's holes (W's own vertex labels) ---
    /// so a caller can read the emergent result via the carried-input transport
    /// (`carriedRepresentative` over the input holes, periods over the result
    /// holes) directly.
    [[nodiscard]] const std::vector<std::vector<std::uint64_t>> &inputHoles()
        const noexcept { return stateHoles_; }
    [[nodiscard]] const std::vector<std::complex<double>> &inputHoleTargets()
        const noexcept { return holeTargets_; }
    [[nodiscard]] const std::vector<std::vector<std::uint64_t>> &resultHoles()
        const noexcept { return resultHoles_; }

    /// The result block's induced-orientation signs (\f$ \pm 1 \f$ per result hole),
    /// applied to the emergent result periods so the read-out is symmetric with the
    /// signed input targets (the relabeling-invariant Stokes charge). Empty for a
    /// topology that supplies no result signing (the raw periods are kept).
    [[nodiscard]] const std::vector<int> &resultSigns() const noexcept {
      return resultSigns_;
    }

    [[nodiscard]] const Stats &stats() const noexcept { return stats_; }

  private:
    std::vector<std::vector<std::complex<double>>> inputStates_{};
    double beta_{1.0};
    double epsilon_{1e-6};
    int maxIters_{400};
    std::uint64_t seed_{0};
    std::size_t stateDim_{0};
    bool verbose_{false};
    std::shared_ptr<TopologyBuilder> topology_{};
    StateResidualMode stateMode_{StateResidualMode::Realizability};

    // r_state: the pinned input holes + induced-orientation-signed targets (scored
    // over the EXACT residualForPeriods), and the emergent result block's holes.
    std::vector<std::vector<std::uint64_t>> stateHoles_{};
    std::vector<std::complex<double>> holeTargets_{};
    std::vector<std::vector<std::uint64_t>> resultHoles_{};
    std::vector<int> resultSigns_{};  // result block's induced-orientation signs

    // === results ===
    std::shared_ptr<Spacetime> cobordism_{};
    std::vector<std::vector<std::uint64_t>> boundaryCells_{};
    std::vector<std::vector<std::uint64_t>> bulkCells_{};
    std::vector<std::complex<double>> result_{};
    Stats stats_{};

    void buildSeed();
    void computeStateTargets();
    void optimize();
    void readResult();
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_TRANSPORTCOBORDISM_H
