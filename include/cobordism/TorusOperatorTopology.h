// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_TORUSOPERATORTOPOLOGY_H
#define TESSERA_COBORDISM_TORUSOPERATORTOPOLOGY_H

#include "cobordism/TopologyBuilder.h"

namespace tessera::cobordism {

/// # TorusOperatorTopology
///
/// The \f$ (T^2 - 3\,\text{holes}) \times S^1 \f$ operator topology: \f$ T^3 \f$
/// minus three \f$ (\text{hole} \times S^1) \f$ solid tori. \f$ \partial W \f$ is
/// three register tori (one per state, \f$ b_1 = 2 \f$), and
/// \f$ \ker L_1(W - \partial W) = d^2 - 1 \f$ is the qubit operator. Each state
/// pins over its torus's two cycles — the hole-circle and the \f$ S^1 \f$ time
/// loop — so the read-out is six signed edge-loops jointly (over-determining the
/// bulk \f$ b_1 \f$ by the one charge-conservation relation).
class TorusOperatorTopology : public TopologyBuilder {
  public:
    [[nodiscard]] std::shared_ptr<Spacetime> build(
        std::size_t stateDim, std::uint64_t seed,
        std::vector<std::vector<std::uint64_t>> &boundaryCells) override;

    void readout(
        const std::shared_ptr<Spacetime> &cobordism,
        const std::vector<std::vector<std::complex<double>>> &states,
        std::vector<EdgeLoop> &loops,
        std::vector<std::complex<double>> &targets) const override;

    [[nodiscard]] std::size_t carriedDim(std::size_t stateDim) const override {
      return stateDim * stateDim - 1;  // ker L_1(W - dW), the Sigma=0 Choi dim
    }

    [[nodiscard]] std::size_t loopsPerState() const override {
      return 2;  // the hole-circle and the S^1 time loop per torus
    }

    [[nodiscard]] std::string name() const override {
      return "(T^2-3holes)xS^1 operator";
    }

  private:
    // Cached by build() for readout(): the three qubit boundary holes (sorted
    // vertex triples) and the S^1 layer stride (vertex-id offset per S^1 layer).
    std::vector<std::vector<std::uint64_t>> holes_{};
    std::uint64_t layerStride_{0};
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_TORUSOPERATORTOPOLOGY_H
