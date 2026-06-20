// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_REGISTERTOPOLOGY_H
#define TESSERA_COBORDISM_REGISTERTOPOLOGY_H

#include <cstdint>
#include <vector>

#include "cobordism/TopologyBuilder.h"

namespace tessera::cobordism {

/// # RegisterTopology
///
/// The \#353-style color register topology (the default `MergeCobordism`
/// topology). The holed icosahedron (\f$ S^2 - 3 \f$ color holes, \f$ b_1 = 2 \f$
/// on the \f$ \sum = 0 \f$ hyperplane — the \f$ S_3 \f$ standard rep, the color
/// singlet \f$ [1, \omega, \omega^2] \f$) is extruded over a **3-layer staircase**
/// (`Spacetime::prismCells`) into one connected 3-complex with
/// \f$ b_1(W) = 2 \f$: **one shared color register** across the three blocks
/// (block A = layer 0, B = layer 1, R = layer 2). That \f$ b_1 = 2 \f$ is the
/// confinement — a \f$ \sum \neq 0 \f$ (colored) config cannot be carried, so its
/// realizability residual floors, while a \f$ \sum = 0 \f$ (color-neutral) config
/// realizes (reproducing #353's realizability map / S₃ gauge invariance).
///
/// The continuous staircase is a genuine valid manifold — every triangle in
/// \f$ \leq 2 \f$ tets, `dualComplexValid` — **not** the welded shared-block
/// construction (a `P x I` transport that buries the result); `build()` asserts
/// the manifold gate and throws on a non-manifold seed.
///
/// Read-out: the color hole-circles only (no \f$ S^1 \f$). Each block's three
/// hole-circle periods carry that state's three color amplitudes, with the target
/// periods pre-multiplied by the induced-orientation covector `kColorSign`
/// (the #353 `SIGN_BLOCK`) so the carried condition is \f$ \text{sign} \cdot
/// \psi = 0 \f$. The carried object is a color rep, so `MergeCobordism` reads a
/// rep, not an operator.
class RegisterTopology : public TopologyBuilder {
  public:
    /// The induced-orientation covector on the three color holes
    /// (`endSignCovector` of the icosahedron color holes; the #353 `SIGN_BLOCK`
    /// `(+1, +1, -1)`). Target periods are pre-multiplied by it in `readout()`.
    static const int kColorSign[3];

    [[nodiscard]] std::shared_ptr<Spacetime> build(
        std::size_t stateDim, std::uint64_t seed,
        std::vector<std::vector<std::uint64_t>> &boundaryCells) override;

    void readout(
        const std::shared_ptr<Spacetime> &cobordism,
        const std::vector<std::vector<std::complex<double>>> &states,
        std::vector<EdgeLoop> &loops,
        std::vector<std::complex<double>> &targets) const override;

    [[nodiscard]] std::size_t carriedDim(std::size_t /*stateDim*/) const override {
      return 2;  // the S_3 standard rep (b_1 = 2 on the Sigma=0 hyperplane)
    }

    [[nodiscard]] std::size_t loopsPerState() const override {
      return 3;  // the three color hole-circles per register block
    }

    /// The color register is \f$ d = 3 \f$ (the color triple on \f$ \sum=0 \f$).
    void validateStateDim(std::size_t d) const override;

    [[nodiscard]] std::string name() const override {
      return "register ((S^2-3color-holes) x [0,2] staircase, b1=2)";
    }

  private:
    // Cached by build() for readout(): the per-block color holes (sorted vertex
    // triples), blocks in [A, B, R] order.
    std::vector<std::vector<std::vector<std::uint64_t>>> blockHoles_{};
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_REGISTERTOPOLOGY_H
