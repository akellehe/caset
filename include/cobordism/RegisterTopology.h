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
/// The \#353-style color register topology, built **without welding** (the
/// default `MergeCobordism` topology). The merge \f$ \psi_A, \psi_B \to \psi_R \f$
/// is three holed-icosahedron register blocks — each \f$ S^2 - 3\,\text{holes} \f$,
/// \f$ b_1 = 2 \f$ on the \f$ \sum = 0 \f$ hyperplane (the \f$ S_3 \f$ standard
/// rep, the color singlet \f$ [1, \omega, \omega^2] \f$) — laid with **disjoint**
/// vertex ids and joined input→result by additive **tubes** (the six lateral
/// triangles of a triangular prism between two triangular hole-circles).
///
/// Connecting by tubes, never by *identifying* a shared block, is what keeps the
/// merge a genuine trivalent manifold: the result \f$ R \f$ is a real lobe of one
/// connected complex, not the buried middle slice the \#353 `merge_cobordism.py`
/// weld produces (a `P x I` transport whose `dualComplexValid` passes only
/// because that gate misses the codim-3 pinch). `build()` asserts a manifold gate
/// — `dualComplexValid` **plus** every edge in \f$ \leq 2 \f$ triangles — and
/// throws if the seed is non-manifold, so a weld cannot recur.
///
/// Read-out: the color hole-circles only (no \f$ S^1 \f$). Each block's three
/// hole-circle periods carry that state's three color amplitudes. The carried
/// object is a color rep, so `MergeCobordism` reads a rep, not an operator.
class RegisterTopology : public TopologyBuilder {
  public:
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
      return "register (S^2-3holes color blocks, tube-joined)";
    }

  private:
    // Cached by build() for readout(): the per-block color holes (sorted vertex
    // triples), blocks in [A, B, R] order.
    std::vector<std::vector<std::vector<std::uint64_t>>> blockHoles_{};
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_REGISTERTOPOLOGY_H
