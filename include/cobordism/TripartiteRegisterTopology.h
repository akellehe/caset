// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_TRIPARTITEREGISTERTOPOLOGY_H
#define TESSERA_COBORDISM_TRIPARTITEREGISTERTOPOLOGY_H

#include <cstdint>
#include <vector>

#include "cobordism/TopologyBuilder.h"

namespace tessera::cobordism {

/// # TripartiteRegisterTopology
///
/// The **trivalent W_ABC junction** (#396): three color-neutral q-qbar pair
/// inputs into one bulk, \f$ \partial W = \psi_A \sqcup \psi_B \sqcup \psi_C \f$,
/// whose **emergent** result block reads out the proton's color register.
///
/// The decisive design fact (#382 / the obstruction note): a single shared color
/// register **averages** its inputs (least-squares projection onto one period
/// space → keeps the symmetric/sextet part) and so can never produce the totally
/// antisymmetric singlet \f$ \varepsilon_{ijk} \f$. The fix is **distinct holes =
/// independent cycles**: the three inputs and the result are **four spatially
/// separated windows of three color holes each**, all on ONE connected base
/// surface, a 2-frequency geodesic icosahedron (\f$ S^2 \f$, 42 vertices) minus
/// 12 vertex-disjoint hole triangles, extruded \f$ \times I \f$ (`prismCells`).
/// The four windows are A, B, C (inputs) and R (the emergent result).
///
/// Confinement is then **conservation at the junction**: on a connected
/// surface-minus-holes the global Stokes relation forces
/// \f$ \sum_{\text{all holes}} (\text{induced period}) = 0 \f$, so
/// \f$ \Sigma_R = -\Sigma_{\text{inputs}} \f$ — three color-neutral inputs
/// (\f$ \Sigma = 0 \f$ each) give a color-neutral result. (Unlike the
/// single-register `RegisterTopology`, a colored input is NOT floored here at the
/// input level — its hole is an independent cycle — so candidate A conserves
/// charge rather than confining at the input; per-input confinement is the
/// heavier tube-connected junction, deferred.)
///
/// Read-out: each input window's three color holes carry that state's three color
/// amplitudes, the target periods pre-multiplied by the window's
/// induced-orientation covector (`ChainComplex::endSignCovector`, the
/// generalization of `RegisterTopology::kColorSign`). The result block R emerges
/// (`emergesResult() == true`); it is read off the relaxed geometry, never
/// pinned. The carried object is a color rep, so `MergeCobordism` reads a rep.
class TripartiteRegisterTopology : public TopologyBuilder {
  public:
    [[nodiscard]] std::shared_ptr<Spacetime> build(
        std::size_t stateDim, std::uint64_t seed,
        std::vector<std::vector<std::uint64_t>> &boundaryCells) override;

    /// The EXACT triangle-hole read-out (the #353 period path). Pins the supplied
    /// input states on windows A, B, C over `residualForPeriods`; returns the
    /// first unpinned window (R) as `resultHoles` to be read after the relax.
    void readoutHoles(
        const std::shared_ptr<Spacetime> &cobordism,
        const std::vector<std::vector<std::complex<double>>> &states,
        std::vector<std::vector<std::uint64_t>> &inputHoles,
        std::vector<std::complex<double>> &inputTargets,
        std::vector<std::vector<std::uint64_t>> &resultHoles) const override;

    /// The result block (window R) EMERGES from the three pinned inputs; the merge
    /// pins inputs alone and reads the emergent result, so no outputStates/U.
    [[nodiscard]] bool emergesResult() const override { return true; }

    [[nodiscard]] std::size_t carriedDim(std::size_t /*stateDim*/) const override {
      return 2;  // the result lives in the S_3 standard rep (Sigma=0, b1 window)
    }

    [[nodiscard]] std::size_t loopsPerState() const override {
      return 3;  // the three color holes per register window
    }

    /// The color register is \f$ d = 3 \f$ (the color triple on \f$ \sum=0 \f$).
    void validateStateDim(std::size_t d) const override;

    [[nodiscard]] std::string name() const override {
      return "tripartite register ((S^2-12holes) x I, 4 windows A,B,C->R)";
    }

  private:
    // Cached by build() for readoutHoles(): the four windows (A,B,C,R) of three
    // color holes each (sorted absolute vertex triples), and their per-hole
    // induced-orientation signs (ChainComplex::endSignCovector of the base
    // surface). Windows live on ONE surface, so holes carry absolute ids (no
    // per-block stride offset, unlike RegisterTopology's layered blocks).
    std::vector<std::vector<std::vector<std::uint64_t>>> blockHoles_{};
    std::vector<std::vector<int>> signTable_{};
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_TRIPARTITEREGISTERTOPOLOGY_H
