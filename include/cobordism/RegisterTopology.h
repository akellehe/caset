// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_REGISTERTOPOLOGY_H
#define TESSERA_COBORDISM_REGISTERTOPOLOGY_H

#include <cstdint>
#include <unordered_map>
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

    /// The EXACT triangle-hole read-out (the \#353 period path). The register's
    /// read-out cycles ARE triangle boundaries (the color hole-circles), so it
    /// overrides `readoutHoles()` — never the soft `readout()` (loops) — so the
    /// merge scores inputs over `residualForPeriods` and reads the result block
    /// over `cyclePeriods`, machine-zero on a carried (color-neutral) target.
    void readoutHoles(
        const std::shared_ptr<Spacetime> &cobordism,
        const std::vector<std::vector<std::complex<double>>> &states,
        std::vector<std::vector<std::uint64_t>> &inputHoles,
        std::vector<std::complex<double>> &inputTargets,
        std::vector<std::vector<std::uint64_t>> &resultHoles,
        std::vector<int> &resultSigns) const override;

    /// The #353 result block EMERGES from the pinned neutral-pair inputs (read
    /// after the relax, not supplied), so MergeCobordism may omit outputStates/U.
    [[nodiscard]] bool emergesResult() const override { return true; }

    /// Twist the staircase tube (#416): supply a vertex permutation \f$ \phi \f$ of
    /// the base holed icosahedron, applied **cumulatively** up the layers by
    /// `Spacetime::prismCells` (the mapping-torus twist) — block A at layer 0 is the
    /// untwisted base, B at layer 1 is \f$ \phi \f$, R at layer 2 is \f$ \phi^2 \f$.
    /// The per-block color holes are tracked through the SAME \f$ \phi^\ell \f$ so
    /// `readoutHoles()` follows the twisted tube. An **orientation-reversing** twist
    /// (one whose induced action on the carried color period is sign-reversing) is a
    /// candidate geometric antisymmetrizer onto the diquark \f$ \bar{\mathbf 3} \f$:
    /// inputs A and B enter the shared register through opposite orientations, the
    /// generic symmetric \f$ \mathbf 6 \f$ part cancels. Default: identity (the
    /// generic bipartite merge of `proton_bipartite_obstruction.tex`). Pass an empty
    /// map to clear.
    /// @param twist a partial vertex permutation \f$ \{v : \phi(v)\} \f$ (a missing
    ///   key maps to itself); must be a bijection on the base vertices it touches.
    void setTwist(const std::unordered_map<std::uint64_t, std::uint64_t> &twist);

    /// The canonical orientation-reversing twist of the holed-icosahedron register
    /// (#416), the **exact geometric antisymmetrizer** onto the diquark
    /// \f$ \bar{\mathbf 3} \f$: it reverses the induced orientation of each of the
    /// three color holes (a within-hole vertex transposition, swapping the two
    /// smallest vertices of each), so each carried color period flips sign. It is an
    /// involution (\f$ \phi^2 = \mathrm{id} \f$): the result block R (\f$ \phi^2 \f$)
    /// returns to the base orientation while input B (\f$ \phi \f$) enters through the
    /// reversed one, so on the uniform metric the two input-transport blocks satisfy
    /// \f$ M_B = -M_A \f$ exactly — the symmetric sextet \f$ \mathbf 6 \f$ cancels and
    /// the merge is purely antisymmetric (A\f$ \leftrightarrow \f$B antisymmetric
    /// fraction \f$ = 1 \f$). It preserves the three hole triangles setwise, so the
    /// read-out blocks are the same removed triangles, only relabeled within. A static
    /// helper so the twist travels with the class — no free function.
    [[nodiscard]] static std::unordered_map<std::uint64_t, std::uint64_t>
    orientationReversingTwist();

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
    // The optional staircase twist (#416): a partial vertex permutation phi of the
    // base holed icosahedron, applied cumulatively up the layers by prismCells.
    // Empty => identity (the generic, untwisted bipartite merge).
    std::unordered_map<std::uint64_t, std::uint64_t> twist_{};

    // Cached by build() for readoutHoles(): the per-block color holes (sorted
    // vertex triples), blocks in [A, B, R] order. Tracked through phi^ell when a
    // twist is set, so the read-out follows the twisted tube.
    std::vector<std::vector<std::vector<std::uint64_t>>> blockHoles_{};
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_REGISTERTOPOLOGY_H
