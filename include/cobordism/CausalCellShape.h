// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_CAUSALCELLSHAPE_H
#define TESSERA_COBORDISM_CAUSALCELLSHAPE_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tessera::mesh { class Simplex; }
namespace tessera::spacetime { class Spacetime; }

namespace tessera::cobordism {
using ::tessera::mesh::Simplex;
using ::tessera::spacetime::Spacetime;

/// # CausalCellShape
///
/// Reads the **causal shape** of top cells (#620): how each cell's vertices split
/// across the temporal direction, and whether the cell has a consistent Lorentzian
/// geometry at all. Both are read-only measurements composed from `Simplex`'s
/// existing helpers; nothing here modifies geometry.
///
/// ## Primal or dual?
///
/// **Cell shape and admissibility are properties of the PRIMAL complex** — a
/// simplex's own vertices, its own edges, its own Gram matrix. The dual is where
/// the Lorentzian spacetime lives: the action \f$ S = \sum_h |\star h|\,
/// \varepsilon_h \f$ is a sum over dual volumes, and it is dual separations that
/// carry the causal character the dynamics feels. The primal complex is
/// bookkeeping for it.
///
/// So a primal cell shape is not itself the physics, and no primal shape statistic
/// should be read as one. It is a **diagnostic**: the primal bipartition is the
/// combinatorial precondition for a foliation-like dual structure, and its absence
/// says the complex has not developed one. `dualHeightCensus` is the measurement
/// that reaches the dual directly, and it is the one whose split between mesh
/// defect and timelike separation says something about \f$ S \f$.
///
/// ## Cell shape (primal)
///
/// A 4-simplex has 5 vertices and \f$ \binom{5}{2} = 10 \f$ edges. If its vertices
/// split into groups of sizes \f$ a \f$ and \f$ b \f$ with \f$ a + b = 5 \f$, then
/// exactly \f$ a\,b \f$ edges cross between the groups. Naming the shapes by that
/// split:
///
///   * \f$ (5,0) \f$ — entirely spacelike, **0** timelike edges;
///   * \f$ (4,1) \f$ — four vertices against one, **4** timelike edges;
///   * \f$ (3,2) \f$ — three against two, **6** timelike edges.
///
/// **0, 4 and 6 are the only timelike-edge counts a genuine bipartition can
/// produce**, so a cell carrying (say) one or two timelike edges corresponds to no
/// consistent temporal splitting at all. Counting timelike edges is therefore not
/// sufficient to classify a cell: `classify` **verifies** that the timelike edges
/// are exactly the crossing set of some bipartition, and reports `NonBipartite`
/// when they are not.
///
/// This distinction is what explained the earlier attempts. With per-edge
/// disposition moves, no \f$ (3,2) \f$ cell ever formed, \f$ (4,1) \f$ appeared at
/// most twice, and most timelike-carrying cells were non-bipartite — eighteen
/// cells in one run holding a single timelike edge where \f$ (4,1) \f$ needs four.
///
/// ## Lorentzian admissibility (primal)
///
/// `ChainComplex::dualComplexIsValid` is purely **combinatorial** — facet coface
/// counts, ridge links, recursive vertex-link checks — and never reads
/// \f$ \ell^2 \f$. Every complex that reached \f$ F \approx 6\times10^{300} \f$ in
/// the per-cell experiment reported a valid verdict from it.
/// `Simplex::assertSpacelikeAdmissible` covers the purely spacelike case but
/// explicitly **skips** any simplex containing a null or timelike edge, since such
/// a cell's admissibility is governed by Lorentzian structure rather than the
/// spacelike triangle inequalities — and that criterion was unimplemented. Under
/// any causal construction essentially every bulk cell contains a timelike edge,
/// so essentially every bulk cell went unchecked.
///
/// `isLorentzianAdmissible` supplies it: the Gram matrix must be non-degenerate
/// with signature \f$ (-,+,+,+) \f$, i.e. exactly **one** timelike direction. The
/// count of negative eigenvalues is read by **Jacobi's criterion** — the number of
/// sign changes in the sequence of leading principal minors \f$ 1, D_1, \ldots,
/// D_d \f$ — which keeps the test Eigen-free and reuses
/// `Simplex::determinant`, exactly as `assertSpacelikeAdmissible` does. A zero
/// minor means a degenerate cell and is inadmissible.
///
/// Note this is a **config-space validity definition**, in the same family as
/// `dualComplexIsValid`: it answers "is this a well-defined Lorentzian simplex",
/// not "should the dynamics be prevented from going here". It belongs at proposal
/// time. No combinatorial move may follow edge relaxation except in a new stage
/// (see `ProtonIngredients::build`'s drive: init, cone-out, evolve, cone-in, then
/// `runStage2`), so nothing here may reach into stage 2.
class CausalCellShape {
  public:
    /// How a top cell's vertices split across the temporal direction.
    enum class Shape {
      Spacelike,      ///< \f$ (5,0) \f$ — no timelike edge.
      FourOne,        ///< \f$ (4,1) \f$ — four vertices against one.
      ThreeTwo,       ///< \f$ (3,2) \f$ — three against two.
      NonBipartite,   ///< timelike edges match no bipartition; no temporal split.
    };

    /// Human-readable name of \p shape, as reported by `distribution`.
    [[nodiscard]] static std::string shapeName(Shape shape);

    /// The causal shape of \p cell. Verifies bipartiteness rather than inferring
    /// it from the timelike-edge count: a cell is `FourOne` or `ThreeTwo` only when
    /// its timelike edges are **exactly** the crossing set of the corresponding
    /// vertex split, and `NonBipartite` otherwise.
    [[nodiscard]] static Shape classify(const Simplex &cell);

    /// Counts of each shape over the top cells of \p spacetime, indexed by the
    /// `Shape` enumerator's underlying value.
    [[nodiscard]] static std::vector<int> distribution(
        const Spacetime &spacetime);

    /// Whether \p cell has a consistent Lorentzian geometry: Gram matrix
    /// non-degenerate with exactly one timelike direction, signature
    /// \f$ (-,+,+,+) \f$. A purely spacelike cell is admissible when its Gram
    /// matrix is positive-definite (zero timelike directions), which is the
    /// existing `assertSpacelikeAdmissible` criterion reached through the same
    /// code path.
    ///
    /// @param tolerance Magnitude below which a leading principal minor counts as
    ///        zero, marking the cell degenerate and therefore inadmissible.
    [[nodiscard]] static bool isLorentzianAdmissible(const Simplex &cell,
                                                     double tolerance = 1e-12);

    /// Why a dual height carried a negative sign, counted over every
    /// `(simplex, coface)` term of the circumcentric dual recursion.
    ///
    /// Each height is formed as
    /// \f$ h = \mathrm{oppositeVertexSign}(cf, s)\cdot
    ///        \mathrm{sgn}\sqrt{R^2(cf) - R^2(s)} \f$
    /// whose two factors mean unrelated things:
    ///
    ///   * **`centerednessDefects`** — the barycentric coordinate of
    ///     \f$ c(cf) \f$ at the single vertex of \p cf not in \p s is negative,
    ///     so the circumcentre has crossed to the far side of that facet. A mesh
    ///     defect. Note this is **one specific** coordinate, not
    ///     \f$ \min(\text{barycentric}) < 0 \f$ — the weaker condition
    ///     overcounts, since negative-at-opposite implies any-negative but not
    ///     conversely.
    ///   * **`timelikeSeparations`** — the radicand \f$ R^2(cf) - R^2(s) \f$ is
    ///     negative, so the circumcentre separation is timelike. **Correct
    ///     Lorentzian structure in the dual, not a defect**, and the dual is where
    ///     \f$ S = \sum_h |\star h|\,\varepsilon_h \f$ lives.
    ///
    /// Measured on a converged charge-conserving complex the split runs about
    /// 4:1 defect to timelike, so remedial work aimed at "eliminate negative dual
    /// volumes" would destroy roughly a fifth of them wrongly. The predicate worth
    /// targeting is the defect alone.
    struct DualHeightCensus {
      int terms{0};                 ///< total `(simplex, coface)` height terms
      int centerednessDefects{0};   ///< sign flipped by the barycentric factor
      int timelikeSeparations{0};   ///< sign flipped by a negative radicand
      /// Terms whose height is negative — exactly one factor negative.
      int negativeHeights{0};
    };

    /// The `DualHeightCensus` over every non-orphan simplex of \p spacetime below
    /// the top dimension. Top cells are excluded: their dual is a point, so they
    /// contribute no heights.
    [[nodiscard]] static DualHeightCensus dualHeightCensus(
        const Spacetime &spacetime);

    /// The number of timelike directions in \p cell's tangent metric — the count
    /// of negative eigenvalues of its Gram matrix, read by Jacobi's criterion.
    /// Returns -1 when the Gram matrix is degenerate to within \p tolerance, since
    /// the signature is then undefined rather than zero.
    [[nodiscard]] static int timelikeDirectionCount(const Simplex &cell,
                                                    double tolerance = 1e-12);
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_CAUSALCELLSHAPE_H
