// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_SLICEDCOBORDISM_H
#define TESSERA_COBORDISM_SLICEDCOBORDISM_H

#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tessera::spacetime { class Spacetime; }

namespace tessera::cobordism {
using ::tessera::spacetime::Spacetime;

/// # SlicedCobordism
///
/// The two-phase **boundary-then-bulk** construction (#614, #615): discover a
/// spatial slice as a closed 3-complex first, then cone it through **timelike**
/// edges to obtain a 4-complex with genuine temporal extent.
///
/// ## Why two phases
///
/// Every host in this project has been a 4-complex from its first cell (a single
/// pentatope), so the spatial slice and the spacetime it bounds were never
/// distinguished. Measured consequence: on canonically built hosts **every edge
/// is spacelike** and \f$ \mathrm{Im}\,S = 0 \f$ at every stage of the drive.
/// `Spacetime::fromCells` starts every edge at \f$ \ell^2 = +1 \f$, and nothing
/// in the drive seeds causal content, so a spacelike start stays spacelike. The
/// bulk exists but is reached spacelike-ly, which is not what a Lorentzian
/// cobordism \f$ \partial W = M_0 \sqcup M_1 \f$ should look like.
///
/// Splitting the construction fixes that at its origin rather than by fiat:
///
///   * **Phase 1** (`closedSlice`) is a closed 3-complex — a spatial slice, all
///     edges spacelike, \f$ \mathrm{Im}\,S = 0 \f$ correct rather than
///     symptomatic. Its register is \f$ \ker L_{d-1} \f$ at \f$ d = 3 \f$, i.e.
///     degree \f$ k = 2 \f$ read as \f$ b_2 \f$, with holes made by **removing**
///     tetrahedra (`SurgicalCone::coneOut`) — a closed complex has no boundary
///     facets for `MultiCobordism::emergentHoles` to read. This is the #453
///     \f$ S^3 \f$ window setting.
///   * **Phase 2** (`coneToBulk`) joins every tetrahedron of the slice to a
///     **single shared apex** through timelike edges, giving one 4-simplex per
///     tetrahedron: \f$ \mathrm{cone}(S^3) = D^4 \f$, a genuine
///     4-manifold-with-boundary whose boundary is the original slice. Its
///     register is degree \f$ k = 3 \f$ (\f$ b_3 \f$), and the Regge term
///     becomes meaningful because there is now 4D bulk for it to act on.
///
/// ## Why a shared apex and not one per tetrahedron
///
/// A 4D CDT slab between consecutive slices needs **both** \f$ (4,1) \f$ and
/// \f$ (3,2) \f$ simplices — four vertices on one slice with one on the next,
/// *and* three with two. Coning each tetrahedron to its own apex produces only
/// \f$ (4,1) \f$ cells; they leave gaps and do not tile a manifold. The shared
/// apex tiles exactly. Its price is a conical singularity at the apex — the whole
/// slice converges to one point — so this is a valid **first bulk step**, not a
/// time *layer*. A genuine layer needs the \f$ (3,2) \f$ cells and is separate
/// work.
///
/// ## What this class does not do
///
/// It constructs; it does not police. The timelike dispositions it seeds are free
/// to be driven spacelike by the geometric relaxation, and nothing here or
/// elsewhere prevents that — a runtime guard on the dynamics is exactly what this
/// project does not do. Whether they survive is a **measurement**, and either
/// answer is a result.
///
/// It also changes nothing that already exists. `Proton`, `ProtonIngredients` and
/// `MultiCobordism` are untouched; a caller opts into this construction or does
/// not.
class SlicedCobordism {
  public:
    /// The default timelike squared length for apex edges, in the CDT convention
    /// \f$ \ell_t^2 = -\alpha\,\ell_s^2 \f$ with \f$ \alpha = 1 \f$ and unit
    /// spacelike edges. Negative real: `Edge` treats a negative real squared
    /// length as a pure-imaginary length, which is what `isTimelike` reads.
    static constexpr double kDefaultTimelikeSquaredLength = -1.0;

    /// Phase 1 seed: the closed 3-complex \f$ \partial \Delta^4 \cong S^3 \f$ —
    /// five tetrahedra on five vertices, every 4-subset of \f$ \{0..4\} \f$.
    ///
    /// Closed by construction (each triangle lies in exactly two tetrahedra), so
    /// it is a spatial slice rather than a 3-ball, which is what `coneToBulk`
    /// requires. Every edge is spacelike. Combinatorial dimension 3.
    [[nodiscard]] static std::shared_ptr<Spacetime> closedSlice();

    /// Phase 2: cone every top cell of the closed 3-complex \p slice to one fresh
    /// shared apex, joined by edges of squared length \p apexEdgeSquaredLength
    /// (timelike by default), yielding a 4-complex.
    ///
    /// `SurgicalCone::coneIn` cannot serve here: it mints a **fresh** apex on
    /// every call, so looping it would give one apex per tetrahedron — the
    /// \f$ (4,1) \f$-only configuration that does not tile. `SurgicalCone` is
    /// deliberately left untouched; broadening a general surgical primitive to
    /// carry this construction's concern would be the wrong place for it. The
    /// coned cell set is therefore assembled directly and put through the **same**
    /// gate every surgical move uses, `ChainComplex::dualComplexIsValid`, so
    /// nothing is inserted by fiat.
    ///
    /// Spacelike edge lengths are carried over from \p slice unchanged; only the
    /// apex-incident edges are written.
    ///
    /// @return `(bulk, reason)`. `reason` is `"ok"` on success, in which case
    ///         `bulk` has combinatorial dimension 4 and one 4-simplex per
    ///         tetrahedron of the slice. On failure `bulk` is null.
    [[nodiscard]] static std::pair<std::shared_ptr<Spacetime>, std::string>
    coneToBulk(const std::shared_ptr<Spacetime> &slice,
               std::complex<double> apexEdgeSquaredLength =
                   std::complex<double>(kDefaultTimelikeSquaredLength, 0.0));

    /// The top-cell vertex tuples of \p complex, each sorted ascending — the same
    /// canonical form `MultiCobordism` snapshots use.
    [[nodiscard]] static std::vector<std::vector<std::uint64_t>> topCells(
        const Spacetime &complex);
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_SLICEDCOBORDISM_H
