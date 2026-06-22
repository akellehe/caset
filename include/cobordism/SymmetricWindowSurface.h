// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_SYMMETRICWINDOWSURFACE_H
#define TESSERA_COBORDISM_SYMMETRICWINDOWSURFACE_H

#include <cstdint>
#include <vector>

namespace tessera::cobordism {

/// # SymmetricWindowSurface
///
/// The shared color-register base surface (#398): a frequency-\f$ N \f$ geodesic
/// icosahedron (\f$ S^2 \f$) carrying the **four** \f$ A_4 \f$-tetrahedral,
/// \f$ C_3 \f$-symmetric windows of three vertex-disjoint hole triangles each. The
/// four windows are one orbit of a tetrahedral subgroup \f$ A_4 \f$ of the
/// icosahedral rotation group --- each window a \f$ C_3 \f$ orbit of three corner
/// sub-triangles seated at one of the icosahedron's four tetrahedral vertex-orbits
/// (\f$ \{2,8,10\},\{1,4,7\},\{0,6,9\},\{3,5,11\} \f$). The windows are
/// \f$ A_4 \f$-equivalent, so the per-window period-transport blocks are cyclically
/// related (the transport intertwines the color \f$ \mathbb{Z}_3 \f$) and a
/// color-symmetric input transports to the EXACT singlet with manifest \f$ S_3 \f$.
///
/// This is the geometry both `TripartiteRegisterTopology` (the trivalent
/// \f$ W_{ABC} \f$ junction: windows A,B,C inputs \f$ \to \f$ R emergent result) and
/// `BipartiteCreationTopology` (the q/\f$ \bar q \f$ creation split: window 0 seed
/// \f$ \to \f$ windows 1,2 emergent q,\f$ \bar q \f$) share, so the validated
/// symmetric-window construction lives in one place rather than being duplicated.
/// Pure and deterministic --- a function of \f$ N \f$ alone.
class SymmetricWindowSurface {
  public:
    /// A sorted vertex triple (a face or a hole triangle).
    using Face = std::vector<std::uint64_t>;

    /// The surface (its \f$ 20 N^2 \f$ sub-triangle faces, sorted) and the four
    /// \f$ A_4 \f$-symmetric windows of three hole triangles each, in the canonical
    /// order A,B,C,R (a `TripartiteRegisterTopology` consumes all four; a
    /// `BipartiteCreationTopology` punches the first three).
    struct Surface {
      std::vector<Face> faces;                  ///< 20*N^2 sub-triangles (sorted)
      std::vector<std::vector<Face>> windows;   ///< 4 windows x 3 holes (A,B,C,R)
    };

    /// Build the surface + windows for geodesic subdivision frequency \f$ N \f$.
    /// @param frequency the subdivision frequency \f$ N \ge 2 \f$ (\f$ N=2 \f$ is the
    ///   42-vertex/80-face base of #398, enough to host the 12 disjoint holes; larger
    ///   N refines the lattice, #404).
    /// @throws std::runtime_error if the generated windows are not a genuine
    ///   \f$ C_3 \f$ orbit, are not base faces, or are not vertex-disjoint.
    [[nodiscard]] static Surface build(int frequency);
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_SYMMETRICWINDOWSURFACE_H
