// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_S3WINDOWSURFACE_H
#define TESSERA_COBORDISM_S3WINDOWSURFACE_H

#include <cstdint>
#include <vector>

namespace tessera::cobordism {

/// # S3WindowSurface
///
/// The genuinely **3+1 D** color-register base slice (#453): a triangulated
/// 3-sphere \f$ S^3 \f$ carrying symmetric, color-\f$ \mathbb{Z}_3 \f$-equivariant
/// windows --- the faithful 4D analog of the \f$ S^2 \f$ `SymmetricWindowSurface`
/// (the geodesic icosahedron), NOT a 2+1 D shortcut.
///
/// ## The dimensional lift (the crux)
/// On \f$ S^2 \f$ the color register is \f$ \ker L_1 \f$ (\f$ b_1 \f$) and a window
/// hole is a removed **top 2-cell** (a triangle), whose boundary 1-loop becomes a
/// 1-cycle (read as a 1-form period at degree \f$ k=1 \f$). The register degree
/// **tracks the spatial dimension**: it is always \f$ \ker L_{d-1} \f$ with holes
/// the removed **top \f$ d \f$-cells**. On \f$ S^3 \f$ (\f$ d=3 \f$) the register is
/// therefore \f$ \ker L_2 \f$ (\f$ b_2 \f$) and a window hole is a removed **top
/// 3-cell** (a tetrahedron), whose boundary 2-sphere becomes a 2-cycle (read as a
/// 2-form period at degree \f$ k=2 \f$). \f$ S^3 \f$ is simply connected
/// (\f$ b_1=0 \f$), so removing tetrahedra cannot make \f$ b_1 \f$ --- the register
/// genuinely lives one degree up, exactly mirroring the \f$ S^2 \f$ construction one
/// dimension down. Removing \f$ n \f$ vertex-disjoint tetrahedra from \f$ S^3 \f$
/// gives \f$ b_2 = n-1 \f$ (the \f$ S^2 \f$ analog: \f$ n \f$ disjoint triangles
/// give \f$ b_1 = n-1 \f$).
///
/// ## The triangulation and the symmetric windows
/// The slice is the **join of two \f$ K \f$-cycles** \f$ C_K * C_K \f$ --- a clean
/// triangulated \f$ S^3 \f$ (Betti \f$ [1,0,0,1] \f$) with \f$ 2K \f$ vertices and
/// \f$ K^2 \f$ top tetrahedra \f$ \{a_i, a_{i+1}, b_j, b_{j+1}\} \f$. Unlike a
/// product/shuffle triangulation, the join is manifestly cyclically symmetric: the
/// rotations \f$ \tau:\,a_i\!\to\!a_{i+1},\,b_j\!\to\!b_{j+1} \f$ (and the
/// rotate-one-factor variants) are simplicial automorphisms. Each **window** is a
/// \f$ \mathbb{Z}_3 \f$ orbit (under \f$ \sigma = \tau^{K/3} \f$) of three
/// vertex-disjoint hole tetrahedra; \f$ \sigma \f$ cyclically permutes the three
/// color holes (the color \f$ \mathbb{Z}_3 \f$), so the transport intertwines color
/// just as the \f$ A_4 \f$ orbits did on \f$ S^2 \f$. The `windowCount` windows are
/// themselves one orbit of \f$ \tau^2 \f$ (the seed of window \f$ w \f$ is shifted
/// by two), so the windows are a genuine symmetry orbit, not a hand-placed set.
/// With \f$ K = 6\cdot\text{windowCount}\cdot\text{granularity} \f$ the
/// \f$ 3\cdot\text{windowCount} \f$ holes are all vertex-disjoint, giving
/// \f$ b_2 = 3\cdot\text{windowCount} - 1 \f$ (windowCount\f$=4 \Rightarrow b_2=11 \f$,
/// the \f$ S^2 \f$ proton's hole budget lifted to \f$ S^3 \f$).
///
/// Pure and deterministic --- a function of (`windowCount`, `granularity`) alone.
class S3WindowSurface {
  public:
    /// A sorted vertex tuple: a top tetrahedron (4 ids) or a hole tetrahedron.
    using Cell = std::vector<std::uint64_t>;

    /// The slice (its \f$ K^2 \f$ top tetrahedra, sorted) and the `windowCount`
    /// \f$ \mathbb{Z}_3 \f$-symmetric windows of three hole tetrahedra each. A 4D
    /// `EmergentEventTopology` (S^3 mode) consumes all the windows (A,B,C,R); the
    /// hole tetrahedra are the **top 3-cells removed** to open the \f$ b_2 \f$
    /// register, read at degree \f$ k=2 \f$.
    struct Surface {
      std::vector<Cell> faces;                  ///< K^2 top tetrahedra (sorted)
      std::vector<std::vector<Cell>> windows;   ///< windowCount x 3 hole tetrahedra
    };

    /// Build the \f$ S^3 \f$ slice + windows.
    /// @param windowCount the number of color windows (\f$ \ge 1 \f$). Four matches
    ///   the \f$ W_{ABC} \f$ A,B,C,R structure; one is the minimal color register.
    /// @param granularity the lattice-refinement factor (\f$ \ge 1 \f$): the cycle
    ///   length is \f$ K = 6\cdot\text{windowCount}\cdot\text{granularity} \f$, so a
    ///   larger value refines the triangulation between the (fixed, disjoint) holes
    ///   (the tunable granularity, the \f$ S^3 \f$ analog of the geodesic frequency).
    /// @throws std::invalid_argument if `windowCount < 1` or `granularity < 1`.
    /// @throws std::runtime_error if the generated windows are not a genuine
    ///   \f$ \mathbb{Z}_3 \f$ orbit, are not base tetrahedra, or are not
    ///   vertex-disjoint.
    [[nodiscard]] static Surface build(int windowCount = 4, int granularity = 1);
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_S3WINDOWSURFACE_H
