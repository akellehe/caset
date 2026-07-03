// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_LIVE_COMPLEX_H
#define TESSERA_OBSERVABLES_LIVE_COMPLEX_H

#include <complex>
#include <cstdint>
#include <map>
#include <memory>
#include <utility>
#include <vector>

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime {
  class Spacetime;
}
namespace tessera::observables {
using namespace ::tessera::spacetime;

/// # LiveComplex
///
/// The loader / transform layer that lives OUTSIDE the pure readers (#593, owner
/// directive): it LOADS a saved combinatorial + metric description back into a
/// LIVE, skeleton-complete `Spacetime`, and produces a relabeled copy for the
/// RELABEL gate. It NEVER builds a spacetime of its own or re-runs the emergent
/// dynamics — those live exclusively in Proton / ProtonIngredients /
/// MultiCobordism. `LiveComplex` only ever reads a recorded geometry back through
/// the canonical `Spacetime::fromCells` entry point:
///
///   * `Spacetime::fromCells` materializes ONLY the top cells (measured: `∂Δ⁵`
///     comes back as its 6 pentatopes and nothing else). The facet/coface
///     skeleton that `dualVolume()` / `lorentzianDeficitAngle()` walk is then
///     completed with the honest direct call `Spacetime::materializeFacets()` —
///     never a solver-named one — which reproduces the `ReggeSolver` +
///     `ChainComplex::fromSpacetime` skeleton bit-for-bit (verified: interior
///     hinge census, `V_dual`, and `m_sum` all agree).
///   * The metric (complex squared lengths) and per-vertex times are loaded back
///     exactly as recorded; a missing edge length throws (a partial metric is
///     never silently defaulted).
///
/// The RegisterContext and every Observable then only ever READ the resulting
/// live complex — they never see a dump and never build anything themselves.
class LiveComplex {
  public:
    /// A relabeled rebuild: the live relabeled complex plus the vertex-id
    /// permutation that produced it (original id → relabeled id), so
    /// vertex-id-bearing observable configuration (e.g. provenance block
    /// regions) can be mapped through it.
    struct Relabeled {
      std::shared_ptr<Spacetime> spacetime;
      std::map<std::uint64_t, std::uint64_t> vertexMap;
    };

    /// Load a live, skeleton-complete complex from explicit top cells + per-edge
    /// complex squared lengths (the schema-1 geometry-dump rehydration core, and
    /// the RELABEL rebuild core). This is a LOAD, not a build: the geometry is
    /// supplied wholesale and only read back through `Spacetime::fromCells`.
    /// `cells` keep their intrinsic vertex order (never sorted — the stored order
    /// carries the orientation); `squaredLengths` maps each `(min id, max id)`
    /// vertex pair to its complex squared length; `vertexTimes` (may be empty)
    /// maps vertex id → recorded time, applied before the lengths. `dimensions`
    /// is the recorded complex dimension (the schema-1 dump's own `dimensions`
    /// field / the source complex's canonical dimension) passed straight through
    /// to `fromCells`, never guessed from a cell. The facet skeleton is completed
    /// with `materializeFacets()` so the resulting complex is immediately
    /// readable.
    /// @throws std::invalid_argument if `cells` is empty.
    /// @throws std::out_of_range if a built edge has no recorded squared
    ///   length — a partial metric is never silently defaulted.
    [[nodiscard]] static std::shared_ptr<Spacetime> load(
        const std::vector<std::vector<std::uint64_t>> &cells,
        const std::map<std::pair<std::uint64_t, std::uint64_t>,
                       std::complex<double>> &squaredLengths,
        const std::map<std::uint64_t, double> &vertexTimes, int dimensions);

    /// LOAD (never build) the block-residual sub-complex: `cells` are ambient
    /// top cells already SELECTED by the caller (the strict subset whose vertices
    /// all lie in a provenance region — no new topology, no surgery, no
    /// dynamics), re-instantiated through the canonical `Spacetime::fromCells`
    /// with a uniform metric (weight 1.0). The uniform metric is the DEFINITION
    /// of the carry diagnostic — it mirrors how the drive's `r_U` scored the
    /// block (metric-independent by design) — so this is byte-identical to the
    /// canonical `MultiCobordism::subcomplexWithinVertexSet` (lifted here so the
    /// reader can score a block without constructing the build driver). No
    /// emergent build; the skeleton is NOT materialized (the `r_state` read this
    /// feeds builds only what it needs). `dimensions` is the ambient complex's
    /// canonical dimension (the caller reads `RegisterContext::dimensions()`);
    /// it is passed straight through to `fromCells`, never guessed from a cell.
    /// @throws std::invalid_argument if `cells` is empty.
    [[nodiscard]] static std::shared_ptr<Spacetime> subcomplex(
        const std::vector<std::vector<std::uint64_t>> &cells, int dimensions);

    /// A relabeled rebuild of `spacetime` under a random vertex-id permutation
    /// (deterministic given `seed`), with the cell enumeration order shuffled too
    /// (catching enumeration-order dependence), carrying the metric and vertex
    /// times across. It reads the recorded geometry off `spacetime` and re-loads
    /// it under the permutation (via `load`). The permutation need only be a
    /// genuine relabeling for the RELABEL gate to compare like with like — the
    /// specific permutation is not part of any reproduced value — so a
    /// `std::mt19937_64` stream suffices (this deliberately does NOT reproduce
    /// the Python framework's NumPy permutation; the gate invariance holds
    /// identically either way).
    [[nodiscard]] static Relabeled relabel(const Spacetime &spacetime,
                                           std::uint64_t seed);
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_LIVE_COMPLEX_H
