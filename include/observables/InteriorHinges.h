// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_INTERIOR_HINGES_H
#define TESSERA_OBSERVABLES_INTERIOR_HINGES_H

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime {
  class Spacetime;
}
namespace tessera::observables {
using namespace ::tessera::spacetime;

/// # InteriorHinges
///
/// The shared 4D mass/radius reader core (#566/#593) — the #451 geometric-proton
/// methodology ported to a genuinely 4D emergent interior, composed (never
/// re-derived) by `EmergentMass` and `EmergentRadius`. Everything here is a
/// post-hoc reader; nothing shapes the lattice.
///
///   * **Hinges are TRIANGLES.** On a `d = 4` complex the Regge hinges are the
///     `(d-2) = 2`-simplices; curvature is the complex Lorentzian deficit angle.
///   * **Closed fans only.** A triangle carries honest curvature only if every
///     tetrahedron of its coface fan is shared by exactly two 4-cells. Open-fan
///     triangles (the `∂W` boundary, the register-hole walls) are boundary
///     artefacts near 2π, excluded, and counted in the census. The fan test is
///     combinatorial over the CURRENT top cells, so Pachner-orphaned
///     sub-simplices never pollute the selection; the readings themselves come
///     off the canonical registered triangle `Simplex` (skeleton required — the
///     `RegisterContext` constructor materializes it C++-side, the #451 lesson).
///   * **Signature-aware readings.** Dual volumes are the circumcentric signed
///     `Simplex::dualVolume`; the deficit is the complex
///     `Simplex::lorentzianDeficitAngle`. Masses use Re ε; |Im ε| (boost
///     content) is always reported, never dropped.
///   * **Dimension-correct radius.** `r = V^{1/4}` on a 4-complex — the root
///     tracks the top dimension.
///
/// Constructed once per (spacetime, holes); the constructor throws
/// `std::invalid_argument` if the complex is not genuinely 4D (5-vertex top
/// cells) — on a `d`-complex the hinge dimension and the radius root both track
/// `d`, so a mismatched reader must refuse rather than read nonsense.
class InteriorHinges {
  public:
    /// One interior (closed-fan) triangle hinge and its curvature.
    struct Hinge {
      std::vector<std::uint64_t> vids;  ///< the 3 sorted vertex ids
      double re = 0.0;                  ///< Re of the complex deficit
      double im = 0.0;                  ///< Im of the complex deficit (boost)
      double dv = 0.0;                  ///< signed circumcentric dual content
      std::optional<int> shell;         ///< BFS distance from the holes (empty
                                        ///< when no holes were given)
    };

    /// The interior/boundary hinge census (reported with every reading).
    struct Census {
      int nTops = 0;
      int nTets = 0;
      int nHingesTotal = 0;
      int nHingesInterior = 0;
      int nHingesBoundary = 0;
      int nBoundaryTets = 0;
      int nHoleVertices = 0;
      std::vector<std::vector<std::uint64_t>> boundaryTets;  ///< vertex-id sets
    };

    /// The three #451 mass readings — one intensive, two extensive — plus the
    /// per-shell means and the imaginary-part accounting.
    struct Masses {
      double mShell = 0.0;   ///< intensive: Σ over BFS shells of the shell-mean
                             ///< Re-deficit (plain mean Re with no holes)
      double mSum = 0.0;     ///< extensive: Σ Re ε
      double mAction = 0.0;  ///< extensive: Σ |★h|·Re ε
      /// per-shell mean Re-deficit, ordered shell-ascending with the unshelled
      /// bin last (`std::nullopt`).
      std::vector<std::pair<std::optional<int>, double>> shellMeans;
      double maxAbsIm = 0.0;
      int nImNonzero = 0;
      bool empty = true;  ///< no interior hinges — every scalar is NaN
    };

    /// The emergent size, dual and primal.
    struct Radii {
      double vDual = 0.0;    ///< Σ |★v| over strictly interior vertices
      double vPrimal = 0.0;  ///< Σ |V₄| over all top 4-cells
      int nInteriorVertices = 0;
      double rDual = 0.0;    ///< V_dual^{1/4} (NaN when V_dual ≤ 0)
      double rPrimal = 0.0;  ///< V_primal^{1/4} (NaN when V_primal ≤ 0)
    };

    /// Per-shell curvature profile entry.
    struct ShellProfile {
      int n = 0;
      double meanRe = 0.0;
      double weightShare = 0.0;
    };

    /// Is the curvature a localized lump or spread out?
    struct Localization {
      double pr = 0.0;             ///< participation ratio of |Re ε·★h| in (0,1]
      double concentration = 0.0;  ///< 1/PR
      double meanRe = 0.0;
      double stdRe = 0.0;
      double stdOverMean = 0.0;
      /// per BFS shell (empty unless every hinge is shelled): the profile.
      std::vector<std::pair<int, ShellProfile>> shellProfile;
      double rmsShellRadius = 0.0;
      double fracWithinShell1 = 0.0;
      bool empty = true;
    };

    /// One r·m combination (`"{r_name} x {m_name}"` -> product).
    struct RmTable {
      std::vector<std::pair<std::string, double>> combos;  ///< 6 entries
      double spreadMin = 0.0;
      double spreadMax = 0.0;
      double physical = 0.0;  ///< the physical anchor m_p·r_p/ħc ≈ 4.0
    };

    /// Physical anchor: m_p·r_p/ħc = 938 MeV · 0.84 fm / 197 MeV·fm ≈ 4.0.
    static constexpr double PHYSICAL_RM = 938.0 * 0.84 / 197.0;
    /// |Im ε| above this counts as genuinely complex (boost content).
    static constexpr double IM_TOL = 1e-12;

    /// Select the interior closed-fan triangle hinges of the 4-complex and read
    /// their curvature. `holes` are the register holes' vertex-id tuples — the
    /// BFS shell seeds (empty ⇒ every hinge reports shell None).
    /// @throws std::invalid_argument if the complex has no top cells or its top
    ///   cells are not all 5-vertex (genuinely 4D).
    /// @throws std::runtime_error if an interior triangle has no registered
    ///   `Simplex` — the C++ skeleton was not materialized.
    ///
    /// The spacetime is held `const`: this is a pure reader and the compiler
    /// enforces that it cannot mutate any build/skeleton state — it touches only
    /// the `const` query surface (`getTopSimplices`/`getBoundary`/`getSimplices`
    /// and the `const` geometry methods on the simplices).
    InteriorHinges(std::shared_ptr<const Spacetime> spacetime,
                   std::vector<std::vector<std::uint64_t>> holes);

    [[nodiscard]] const std::vector<Hinge> &hinges() const noexcept {
      return hinges_;
    }
    [[nodiscard]] const Census &census() const noexcept { return census_; }

    /// The three mass readings over the interior hinges.
    [[nodiscard]] Masses masses() const;
    /// The dual/primal size of the interior.
    [[nodiscard]] Radii radii() const;
    /// The curvature localization.
    [[nodiscard]] Localization localization() const;
    /// Every r·m combination (3 masses × 2 radii), the definitional spread
    /// stated first (#451: r·m is too definition-sensitive to quote as one
    /// number).
    [[nodiscard]] RmTable rmTable(const Masses &mass, const Radii &rad) const;

  private:
    std::shared_ptr<const Spacetime> spacetime_;
    std::vector<std::vector<std::uint64_t>> holes_;
    std::vector<Hinge> hinges_;
    Census census_;
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_INTERIOR_HINGES_H
