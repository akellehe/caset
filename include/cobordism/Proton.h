// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_PROTON_H
#define TESSERA_COBORDISM_PROTON_H

#include <cstdint>
#include <memory>
#include <set>
#include <vector>

namespace tessera::spacetime {
class Spacetime;
}

namespace tessera::cobordism {

using ::tessera::spacetime::Spacetime;

/// # Proton
///
/// A high-level, **fully emergent** builder for the proton, composing the
/// `MultiCobordism` engine (#503, part of #410). It exists so a user can obtain a
/// converged emergent proton without re-implementing the error-prone build by
/// hand — the colour targets, the construct order, the two-stage optimisation, the
/// restart loop, and the relaxed-metric block carve are all encapsulated.
///
/// ## Everything emerges — nothing is hand-placed
/// The colour register (the three quark "holes") is **grown by gated surgical
/// coning** on a bare host; **no topology is seeded**. The host is a bare closed
/// \f$ S^4 \f$ (`SimplexBoundarySphere(4)` refined by `PreGeometric` Pachner moves
/// purely to give surgery room), with **no** windows, holes, or register placed in
/// it. The register that carries the colour singlet emerges from the optimisation
/// alone (`MultiCobordism`'s `coneOut`/`coneIn`, kept only by the objective and the
/// `dualComplexValid` manifold gate).
///
/// ## The proton is built in two steps (a proton is three quarks)
/// A proton cannot be solved for in a single step — it forms via a diquark. With
/// \f$ \omega = e^{2\pi i/3} \f$:
///   * **Step A (recombination):** two neutral \f$ q\bar q \f$ pairs
///     `{1,-1,0}`, `{1,0,-1}` form a **diquark** `{1,ω}` and an **antidiquark**
///     `{1,ω²}`. A diquark is *coloured* (an \f$ SU(3) \f$ \f$ \bar 3 \f$), hence a
///     **2-vector**, never the singlet.
///   * **Step B (formation):** the diquark `{1,ω}` and the third quark `{ω²}` form
///     the **proton** `{1,ω,ω²}` — the colourless colour singlet (a 3-vector;
///     \f$ 1+\omega+\omega^2 = 0 \f$). Step B's output block is the proton.
///
/// The colour singlet `{1,ω,ω²}` is therefore **only** the step-B output; the
/// diquark is its own coloured 2-vector. Steps A and B are separate `MultiCobordism`
/// runs (the diquark *state* feeds B as an input target — compose by result-state,
/// not by welding interiors).
///
/// ## Convergence is emergent and stochastic
/// Because the register emerges from random gated surgery, a given seed may not
/// grow the full three-hole proton. `Proton` therefore **restarts across seeds**
/// (`nAttempts`) until step B's proton block carries the singlet with at least
/// three emergent holes. The number of attempts and the per-stage step budgets are
/// **user-controlled** (see the constructor).
class Proton {
 public:
  /// Build a converged emergent proton.
  ///
  /// @param nAttempts   how many distinct seeds to try (the restart budget) before
  ///                    giving up. Each attempt is one full two-step build. The
  ///                    register growth is stochastic, so more attempts raise the
  ///                    chance of a converged three-hole proton.
  /// @param stage1Steps the maximum number of **stage-1** (combinatorial / gated
  ///                    surgical-coning) steps per `MultiCobordism` run.
  /// @param stage2Steps the maximum number of **stage-2** (geometric Regge
  ///                    relaxation) iterations per `MultiCobordism` run.
  /// @param nRefine     bare-host refinement (Pachner add-moves) — surgery room; it
  ///                    seeds **no** register, only host volume.
  /// @param gamma       the realizability (`r_U`) weight in the `MultiCobordism`
  ///                    objective \f$ \lVert\nabla S\rVert^2 + \gamma\,r_U \f$.
  /// @param seed0       the first seed; attempts use `seed0, seed0+1, ...`.
  explicit Proton(int nAttempts = 40, int stage1Steps = 80, int stage2Steps = 20,
                  int nRefine = 18, double gamma = 1.0, std::uint64_t seed0 = 1);

  /// Whether a converged proton was found within `nAttempts` (the singlet carried
  /// on a \f$\ge 3\f$-hole proton block).
  [[nodiscard]] bool converged() const noexcept { return converged_; }

  /// The seed of the converged build (or the last seed tried if none converged).
  [[nodiscard]] std::uint64_t seed() const noexcept { return seed_; }

  /// How many attempts were consumed (\f$\le\f$ `nAttempts`).
  [[nodiscard]] int attempts() const noexcept { return attemptsUsed_; }

  /// The full step-B cobordism (the proton formation), or `nullptr` if not
  /// converged.
  [[nodiscard]] std::shared_ptr<Spacetime> spacetime() const noexcept { return st_; }

  /// The **proton block** sub-complex — the proton output block carved out with the
  /// **relaxed metric copied in** (not the unit-metric `subOf`). This is the object
  /// downstream observable reads (charge, flavour, spin) act on. `nullptr` if not
  /// converged.
  [[nodiscard]] std::shared_ptr<Spacetime> block() const noexcept { return block_; }

  /// The three (or more) emergent quark holes of the proton block (each a sorted
  /// vertex-id tuple of a removed top cell), at register degree \f$ k=3 \f$.
  [[nodiscard]] const std::vector<std::vector<std::uint64_t>> &quarkHoles()
      const noexcept {
    return holes_;
  }

  /// The colour residual \f$ r_U \f$ of the singlet `{1,ω,ω²}` on the proton block
  /// (\f$\approx 0\f$ ⇒ the singlet is carried — confinement).
  [[nodiscard]] double colorResidual() const noexcept { return colorResidual_; }

  /// The diquark residual \f$ r_U \f$ from step A (\f$\approx 0\f$ ⇒ step A formed a
  /// valid diquark), for confirming the first formation step converged.
  [[nodiscard]] double diquarkResidual() const noexcept { return diquarkResidual_; }

 private:
  /// Build a bare emergent host: a closed \f$ S^4 \f$ (`SimplexBoundarySphere(4)`)
  /// refined by `nRefine` `PreGeometric` add-moves for surgery room. No register.
  static std::shared_ptr<Spacetime> buildHost(int nRefine, std::uint64_t seed);

  /// Carve a boundary block's own sub-complex (cells whose vertices all lie in
  /// `verts`), copying the parent's **relaxed** edge metric (a unit-metric rebuild
  /// would make downstream geometric reads degenerate). `nullptr` if < 2 cells.
  static std::shared_ptr<Spacetime> carveBlock(
      const std::shared_ptr<Spacetime> &st, const std::set<std::uint64_t> &verts);

  bool converged_ = false;
  std::uint64_t seed_ = 0;
  int attemptsUsed_ = 0;
  std::shared_ptr<Spacetime> st_;
  std::shared_ptr<Spacetime> block_;
  std::vector<std::vector<std::uint64_t>> holes_;
  double colorResidual_ = 0.0;
  double diquarkResidual_ = 0.0;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_PROTON_H
