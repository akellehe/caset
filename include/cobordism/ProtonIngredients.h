// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_PROTON_INGREDIENTS_H
#define TESSERA_COBORDISM_PROTON_INGREDIENTS_H

#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

#include "cobordism/Proton.h"

namespace tessera::spacetime { class Spacetime; }

namespace tessera::cobordism {
using ::tessera::spacetime::Spacetime;

class MultiCobordism;  // returned (seeded, not run) by the node factories below

/// # ProtonIngredients
///
/// The **emergent arm** of the proton build (#555). `Proton` is the canonical line in
/// the sand and is composed here **unchanged**; `ProtonIngredients` prepares exactly the
/// same ingredients through exactly the same two-step drive, **except that the final
/// state is never pinned** — step B's `outputTargets` is empty, so the objective is
///
///   `F = ‖∇S_Regge‖² + Γ·Σᵢ r_U(inputᵢ)`
///
/// and whatever the whole cobordism comes to carry is **read** afterwards, never driven.
/// Exactly one variable differs from `Proton::build()` (the singlet output target), so
/// the two classes form a clean A/B experiment.
///
///   * **Step A — recombination**: literally `Proton::recombinationNode` (delegated to a
///     composed `Proton` configured identically): two neutral q-q̄ pairs `{1,-1,0}` ⊔
///     `{1,0,-1}` → a diquark `{1,ω}` ⊔ antidiquark `{1,ω²}`.
///   * **Step B — formation, nothing pinned**: the same ideal diquark `{1,ω}` + third
///     quark `{ω²}` inputs on the same single-Δ⁴ seed as `Proton::formationNode`, but
///     with an **empty output-target list** — no singlet anywhere in the drive.
///
/// The seed stays uniform and all-spacelike (`ℓ² = +1`) **by design**: at initialization
/// no time has passed — causal structure marks sequences of events (in the causal-set
/// sense) and may only *emerge* from the optimization, never be initialized in.
///
/// **Convergence carries no answer-shaped gate** (no color tolerance, no minimum hole
/// count). An attempt converges iff it is
///
///   * **stationary** — step B's `runStage2` stopped on its relative-tolerance
///     stationarity test rather than its iteration budget, and
///   * **persistent** — one further evolution pass (`runStage1`, ∂W frozen) plus
///     relaxation leaves the answer-agnostic summary stable: the emergent hole count and
///     `b_k` unchanged, and `F` within a relative tolerance.
///
/// Everything physical is a **post-hoc observable**: `emergentHoles()`, the final
/// objective, the inputs-only residual, and `singletResidual()` — the singlet `r_state`
/// of `Proton::singlet()` against the whole, reported purely as a **diagnostic** so the
/// emergent result is directly comparable to the canonical build's carried level.
class ProtonIngredients {
 public:
  /// Configure an emergent-arm build. The knobs (and defaults) are `Proton`'s, so the
  /// two arms differ only in what is pinned; see `Proton::Proton` for their meaning.
  explicit ProtonIngredients(std::uint64_t seed = 0, int registerDegree = 3,
                             double gamma = 50.0, double inputWeight = 20.0,
                             int precone = 0, bool shouldUseDirectedSurgery = false);

  /// Build the emergent arm: run step A then step B with `Proton::build()`'s exact
  /// drive (init pass with `grow_boundaries=true`, evolution pass with ∂W frozen,
  /// optional directed cone probes, then `runStage2`), restarting across seeds until an
  /// attempt is **stationary and persistent** (or `maxRestarts` is exhausted — the
  /// lowest-final-`F` attempt is kept and `converged()` is then false). The persistence
  /// pass reuses `evolveSteps`/`stage2MaxIters`, and `persistRelTol` is the relative
  /// `F`-stability tolerance. No color tolerance, no minimum hole count. Idempotent.
  void build(int maxRestarts = 16, int initSteps = 180, int evolveSteps = 60,
             int stage1CandidateMoves = 8, int stage1Patience = 15,
             double stage2Beta = 1.0, int stage2MaxIters = 10,
             double persistRelTol = 0.05);

  /// Step A verbatim: `Proton::recombinationNode` on the composed canonical `Proton` —
  /// the identical seeded (not-yet-run) 2→2 node `Proton::build()` drives.
  [[nodiscard]] std::shared_ptr<MultiCobordism> recombinationNode(std::uint64_t seed) const;
  /// Step B with nothing pinned: the same single-Δ⁴ seed and ideal diquark `{1,ω}` +
  /// third quark `{ω²}` inputs as `Proton::formationNode`, but `outputTargets = {}` —
  /// the final state emerges and is read off the whole afterwards.
  [[nodiscard]] std::shared_ptr<MultiCobordism> formationNode(std::uint64_t seed) const;
  /// The joint inputs-only node (the design note's Rungs 1+2 collapsed to the
  /// inputs-only shape): ONE MultiCobordism whose inputs are the three Z₃-symmetric
  /// neutral q-q̄ pairs `{1,−1,0} ⊔ {0,1,−1} ⊔ {−1,0,1}` (each Σ = 0 — the only
  /// prepared content, held representable through their `r_U` terms for the whole
  /// build) and whose `outputTargets = {}`. No diquark, no bare quark, no
  /// intermediate is ever imposed: the objective is `‖∇S‖² + Γ·Σᵢ w·r_U(inputᵢ)` and
  /// whatever the whole cobordism comes to carry — the pre-registered expectation is
  /// a baryon with a conjugate partner — is READ afterwards (singlet and
  /// conjugate-singlet residuals as diagnostics, never drives). Inputs seeded at
  /// v0/v1/v2 of the single Δ⁴ seed; the two-step nodes above remain the reference
  /// oracle. NOT run (the caller drives it).
  [[nodiscard]] std::shared_ptr<MultiCobordism> jointNode(std::uint64_t seed) const;

  /// True iff the kept attempt was stationary AND persistent (never a statement about
  /// the singlet or the hole count). Triggers `build()`.
  [[nodiscard]] bool converged();
  /// Whether the kept attempt's final `runStage2` stopped on stationarity. Triggers
  /// `build()`.
  [[nodiscard]] bool stationary();
  /// Whether the kept attempt survived the continued evolution+relaxation pass with
  /// holes, `b_k`, and `F` stable. Triggers `build()`.
  [[nodiscard]] bool persistent();
  /// The base seed of the kept attempt. Triggers `build()`.
  [[nodiscard]] std::uint64_t seed();
  /// The full relaxed emergent step-B complex. Triggers `build()`.
  [[nodiscard]] std::shared_ptr<Spacetime> spacetime();
  /// The emergent object IS the whole step-B cobordism (API parity with
  /// `Proton::block()`). Triggers `build()`.
  [[nodiscard]] std::shared_ptr<Spacetime> block();
  /// The emergent `(k+2)`-vertex register holes on the whole — an observable, not a
  /// gate; may be any count, including zero. Triggers `build()`.
  [[nodiscard]] std::vector<std::vector<std::uint64_t>> emergentHoles();
  /// **Diagnostic only**: the relabeling-invariant singlet `r_state` of
  /// `Proton::singlet()` against the whole's `L_k` harmonic — reported so the emergent
  /// result is comparable to the canonical build's carried level (`≈0` there). It never
  /// steers or gates this build. Triggers `build()`.
  [[nodiscard]] double singletResidual();
  /// Step B's inputs-only realizability residual `r_U` (the whole matter term of the
  /// emergent arm's objective). Triggers `build()`.
  [[nodiscard]] double inputResidual();
  /// The kept attempt's final objective `F`. Triggers `build()`.
  [[nodiscard]] double finalObjective();
  /// Step A's `r_U` — reported exactly as `Proton` reports it. Triggers `build()`.
  [[nodiscard]] double diquarkResidual();

 private:
  /// Lazily run `build()` with default parameters on first accessor use.
  void ensureBuilt();
  /// The same minimal seed as `Proton`: a single Δ⁴ simplex with the uniform
  /// all-spacelike metric (`ℓ² = +1`; see the class note on why no causal structure is
  /// initialized). Mirrors `Proton::buildMinimalSeed`, which is private there.
  [[nodiscard]] static std::shared_ptr<Spacetime> buildMinimalSeed();

  // ---- configuration ----
  /// The canonical arm, composed unchanged: supplies step A's node verbatim and the
  /// configuration contract (seed/degree/Γ/input weight/precone/directed surgery).
  Proton proton_;
  std::uint64_t baseSeed_;
  int registerDegree_;
  double gamma_;
  double inputResidualWeight_;
  int precone_;
  bool shouldUseDirectedSurgery_;

  // ---- build state (populated by build()) ----
  bool attempted_ = false;
  bool converged_ = false;
  bool stationary_ = false;
  bool persistent_ = false;
  std::uint64_t keptSeed_ = 0;
  std::shared_ptr<Spacetime> spacetime_;
  std::vector<std::vector<std::uint64_t>> emergentHoles_;
  double singletResidual_ = 0.0;
  double inputResidual_ = 0.0;
  double finalObjective_ = 0.0;
  double diquarkResidual_ = 0.0;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_PROTON_INGREDIENTS_H
