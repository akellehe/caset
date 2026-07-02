// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_PROTON_INGREDIENTS_H
#define TESSERA_COBORDISM_PROTON_INGREDIENTS_H

#include <complex>
#include <cstddef>
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
/// The class also hosts the **joint arm** (#560, rung 1 of the design note): the
/// two-step event graph collapsed into ONE co-optimized node (`jointNode` /
/// `buildJoint`) — inputs = the Z₃-orbit neutral triple, outputs = a baryon ⊔
/// antibaryon block pair — driven by the same per-node drive and judged by the same
/// stationarity + persistence convergence as `build()`. Exactly one variable then
/// differs from `Proton::build()`: the event graph (joint vs two-step). Microcausality
/// lives in the **move history** (every accepted change is one gated local move), not
/// in the boundary-block count, so the joint node is physically admissible (epic #559
/// decision log).
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

  /// Build the JOINT arm (#560): identical to `build()` — the same per-node drive, the
  /// same stationarity + persistence convergence, the same restart policy — except each
  /// attempt drives ONE `jointNode` instead of the A-then-B pair (restart `i` seeds it
  /// at `seed + i`; there is only one node, so the enumeration is dense). Observables
  /// are read after the fact exactly as in `build()`, plus the per-output-block singlet
  /// residuals `baryonResidual()`/`antibaryonResidual()`; `diquarkResidual()` is NaN
  /// (the diquark intermediate is collapsed away — nothing measured it). Idempotent,
  /// and mutually exclusive with `build()`: whichever runs first claims the instance
  /// (the accessors' lazy default remains the two-step `build()`).
  void buildJoint(int maxRestarts = 16, int initSteps = 180, int evolveSteps = 60,
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
  /// The JOINT formation node (#560, rung 1): the #489 two-step event graph collapsed
  /// into ONE fresh, seeded, NOT-run co-optimized node. Inputs = the **Z₃-orbit neutral
  /// triple** `{1,-1,0}`, `{0,1,-1}`, `{-1,0,1}` (the #398 symmetric-input lesson),
  /// seeded at the Δ⁴ seed's v0,v1,v2. Outputs = **two localized blocks** (the
  /// multi-output `rU` branch, exactly as the 2→2 recombination): the baryon
  /// `[1,ω,ω²]` at v3 ⊔ the antibaryon `[1,ω̄,ω̄²]` at v4 — with three pairs the whole
  /// must also host the conjugate sector, so a single whole-read singlet would fight
  /// the antibaryon's periods. **Conjugation subtlety:** `[1,ω̄,ω̄²]` is a
  /// component-permutation of `[1,ω,ω²]` (ω̄ = ω²) and the block residual is
  /// relabeling-invariant, so the two targets score identically as multisets — the
  /// conjugation is carried by the block's *location/orientation* in the emergent
  /// complex (which region hosts which sector), never by the residual's value.
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
  /// The kept node's full matter term `r_U`: after `build()` this is step B's
  /// inputs-only residual (nothing is pinned downstream there); after `buildJoint()`
  /// it also contains the two output-block terms. Triggers `build()`.
  [[nodiscard]] double inputResidual();
  /// The kept attempt's final objective `F`. Triggers `build()`.
  [[nodiscard]] double finalObjective();
  /// Step A's `r_U` — reported exactly as `Proton` reports it. NaN after
  /// `buildJoint()`: the joint arm has no step A (never a stale zero). Triggers
  /// `build()`.
  [[nodiscard]] double diquarkResidual();
  /// The baryon output block's singlet residual (#560) — its target `[1,ω,ω²]` scored
  /// against the block's OWN emergent sub-complex, read after the fact exactly as the
  /// drive's `rU` scores it. NaN unless `buildJoint()` ran (the two-step's step B has
  /// no localized output blocks). Triggers `build()`.
  [[nodiscard]] double baryonResidual();
  /// The antibaryon output block's residual — its conjugate target `[1,ω̄,ω̄²]` against
  /// its own sub-complex (see `jointNode` on why the conjugation lives in the block's
  /// location, not the residual's value). NaN unless `buildJoint()` ran. Triggers
  /// `build()`.
  [[nodiscard]] double antibaryonResidual();

 private:
  /// Lazily run `build()` with default parameters on first accessor use.
  void ensureBuilt();
  /// The same minimal seed as `Proton`: a single Δ⁴ simplex with the uniform
  /// all-spacelike metric (`ℓ² = +1`; see the class note on why no causal structure is
  /// initialized). Mirrors `Proton::buildMinimalSeed`, which is private there.
  [[nodiscard]] static std::shared_ptr<Spacetime> buildMinimalSeed();
  /// `Proton::build()`'s exact per-node drive, shared by `build()` and `buildJoint()`:
  /// INITIALIZATION pass (`grow_boundaries=true`), optional directed cone-out,
  /// EVOLUTION pass (∂W frozen), optional directed cone-in, then the geometric
  /// relaxation.
  void driveNode(MultiCobordism &node, int initSteps, int evolveSteps,
                 int stage1CandidateMoves, int stage1Patience, double stage2Beta,
                 int stage2MaxIters) const;
  /// The persistence verdict shared by `build()` and `buildJoint()`: continued
  /// evolution (∂W frozen) + relaxation must leave the answer-agnostic summary — the
  /// emergent hole count, `b_k`, and `F` (within `persistRelTol` relative) — stable.
  /// Up to `kMaxPersistencePasses` passes may settle a not-quite-stationary complex;
  /// the LAST pass has to be the stable one.
  [[nodiscard]] bool settleAndVerifyPersistence(MultiCobordism &node, int evolveSteps,
                                                int stage1CandidateMoves,
                                                int stage1Patience, double stage2Beta,
                                                int stage2MaxIters,
                                                double persistRelTol) const;
  /// The emergent hole count at `registerDegree_` (one leg of the answer-agnostic
  /// persistence summary).
  [[nodiscard]] int emergentHoleCount(const std::shared_ptr<Spacetime> &whole) const;
  /// `b_k` at `registerDegree_` (the other combinatorial leg of the summary).
  [[nodiscard]] int bettiAtRegisterDegree(const std::shared_ptr<Spacetime> &whole) const;
  /// One output block's residual, read after the fact for `buildJoint()`'s per-block
  /// observables: the block's own sub-complex (the ambient complex's top cells whose
  /// vertices all lie in the block's region) scored against the block's target —
  /// mirroring `MultiCobordism::residualForBoundaryBlock` (private there) at this
  /// class's single register degree, so the read matches how the drive's `rU` scored
  /// the block. The full leak `‖target‖²` when the region contains no full cell.
  [[nodiscard]] double outputBlockResidual(const MultiCobordism &node,
                                           std::size_t blockIndex) const;

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
  // Joint-arm per-output-block reads (#560); NaN whenever the two-step build() ran
  // instead (its step B has no localized output blocks), set in the .cpp so the
  // header stays <limits>-free.
  double baryonResidual_;
  double antibaryonResidual_;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_PROTON_INGREDIENTS_H
