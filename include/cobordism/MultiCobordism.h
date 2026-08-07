// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_MULTICOBORDISM_H
#define TESSERA_COBORDISM_MULTICOBORDISM_H

#include <complex>

#include "spacetime/pachner/AddMove.h"
#include "spacetime/pachner/FlipMove.h"
#include "spacetime/pachner/IFlipMove.h"
#include "spacetime/pachner/RemoveMove.h"
#include <cstdint>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <utility>
#include <vector>

namespace tessera::spacetime { class Spacetime; }

namespace tessera::cobordism {
using ::tessera::spacetime::Spacetime;

/// # MultiCobordism
///
/// The C++ source-of-truth emergent-merge optimizer
/// (epic #457 / T5, #491): the merge as a **fully emergent** optimization — no
/// prescribed topology, no hand-placed register. From a bare host it grows the
/// register by **gated surgical moves** under the objective and reads the register
/// **dynamically** off `getBoundary` at a **user-defined degree k**.
///
/// Objective (the four-term `F`, extremize δS=0 — never minimize |S|):
/// \f[ F = \lVert\nabla S_{\text{Regge}}\rVert^2
///        + \Gamma\,\big( r_U(\text{output}) + \textstyle\sum_i r_U(\text{input}_i) \big) \f]
/// summed over the register `degrees`. `‖∇S‖²` is the **full complex**
/// `Σ_e |actionGradientExact_e|²`; each `r_U` is the relabeling-invariant,
/// zero-filled `residualForPeriods` over the emergent holes (the whole's holes for
/// the output; each input sub-complex's own holes for the inputs).
///
/// Two stages, exactly as the reference:
///   * **Stage 1 (combinatorial):** greedy best-ΔF single random moves
///     `{add,remove,flip,iflip,cone_out,cone_in}`, each gated by `dualComplexValid`
///     and "no input vertex removed", committed only if ΔF < 0; re-seed on stall.
///   * **Stage 2 (geometric):** relax every edge `ℓ²` along the **real signed-ℓ²
///     manifold** toward a stationary point of `β‖∇S‖² + Γ·r_U` (steepest descent
///     on `Re(2β·H̄·g)` — the exact restriction of the Wirtinger gradient to the
///     real axis — with a backtracking line search), re-opening the scale DOF.
class MultiCobordism {
 public:
  /// An emergent boundary block of the cobordism — an input OR an output. A block is
  /// NOT itself a complex: it stores the vertex SET it occupies plus the target period
  /// vector its own `L_k` sub-complex must carry. The sub-complex is recovered on
  /// demand from `vertices` by `subcomplexWithinVertexSet` (the ambient complex's top
  /// cells whose vertices all lie in the set), so the vertex set — together with the
  /// ambient triangulation — determines the block's complex.
  struct BoundaryBlock {
    std::set<std::uint64_t> vertices;
    std::vector<std::complex<double>> target;
  };

  /// `outputTargets` is a LIST of output boundary blocks (the full cobordism
  /// `∂W = inputs ⊔ outputs`, #491): a merge has one, a 2→2 recombination has two
  /// (diquark ⊔ antidiquark). Each output — like each input — is an emergent
  /// boundary sub-complex carrying its target, scored by its own `r_U`; the bulk
  /// routes the connectivity (which input constituent reaches which output).
  ///
  /// An **empty** `outputTargets` is a supported shape (#555): nothing is pinned
  /// downstream, `rU` sums only the input blocks (the objective is
  /// `‖∇S‖² + Γ·Σᵢ r_U(inputᵢ)`), and whatever the whole comes to carry is read
  /// after the fact — the emergent arm `ProtonIngredients` builds on this.
  ///
  /// `precone` (default 0) pre-grows the host by that many **gated cone-in moves**
  /// before any optimization — the emergent way to give surgery room to act, in
  /// place of a prebuilt host refinement. Each cone-in adds one top cell on a fresh
  /// apex over a random facet and is accepted only through the `dualComplexValid`
  /// gate (see `preconeCells`); on the single-Δ⁴ seed (a 4-ball) this enlarges the
  /// 4-ball. Reproducible given `seed`; `precone = 0` leaves the host untouched.
  MultiCobordism(
      std::shared_ptr<Spacetime> host,
      const std::vector<std::vector<std::complex<double>>> &inputTargets,
      const std::vector<std::vector<std::complex<double>>> &outputTargets,
      const std::vector<int> &degrees = {3}, double gamma = 1.0,
      std::uint64_t seed = 0, int precone = 0,
      bool shouldProposeDispositions = false);

  /// Move-kind names. Named rather than spelled as string literals at each site:
  /// every kind is written in the draw and compared in the apply, and a typo in
  /// either place would not fail to compile — it would silently misroute or
  /// disable the move.
  /// The four Pachner kinds are NOT redefined here — each move class owns its
  /// name (`AddMove::MOVE_TYPE` and siblings), and these alias those so there is
  /// exactly one definition per kind rather than one per dispatch site.
  static constexpr const char *ADD_MOVE = ::tessera::spacetime::AddMove::MOVE_TYPE;
  static constexpr const char *REMOVE_MOVE =
      ::tessera::spacetime::RemoveMove::MOVE_TYPE;
  static constexpr const char *FLIP_MOVE =
      ::tessera::spacetime::FlipMove::MOVE_TYPE;
  static constexpr const char *IFLIP_MOVE =
      ::tessera::spacetime::IFlipMove::MOVE_TYPE;
  /// Surgical kinds, owned here: they are `SurgicalCone` operations reached only
  /// through this draw, with no other definition to alias.
  static constexpr const char *CONE_OUT = "cone_out";
  static constexpr const char *CONE_IN = "cone_in";
  static constexpr const char *NOOP = "noop";
  /// The two disposition moves (#613).
  static constexpr const char *CONE_IN_TIMELIKE = "cone_in_timelike";
  static constexpr const char *FLIP_DISPOSITION = "flip_disposition";

  /// A `FLIP_DISPOSITION` payload names one edge by its two endpoint vertex ids.
  static constexpr std::size_t EDGE_ENDPOINT_COUNT = 2;

  /// True when \p payload names an edge — exactly two endpoint vertex ids. Reads
  /// as the question being asked, where a bare `size() == 2` does not.
  [[nodiscard]] static bool payloadNamesAnEdge(
      const std::vector<std::uint64_t> &payload) {
    return payload.size() == EDGE_ENDPOINT_COUNT;
  }

  /// Whether the stage-1 move draw also proposes CAUSAL DISPOSITIONS (#613): a
  /// timelike cone-in, and a disposition flip on an existing edge. Both are
  /// ordinary candidate moves — drawn at random, scored by `deltaF`, committed
  /// only when they lower `F`. Nothing prescribes causal structure; the objective
  /// decides whether it wants any.
  ///
  /// Drawn as DISCRETE moves rather than left to `runStage2` because a continuous
  /// descent cannot carry `ℓ²` across zero — a null, degenerate configuration
  /// where deficit angles and circumcentric dual volumes are singular — so the
  /// Euclidean orthant is a trap. Measured on canonical hosts: every edge stays
  /// spacelike and `Im S = 0` through 110+ relaxation iterations, with
  /// `‖∇S‖² = 9.46` still far from stationary.
  ///
  /// Default `false`, which leaves the six-move draw and every existing path —
  /// `Proton`, `ProtonIngredients`, the campaign — byte-identical.
  [[nodiscard]] bool shouldProposeDispositions() const {
    return shouldProposeDispositions_;
  }

  // ---- module-level helpers (static) ----
  /// Betti numbers (combinatorial, geometry-free).
  [[nodiscard]] static std::vector<int> betti(const Spacetime &st);
  /// The emergent k-register, read off `getBoundary`: the `(k+2)`-vertex tuples
  /// all of whose drop-one facets are boundary facets. Nothing placed.
  [[nodiscard]] static std::vector<std::vector<std::uint64_t>> emergentHoles(
      const Spacetime &st, int k);
  /// `Σ_e |actionGradientExact_e|²` — the full-complex Regge extremization term.
  [[nodiscard]] static double reggeActionGradient(const std::shared_ptr<Spacetime> &st);
  /// The relabeling-invariant, zero-filled residual of `targetState` against the
  /// `L_k` harmonic of `spacetime` over its emergent holes (`r_state` in the
  /// reference, the Python-binding name). For each register degree `k` it reads the
  /// emergent holes' cycle periods, least-squares-fits the target against them up to
  /// a relabeling of the target's components, and returns the smallest residual
  /// `\f$\lVert P c - t\rVert^2\f$`; with no emerged register it is the full leak
  /// `\f$\lVert t\rVert^2\f$`. Elemental: `residualForBoundaryBlock` sums this over
  /// the register degrees.
  [[nodiscard]] static double residualOfTargetStateAgainstHarmonic(
      const std::shared_ptr<Spacetime> &spacetime, int registerDegree,
      const std::vector<std::complex<double>> &targetState);

  // ---- objective ----
  /// The per-block register residual summed over `registerDegrees_`: `Σ r_U(boundary
  /// block)` over EVERY input and output block (the symmetric cobordism objective).
  [[nodiscard]] double rU(const std::shared_ptr<Spacetime> &st) const;
  /// `F = reggeActionGradient (Regge extremization) + gamma * rU`.
  [[nodiscard]] double objective() const;
  /// Weight on each INPUT block's residual in `rU` (the output/whole term keeps
  /// weight 1). Raising it makes the optimizer prioritize keeping the input states
  /// represented, rather than only driving the whole to the output. Default 1.
  void setInputResidualWeight(double weight) { inputResidualWeight_ = weight; }

  // ---- the two stages + boundary-block construction ----
  /// Seed one INPUT block per seed vertex (region = the seed's cell-neighbourhood,
  /// tagged with its target). NOT grown here — runStage1's growBoundaryRegions grows
  /// it emergently under the objective.
  void seedInputs(const std::vector<std::uint64_t> &seeds);
  /// Seed one OUTPUT block per seed vertex (see seedInputs).
  void seedOutputs(const std::vector<std::uint64_t> &seeds);
  /// Stage 1 (combinatorial): repeatedly `step()` — price moves and commit the best
  /// one that lowers `F` — with the trap door for escaping a too-small complex.
  ///
  /// `growBoundaries` is the INITIALIZATION pass: while true the boundary regions
  /// grow to track the bulk until they carry their states (growBoundaryRegions);
  /// run the bulk EVOLUTION with it false, so ∂W stays frozen.
  ///
  /// `nCandidateMoves` is the per-step trial budget: how many moves `step()` samples
  /// before concluding it cannot descend. It is ignored only in `step()`'s final
  /// check, which prices the whole move set before reporting failure.
  std::vector<double> runStage1(int maxSteps = 200, int nCandidateMoves = 12,
                                int patience = 8, bool growBoundaries = false);
  /// Stage 2 (geometric): relax every edge `ℓ²` along the **real signed-ℓ² manifold**
  /// toward a stationary point of `β‖∇S‖² + Γ·r_U`. The configuration space is real
  /// signed `ℓ²` (ordinary Lorentzian Regge; the complexified theory is unbuilt), so
  /// the descent direction is the exact gradient of `F` restricted to that manifold:
  /// for real `F` of a complex variable on the real axis `dF/dx = 2·Re(∂F/∂z̄)`, i.e.
  /// `Re(2β·H̄·g)` — the real part of the Wirtinger direction. Every trial is
  /// constructed exactly real, so **`Im ℓ² ≡ 0` holds for all time by construction**
  /// — no writer of `Im ℓ²` exists anywhere in the dynamics, nothing is enforced at
  /// runtime, and the invariant is proven by the suite tests.
  ///
  /// The line search takes **every step that lowers `F`, however little**. There is no
  /// improvement threshold: any threshold abandons a descent while `F` is still
  /// decreasable, which is a stop for a reason that is neither reaching the target nor
  /// running out of downhill. Among the halving ladder's rungs it keeps the **best**,
  /// not the first that improves.
  ///
  /// The run ends in exactly one of three ways, reported by `lastStage2Outcome()`:
  /// `F <= convergenceTarget` (`Converged` — the objective is met), no rung of the
  /// ladder lowers `F` while `F` is still above target (`Stalled` — a local minimum at
  /// non-zero value, which is a FAILURE to converge), or the `maxIters` budget runs out
  /// with `F` still descendable (`Truncated`). Returns the `F` trace.
  ///
  /// Trials are UNBOUNDED on the real axis — fully Lorentzian, no clamp, no causal
  /// guard (#565): a trial `Re ℓ²` may land spacelike, timelike, or lightlike (either
  /// sign or inside any `(-ε, ε)` band). The objective is total on the real manifold,
  /// so no trial can fail to evaluate — there is no backoff and no rejection beyond
  /// the line search's own variational acceptance; a genuine error propagates loudly.
  /// Epic #559's rule still holds — nothing here seeds causal content; the whole
  /// timelike/lightlike range is merely admissible, so causal content may EMERGE from
  /// the dynamics (its absence is equally a finding).
  std::vector<double> runStage2(double beta = 1.0, int maxIters = 200,
                                  double alpha0 = 0.05,
                                  double convergenceTarget =
                                      DEFAULT_CONVERGENCE_TARGET);

  /// One canonical solve action on THIS node, the unit a search policy (Proton's build
  /// restart loop, a greedy driver, or the RL agent) composes — so the solve is driven
  /// through the engine, not re-implemented by each consumer.
  enum class BuildAction { Grow, Evolve, Relax, ConeOut, ConeIn };

  /// Candidate ordering for the directed cone-out probe's *secondary* sort (both orders are
  /// interior-first): `AdjacentHolesLast` sends cells that share vertices with the existing
  /// holes to the back, so new holes come out separated; `AdjacentHolesFirst` brings them to
  /// the front, so the register clusters. (For the first hole the orders coincide.)
  enum class HolePlacementStrategy { AdjacentHolesFirst, AdjacentHolesLast };

  /// Apply one `BuildAction` to this node (in place). Grow/Evolve = `runStage1` with
  /// `growBoundaries` true/false; Relax = `runStage2`; ConeOut/ConeIn = the directed probes
  /// below. Irrelevant params for a given action are ignored.
  void buildStep(BuildAction action, int maxSteps = 30, int nCandidateMoves = 8,
                 int patience = 15, double stage2Beta = 1.0, int stage2MaxIters = 10,
                 double stage2Alpha0 = 0.05,
                 HolePlacementStrategy holePlacementStrategy = HolePlacementStrategy::AdjacentHolesLast);

  /// Directed, gated cone-OUT: open register holes deliberately. Enumerates candidate top
  /// cells interior-first; `AdjacentHolesLast` then sends cells sharing vertices with the
  /// existing holes to the back (new holes separated), `AdjacentHolesFirst` to the front
  /// (register clusters). Tries each with a gated `SurgicalCone::coneOut` (rolled back),
  /// skipping any that would strand a `pinnedBoundaryVertices()` vertex, and keeps the
  /// hole-opener that most lowers this node's `rU` (its realizability residual — which absorbs
  /// the output `r_state`, so this drives the register toward carrying the target on BOTH the
  /// 2→1 and 2→2 steps). Repeats up to `maxOpen`; stops when no opener lowers `rU`. Returns
  /// #holes opened.
  [[nodiscard]] int directedConeOut(HolePlacementStrategy strategy = HolePlacementStrategy::AdjacentHolesLast,
                                    int maxOpen = 6);

  /// Directed, gated cone-IN: select the register. Enumerates the boundary facets of the
  /// current emergent holes (capping one closes that hole), tries each with a gated
  /// `SurgicalCone::coneIn` (a fresh vertex, so nothing pinned is stranded), and keeps the
  /// cap that most lowers `rU` — i.e. drops the hole that hurts the carry. Repeats up to
  /// `maxClose`; stops when no cap lowers `rU`. Returns #holes capped.
  [[nodiscard]] int directedConeIn(int maxClose = 6);

  [[nodiscard]] std::shared_ptr<Spacetime> spacetime() const { return spacetime_; }
  [[nodiscard]] const std::vector<BoundaryBlock> &inputs() const {
    return inputBlocks_;
  }
  [[nodiscard]] const std::vector<BoundaryBlock> &outputs() const {
    return outputBlocks_;
  }

  /// Backtracking halvings the stage-2 line search will try before concluding
  /// there is no downhill direction. From `alpha0 = 0.05` this reaches a step
  /// scale of `0.05 * 2^-64 ~ 3e-21`, far below the point at which a step
  /// perturbs unit-scale `l^2` at all — so exhausting the ladder means the step
  /// underflowed the geometry, not that it was merely small. The previous value
  /// of 24 bottomed out at `~3e-9`, which could stop a descent that was still
  /// available.
  static constexpr int MAX_LINE_SEARCH_HALVINGS = 64;

  /// The value `F` must reach for `runStage2` to report `Converged`.
  static constexpr double DEFAULT_CONVERGENCE_TARGET = 1e-15;

  /// How the last `runStage2` ended. Convergence means the objective REACHED ITS
  /// TARGET; a descent that merely ran out of downhill is `Stalled`, which is a
  /// failure to converge and is reported as one.
  ///
  /// The previous name for this flag was "stationary" and it covered both cases
  /// at once, which invited reading it as `∇S = 0`. It never meant that: runs
  /// reporting it sat at `‖∇S‖²` between 2 and 15, a gradient of magnitude 1.5
  /// to 4.
  enum class Stage2Outcome {
    /// `F <= convergenceTarget`. The objective is MET. This is the only outcome
    /// that may be called convergence.
    Converged,
    /// No step at any scale lowers `F`, but `F` is still above the target. A
    /// local minimum of `F` at non-zero value — the descent is stuck, not done.
    /// This is NOT convergence and must never be reported as such.
    Stalled,
    /// The `maxIters` budget ran out with `F` still descendable.
    Truncated,
  };

  [[nodiscard]] Stage2Outcome lastStage2Outcome() const {
    return lastStage2Outcome_;
  }

 private:
  using Snapshot =
      std::pair<std::vector<std::vector<std::uint64_t>>,
                std::map<std::pair<std::uint64_t, std::uint64_t>,
                         std::complex<double>>>;
  using MoveSpec = std::pair<std::string, std::vector<std::uint64_t>>;

  /// The pinned boundary (input + output) vertices — none may be removed by a move. The move
  /// gate (`applyMoveSpecification`) and the directed cone-out probe consult it to avoid
  /// stranding a pinned vertex. (Currently empty — the boundary states are held by their `r_U`
  /// terms, not by freezing vertices.)
  [[nodiscard]] std::set<std::uint64_t> pinnedBoundaryVertices() const;

  /// The sub-complex carried by a boundary block: a freshly-built `Spacetime` of
  /// exactly the top cells of `spacetime` all of whose vertices lie in `vertexSet`
  /// (the block's region). Returns `nullptr` when the region contains no full cell.
  /// This is where a block's vertex-set becomes an actual complex — the block itself
  /// only stores the vertex-set and target, never the cells.
  [[nodiscard]] std::shared_ptr<Spacetime> subcomplexWithinVertexSet(
      const std::shared_ptr<Spacetime> &spacetime,
      const std::set<std::uint64_t> &vertexSet) const;
  /// One boundary block's `r_U` term: the sum over the register degrees of
  /// `residualOfTargetStateAgainstHarmonic` evaluated on the block's own
  /// sub-complex (`subcomplexWithinVertexSet`) against the block's target. When the
  /// block has no full sub-complex yet, the full leak summed over the degrees.
  [[nodiscard]] double residualForBoundaryBlock(
      const BoundaryBlock &boundaryBlock,
      const std::shared_ptr<Spacetime> &spacetime) const;
  // Seed one boundary block per (seed, target) — region = the seed's cell-neighbourhood
  // — appended to `destinationBlocks` (shared by seedInputs/seedOutputs). The blocks are
  // grown later by growBoundaryRegions, not here.
  void seedBlocks(const std::vector<std::uint64_t> &seeds,
                  const std::vector<std::vector<std::complex<double>>> &targets,
                  std::vector<BoundaryBlock> &destinationBlocks);

  [[nodiscard]] Snapshot snapshotOf(const Spacetime &spacetime) const;
  [[nodiscard]] Snapshot snapshot() const;
  [[nodiscard]] std::shared_ptr<Spacetime> build(
      const Snapshot &complexSnapshot) const;

  /// Every target of one Pachner `moveKind` on `spacetime`, from that move class's
  /// `PachnerMove::candidates()`. Prefiltered, not validated — a target here may
  /// still be rejected by the move's own `propose`.
  [[nodiscard]] static std::vector<std::vector<std::uint64_t>> pachnerTargets(
      const Spacetime &spacetime, const std::string &moveKind);

  /// EVERY move reachable from `spacetime`: the four Pachner kinds over all their
  /// targets, `cone_out` over every top cell, `cone_in` over every codim-1 facet,
  /// and (with dispositions on) `cone_in_timelike` over every facet and
  /// `flip_disposition` over every edge.
  ///
  /// This is what makes "nothing lowers `F`" a statement about the MOVE SET rather
  /// than about a handful of random draws — the difference between a genuine local
  /// minimum and a search that never looked in the right place, which the trace
  /// alone cannot distinguish.
  [[nodiscard]] std::vector<MoveSpec> enumerateMoveSpecifications(
      const Spacetime &spacetime) const;

  /// Draw one random stage-1 move specification on `spacetime`: a `{kind, payload}`
  /// pair where `kind` is one of `add`/`remove`/`flip`/`iflip` (payload = the move's
  /// TARGET, per `PachnerMove::Target`) or `cone_out`/`cone_in` (payload = the
  /// cell/face to cone). The move is only described here, not applied — see
  /// `applyMoveSpecification`.
  ///
  /// The Pachner payload used to be a random SEED rather than a target, so a
  /// specification named no particular move: the move class was handed a fresh
  /// engine and chose its own target, often one that failed to propose at all.
  [[nodiscard]] MoveSpec drawRandomMoveSpecification(const Spacetime &spacetime);
  /// Apply a move specification from `drawRandomMoveSpecification` to `spacetime`
  /// in place. Returns true iff the move was applied AND it left every pinned
  /// boundary vertex intact AND the result passes the `dualComplexValid` gate at
  /// `dualComplexGateDegree_`; otherwise the caller discards the candidate.
  [[nodiscard]] bool applyMoveSpecification(
      const std::shared_ptr<Spacetime> &spacetime,
      const MoveSpec &moveSpecification);
  [[nodiscard]] double deltaF(
      const std::shared_ptr<Spacetime> &candidateSpacetime, double baseResidualU,
      const std::set<std::vector<std::uint64_t>> &baseCellSet) const;
  /// One stage-1 surgery step: price moves against `deltaF` and commit the single
  /// best one that lowers `F`. Returns its `ΔF`, or `0.0` when nothing lowers `F`.
  ///
  /// Two passes:
  ///
  ///   1. **Normal** — at most `nCandidateMoves` random draws. Bounded work: pricing
  ///      one move costs a complex rebuild, a `dualComplexValid` check and two
  ///      `ReggeSolver` constructions, so the trial budget is what keeps a long run
  ///      tractable. If any draw lowers `F`, the best is committed and the step ends
  ///      here.
  ///   2. **Final check** — reached only when the draws found nothing, i.e. when the
  ///      step is about to report that it cannot descend. That claim is about the
  ///      whole move SET, which `nCandidateMoves` samples cannot support, so before
  ///      reporting failure every available move is priced.
  ///      `enumerateMoveSpecifications` supplies the set, and `nCandidateMoves` is
  ///      ignored here and only here. It takes the **first** move that lowers `F`
  ///      and stops: this pass is a rescue from an apparent dead end, not a second
  ///      optimization pass — choosing well among improvers is pass 1's job, and any
  ///      descent at all already answers the question it was entered to ask.
  ///
  /// Entry into the final check is logged, as is the improving move it finds with
  /// its `ΔF`, and the case where the entire set yields nothing. Without that, a
  /// step that reports no improvement while improving moves exist and were simply
  /// never drawn is indistinguishable in the trace from a genuine local minimum.
  double step(int nCandidateMoves);
  /// The trap door (#503): when no candidate move lowers the objective, take the
  /// first GATED move from the FULL range — Pachner `add`/`remove`/`flip`/`iflip`
  /// plus surgical `cone_out`/`cone_in` — within `attempts` tries. It is an
  /// escape/exploration step that need NOT lower F, so the optimizer can leave a
  /// local minimum (e.g. add material to a too-small complex, or rearrange) rather
  /// than stall, letting stage 1 build topology from as little as a seed simplex.
  /// Returns the move's ΔF, or NaN if no valid move exists (a true stall).
  double trapDoorMove(int attempts);
  /// Grow each localized boundary block's region to track the bulk's growth: expand
  /// its vertex set by one shell (every top cell touching the current region), so the
  /// block gets room to develop the holes that carry its state — instead of staying
  /// frozen at the (too-small) construct-time region. Applies to every INPUT block
  /// and every localized OUTPUT block (a multi-output recombination); a single output
  /// reads off the whole and has no block here. Bounded: a block already carrying
  /// (residual < inputCarriedTolerance_) is left alone, so it stops growing once it
  /// represents its state.
  void growBoundaryRegions();
  /// Pre-grow the seed by `count` **gated cone-in moves** before any optimization
  /// (the constructor calls this once when `precone > 0`): each cones a fresh apex
  /// onto a random codim-1 facet of a random top cell and is committed only through
  /// `applyMoveSpecification`'s `dualComplexValid` gate — the same gate stage 1 and
  /// the trap door use, so nothing is inserted by fiat. It enlarges the complex so
  /// surgery has room to act — the emergent analogue of a prebuilt host refinement.
  /// `count <= 0` is a no-op (RNG untouched). Best-effort: a draw onto an already-
  /// saturated facet is rejected by the gate and retried; if no valid cone-in is
  /// found for a cell, it stops early.
  void preconeCells(int count);

  std::shared_ptr<Spacetime> spacetime_;
  std::vector<std::vector<std::complex<double>>> inputTargets_;
  std::vector<std::vector<std::complex<double>>> outputTargets_;
  /// The register degrees `k` the objective scores at once (every `r_U` term is
  /// summed over these); a `b_k` register is forced to emerge for each.
  std::vector<int> registerDegrees_;
  /// The single degree at which the `dualComplexValid` move gate runs — the maximum
  /// register degree (the degree-free validity check needs only the coarsest one).
  int dualComplexGateDegree_;
  double gamma_;
  /// Weight on the input-block residual terms in `rU` (see setInputResidualWeight).
  double inputResidualWeight_ = 1.0;
  /// An input region stops growing (growInputRegions) once its residual drops below
  /// this — i.e. once it carries its state.
  double inputCarriedTolerance_ = 0.5;
  /// The move/restart random source driving stage 1 and block construction.
  std::mt19937_64 randomNumberGenerator_;
  /// #613: whether the move draw offers the disposition moves. See the accessor.
  bool shouldProposeDispositions_{false};
  double convergenceTolerance_ = 1e-9;
  /// Set by `runStage2`: `true` iff its last call stopped on the relative-tolerance
  /// How the last `runStage2` ended. See `lastStage2Outcome`.
  Stage2Outcome lastStage2Outcome_ = Stage2Outcome::Truncated;
  std::vector<BoundaryBlock> inputBlocks_;
  std::vector<BoundaryBlock> outputBlocks_;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_MULTICOBORDISM_H
