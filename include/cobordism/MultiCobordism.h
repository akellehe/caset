// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_MULTICOBORDISM_H
#define TESSERA_COBORDISM_MULTICOBORDISM_H

#include <complex>
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
/// The C++ source-of-truth port of `examples/cobordism/emergent_optimizer.py`
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
///   * **Stage 2 (geometric):** relax every (complex) edge `ℓ²` toward a stationary
///     point of `β‖∇S‖² + Γ·r_U` (Wirtinger steepest descent, backtracking line
///     search), re-opening the scale DOF.
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
      std::uint64_t seed = 0, int precone = 0);

  // ---- module-level helpers (static; the reference's free functions) ----
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
  /// `growBoundaries` is the INITIALIZATION pass: while true the boundary regions
  /// grow to track the bulk until they carry their states (growBoundaryRegions);
  /// run the bulk EVOLUTION with it false, so ∂W stays frozen.
  std::vector<double> runStage1(int maxSteps = 200, int nCandidateMoves = 12,
                                int patience = 8, bool growBoundaries = false);
  /// Stage 2 (geometric): relax every (complex) edge `ℓ²` toward a stationary point of
  /// `β‖∇S‖² + Γ·r_U` (Wirtinger steepest descent, backtracking line search). The line
  /// search accepts a step only when it lowers `F` by more than `relTol·max(|F|,1)` — a
  /// RELATIVE stationarity test (an absolute floor of `relTol` for `|F| < 1`), so the
  /// criterion scales with the objective rather than the absolute `convergenceTolerance_`
  /// the surgery stages use (for `F ≈ 100` that absolute `1e-9` accepted ~`1e-11` relative
  /// steps — the rounding floor). "No line-search step beats the threshold" is the
  /// stationary stop; `maxIters` is the safety budget cap. `lastStage2Stationary()` reports
  /// which of the two ended the run. Returns the `F` trace.
  std::vector<double> runStage2(double beta = 1.0, int maxIters = 200,
                                  double alpha0 = 0.05, double relTol = 1e-9);

  [[nodiscard]] std::shared_ptr<Spacetime> spacetime() const { return spacetime_; }
  [[nodiscard]] const std::vector<BoundaryBlock> &inputs() const {
    return inputBlocks_;
  }
  [[nodiscard]] const std::vector<BoundaryBlock> &outputs() const {
    return outputBlocks_;
  }
  /// Whether the last `runStage2` ended on the relative-tolerance stationarity test (no
  /// line-search step lowered `F` by more than `relTol·max(|F|,1)`) — `true` — versus
  /// hitting the `maxIters` budget cap — `false`. Lets a caller report "stopped:
  /// stationary" vs "stopped: budget". `false` before the first `runStage2`.
  [[nodiscard]] bool lastStage2Stationary() const { return lastStage2Stationary_; }

 private:
  using Snapshot =
      std::pair<std::vector<std::vector<std::uint64_t>>,
                std::map<std::pair<std::uint64_t, std::uint64_t>,
                         std::complex<double>>>;
  using MoveSpec = std::pair<std::string, std::vector<std::uint64_t>>;

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
  // All pinned boundary (input + output) vertices — none may be removed by a move.
  [[nodiscard]] std::set<std::uint64_t> boundaryVerts() const;

  [[nodiscard]] Snapshot snapshotOf(const Spacetime &spacetime) const;
  [[nodiscard]] Snapshot snapshot() const;
  [[nodiscard]] std::shared_ptr<Spacetime> build(
      const Snapshot &complexSnapshot) const;

  /// Draw one random stage-1 move specification on `spacetime`: a `{kind, payload}`
  /// pair where `kind` is one of `add`/`remove`/`flip`/`iflip` (payload = a seed for
  /// the Pachner move) or `cone_out`/`cone_in` (payload = the cell/face to cone). The
  /// move is only described here, not applied — see `applyMoveSpecification`.
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
  double convergenceTolerance_ = 1e-9;
  /// Set by `runStage2`: `true` iff its last call stopped on the relative-tolerance
  /// stationarity test, `false` iff it hit the `maxIters` budget. See lastStage2Stationary.
  bool lastStage2Stationary_ = false;
  std::vector<BoundaryBlock> inputBlocks_;
  std::vector<BoundaryBlock> outputBlocks_;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_MULTICOBORDISM_H
