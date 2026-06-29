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
  MultiCobordism(
      std::shared_ptr<Spacetime> host,
      const std::vector<std::vector<std::complex<double>>> &inputTargets,
      const std::vector<std::vector<std::complex<double>>> &outputTargets,
      const std::vector<int> &degrees = {3}, double gamma = 1.0,
      std::uint64_t seed = 0);

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

  // ---- the two stages + boundary-block construction ----
  /// Grow each INPUT block's emergent sub-complex near its seed vertex.
  void constructInputs(const std::vector<std::uint64_t> &seeds, int rounds = 24);
  /// Grow each OUTPUT block's emergent sub-complex near its seed vertex.
  void constructOutputs(const std::vector<std::uint64_t> &seeds, int rounds = 24);
  std::vector<double> runStage1(int maxSteps = 200, int nCandidates = 12,
                                int patience = 8);
  std::vector<double> runStage2(double beta = 1.0, int maxIters = 40,
                                  double alpha0 = 0.05);

  [[nodiscard]] std::shared_ptr<Spacetime> spacetime() const { return spacetime_; }
  [[nodiscard]] const std::vector<BoundaryBlock> &inputs() const {
    return inputs_;
  }
  [[nodiscard]] const std::vector<BoundaryBlock> &outputs() const {
    return outputs_;
  }

  /// # Composite spin — loops-as-quarks closed-loop holonomy (#517)
  /// The total-spin Casimir \f$ J^2 \f$ of output block `outputBlockIndex`, read in the
  /// edge (loops-as-quarks) basis. The pair-encircling loop \f$ \gamma_{ij} \f$ is the
  /// Poincare dual of the complementary hole \f$ k \f$, so its closed-loop spin holonomy is
  /// the deficit \f$ \varepsilon_k \f$ at hole \f$ k \f$ (sum of `deficitAngle` over the
  /// hole's hinges), lifted through the Dirac-Kahler spin-1/2 double cover to the pairwise
  /// correlation \f$ \langle S_i\cdot S_j\rangle = \tfrac14\cos\varepsilon_k \f$. Then
  /// \f[ J^2 = 3\cdot\tfrac34 + 2\sum_{i<j}\langle S_i\cdot S_j\rangle
  ///         = \tfrac94 + \tfrac12\sum_k \cos\varepsilon_k, \f]
  /// the \f$ \tfrac34 \f$ being the structural spin-1/2 Casimir of one `DiracKahler` fiber.
  /// Honest: this reduces to the closed-loop holonomy / pairwise correlator and **floors**
  /// above the entangled proton \f$ \tfrac34 \f$ (it does not bypass the fiber-cells kernel);
  /// the value now lives on the source-of-truth class. Throws if the block has no 3-hole
  /// (\f$ b_3 \f$) register.
  [[nodiscard]] double compositeSpinJ2(std::size_t outputBlockIndex = 0) const;

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
  // Build the emergent boundary sub-complexes for `targets` near `seeds`, append
  // to `destinationBlocks` (shared by constructInputs/constructOutputs).
  void constructBlocks(const std::vector<std::uint64_t> &seeds,
                       const std::vector<std::vector<std::complex<double>>> &targets,
                       std::vector<BoundaryBlock> &destinationBlocks, int rounds);
  // All pinned boundary (input + output) vertices — none may be removed by a move.
  [[nodiscard]] std::set<std::uint64_t> boundaryVerts() const;

  /// The structural spin-1/2 Casimir Sum_a S_a^2 = 3/4 of one Dirac-Kahler constituent,
  /// from the spatial rotation generators Sigma_ij = 1/4 [gamma_i, gamma_j] of `DiracKahler`.
  [[nodiscard]] static double diracKahlerSpinCasimir(
      const std::shared_ptr<Spacetime> &spacetime);
  /// The closed-loop spin holonomy of the pair-loop dual to `hole`: the total deficit at the
  /// hole, Sum of `deficitAngle` over its hinge (triangle) faces, on the already-materialized
  /// `spacetime` (skeleton built by a ReggeSolver in `compositeSpinJ2`).
  [[nodiscard]] static double holeDeficit(
      const std::shared_ptr<Spacetime> &spacetime,
      const std::vector<std::uint64_t> &hole);

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
  double step(int nCandidates);

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
  /// The move/restart random source driving stage 1 and block construction.
  std::mt19937_64 randomNumberGenerator_;
  double convergenceTolerance_ = 1e-9;
  std::vector<BoundaryBlock> inputs_;
  std::vector<BoundaryBlock> outputs_;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_MULTICOBORDISM_H
