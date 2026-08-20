// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_MULTICOBORDISM_H
#define TESSERA_COBORDISM_MULTICOBORDISM_H

#include <complex>

#include <Eigen/Core>

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
///     `{add,remove,flip,iflip,cone_out,cone_in,cone_in_timelike,flip_disposition}`
///     (the last two are the causal dispositions — see `shouldProposeDispositions`),
///     each gated by `dualComplexValid` and "no input vertex removed", committed
///     only if ΔF < 0; a batch with no improving move simply redraws (halting
///     only once the register is carried).
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
  /// `preconeTimelike` draws every precone cone-in as the TIMELIKE disposition
  /// (#613, apex edges ℓ² = −1); `preconeAlternate` instead ALTERNATES the
  /// cone-ins timelike/spacelike for balanced causal content at one uniform
  /// edge-length magnitude (it wins when both are set). Defaults keep the
  /// all-spacelike precone.
  /// `einsteinHilbert` (default true) keeps the discrete Einstein-Hilbert term
  /// `‖∇S_Regge‖²` in the objective. Set it false to optimize `F = gamma * rU`
  /// alone. NOTE what that does to stage 2: its descent direction is built from
  /// the Regge gradient and Hessian only, so with the term gone the direction is
  /// no longer a descent direction for what is being minimized. The line search
  /// still accepts only trials that lower the true F, so the drive stays
  /// monotone, but it searches along a ray derived from a term the objective no
  /// longer contains and will accept far fewer steps — in this mode the
  /// combinatorial moves do most of the work.
  ///
  /// `singularValueRatio` swaps the WHOLE-COMPLEX term of `rU` — both regimes:
  /// the single-output period residual and its `nearKernelResidual`
  /// continuation — for the scale-invariant singular-value half-sum ratio
  /// (`singularValueHalfSumRatio`). The input-block residuals keep anchoring
  /// the input states; nothing prescribes WHAT the whole comes to carry.
  MultiCobordism(
      std::shared_ptr<Spacetime> host,
      const std::vector<std::vector<std::complex<double>>> &inputTargets,
      const std::vector<std::vector<std::complex<double>>> &outputTargets,
      const std::vector<int> &degrees = {3}, double gamma = 1.0,
      std::uint64_t seed = 0, int precone = 0,
      bool shouldProposeDispositions = true, bool preconeTimelike = false,
      bool preconeAlternate = false,
                 bool balancedEdgeWiring = false,
                 bool singularValueRatio = false,
                 bool einsteinHilbert = true);

  /// Move-kind names. Named rather than spelled as string literals at each site:
  /// every kind is written in the draw and compared in the apply, and a typo in
  /// either place would not fail to compile — it would silently misroute or
  /// disable the move.
  /// The four Pachner kinds are NOT redefined here — each move class owns its
  /// name (`AddMove::kMoveType` and siblings), and these alias those so there is
  /// exactly one definition per kind rather than one per dispatch site.
  static constexpr const char *kAddMove = ::tessera::spacetime::AddMove::kMoveType;
  static constexpr const char *kRemoveMove =
      ::tessera::spacetime::RemoveMove::kMoveType;
  static constexpr const char *kFlipMove =
      ::tessera::spacetime::FlipMove::kMoveType;
  static constexpr const char *kIFlipMove =
      ::tessera::spacetime::IFlipMove::kMoveType;
  /// Surgical kinds, owned here: they are `SurgicalCone` operations reached only
  /// through this draw, with no other definition to alias.
  static constexpr const char *kConeOut = "cone_out";
  static constexpr const char *kConeIn = "cone_in";
  static constexpr const char *kNoop = "noop";
  /// The two disposition moves (#613).
  static constexpr const char *kConeInTimelike = "cone_in_timelike";
  static constexpr const char *kFlipDisposition = "flip_disposition";

  /// A `kFlipDisposition` payload names one edge by its two endpoint vertex ids.
  static constexpr std::size_t kEdgeEndpointCount = 2;

  /// True when \p payload names an edge — exactly two endpoint vertex ids. Reads
  /// as the question being asked, where a bare `size() == 2` does not.
  [[nodiscard]] static bool payloadNamesAnEdge(
      const std::vector<std::uint64_t> &payload) {
    return payload.size() == kEdgeEndpointCount;
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
  /// Default **`true`** (#632): the causal moves are the seed's ONLY descent
  /// directions, so a draw without them is not a neutral default — it hides the
  /// physics. Measured on the single-Δ⁴ seed by enumerating EVERY move that adds a
  /// vertex (each of the C(5,4) facets coned in, both apex dispositions, plus the
  /// Pachner insertion): each of the five spacelike cone-ins RAISES `F` by `+0.777`
  /// and the Pachner add by `+2.58`, while each of the five timelike cone-ins LOWERS
  /// `F`, `‖∇S‖²` and `Re S` (`ΔF = -0.208`, all five equal by the seed's S₅
  /// symmetry), and they are the only moves giving `Im S ≠ 0`. With the six-move draw
  /// stage 1 finds nothing that lowers `F`, reports a stall it does not actually
  /// have, and left the seed through the since-removed trap-door escape — building
  /// an all-spacelike complex whose `Im S` is identically zero.
  ///
  /// KNOWN, UNRESOLVED (#632): with the moves in the draw, `‖∇S‖²` runs to `1.1e+15`
  /// on a 13-cell host and `1.03e+298` on a `Proton` node, while the ACTION itself
  /// stays finite (`S = -24.45 - 10.15i`). Value finite, gradient astronomical. The
  /// causal structure that emerges is real; the objective scoring it is not yet
  /// trustworthy on these hosts.
  ///
  /// The cause is NOT mixed disposition as such — it is exact degeneracy of the
  /// **tetrahedral facets**. The circumcentre solves `G β = ½ diag G`, so it is
  /// undefined exactly when `det G = 0`. The metric is Lorentzian throughout, so `G`
  /// is indefinite and `det G = 0` is a configuration the complex can actually reach:
  /// a facet whose span is tangent to the light cone — a NULL 3-face, zero 3-volume
  /// even though every one of its edges has `|ℓ²| = 1`. Quantising every `ℓ²` to
  /// exactly `±1` then lands on that locus **exactly**, rather than with measure zero
  /// as generic lengths would. Enumerating all `2⁶` sign patterns of a tetrahedron's
  /// edges:
  ///
  ///   * `0,1,2,4,5,6` timelike edges — never degenerate (`|det G| ∈ {0.5,1.0,1.5}`)
  ///   * **`3/6` timelike — 12 of those 20 patterns give `det G = 0` exactly**
  ///
  /// so `12/64` of all patterns are degenerate. Triangles and pentatopes never are at
  /// `±1` (`min|det G|` = `0.75` and `0.3125`); it is only the facets, and they poison
  /// the DEC dual recursion, which evaluates `circumradiusSquared` on the hinge's
  /// cofaces. `Simplex::circumFromGram` divides by `detG` under an EXACT-zero guard,
  /// so a rounding-level residue instead of a clean `0` gives `β ~ 1/detG`.
  ///
  /// The lever is therefore the uniform `±1` initialisation sitting on the degenerate
  /// locus, not a clamp in the dynamics: generic edge lengths essentially never hit
  /// `det G = 0`.
  ///
  /// Pass `false` to recover the spacelike-only six-move draw — every edge stays
  /// spacelike, `Im S` is identically `0`, and the objective is well-behaved.
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
  /// The same residual, with the winning relabeling RECORDED so no two registers in
  /// one `r_U` evaluation are scored against the same one. Every register scored
  /// independently picks its own argmin relabeling, and nothing stops a second
  /// register from picking that same matching — which reads both registers as
  /// carrying the same target component and rewards a complex whose registers all
  /// carry equal weights. `claimedMatchings` holds the relabelings the registers
  /// before this one already won: they are skipped here, and this register's argmin
  /// is inserted on the way out. Once every relabeling is claimed (more registers
  /// than the `d!` the target admits) the set is cleared and the exclusion restarts,
  /// so the residual is never the empty minimum.
  [[nodiscard]] static double residualOfTargetStateAgainstHarmonicWithDistinctMatching(
      const std::shared_ptr<Spacetime> &spacetime, int registerDegree,
      const std::vector<std::complex<double>> &targetState,
      std::set<std::vector<int>> &claimedMatchings);

  // ---- objective ----
  /// The per-block register residual summed over `registerDegrees_`: `Σ r_U(boundary
  /// block)` over EVERY input and output block (the symmetric cobordism objective),
  /// PLUS the near-kernel residual per degree (see `nearKernelResidual`) — the
  /// pre-topological register signal. The period residual alone is a STEP function
  /// in the topology: before the first register opens it sits exactly at its
  /// zero-filled-leak floor (measured: `gamma * r_U = 50.000` for the seed and for
  /// every candidate cone-in), so F carries no register-seeking gradient at all
  /// until a hole exists. The near-kernel term is its analytic continuation below
  /// the topological threshold: on the near-kernel the period residual is a
  /// target-weighted sum of the smallest `|lambda|^2`, and this term is the same
  /// functional evaluated BEFORE the modes reach the kernel. The two meet at the
  /// opening: once `b_k` reaches the expected register count the smallest singular
  /// values are exactly zero, the near-kernel term saturates at 0, and the period
  /// residual takes over scoring WHAT the registers carry.
  [[nodiscard]] double rU(const std::shared_ptr<Spacetime> &st) const;

  /// The pre-topological register signal at one degree: the sum of the
  /// `expectedRegisterCount` smallest squared SINGULAR values of the METRIC
  /// `L_k` (the signed operator under the process weight convention),
  /// normalized scale-free.
  ///
  /// * **Metric, deliberately**: the term feels the continuously-valued edge
  ///   lengths, so it descends along TWO channels — stage-1 surgery (a genuine
  ///   hole zeroes the corresponding singular values exactly), and stage-2
  ///   tuning of the CAUSAL STRUCTURE toward null directions, which opens
  ///   near-kernels with no holes at all. The second channel is the point,
  ///   not a loophole: whether such causal near-kernels can carry a register
  ///   is the next level of exploration (readout semantics not implemented
  ///   here).
  /// * **Singular values, not eigenvalues**: the signed operator is
  ///   non-normal; singular values are the eigenvalue magnitudes of the
  ///   Hermitian `L^dagger L`, share the kernel exactly, and are smooth where
  ///   `|lambda|` is not — at a defective point the m smallest `|lambda|`
  ///   double-count a one-dimensional kernel, where the sigma count dimensions
  ///   honestly.
  /// * **Normalization**: `n * (sum of the m smallest sigma^2) / (sum of all
  ///   sigma^2)` — a generic mode contributes ~1, the range is [0, m]. `L_k`
  ///   is homogeneous of degree −1 in `l^2`, so a RAW spectral sum would hand
  ///   stage 2 a pure conformal-inflation descent channel (scale the geometry
  ///   up, every sigma shrinks); the ratio is degree 0 and closes exactly that
  ///   channel while leaving the causal-tuning channel open.
  /// * **Count**: `m` = the expected register count, read off the TARGETS
  ///   (`expectedRegisterCount`), never a constant — the term is the soft
  ///   relaxation of `b_k >= m`. Missing dimensions (`n < m`) count 1 each,
  ///   the worst case on the normalized scale.
  [[nodiscard]] static double nearKernelResidual(
      const std::shared_ptr<Spacetime> &st, int registerDegree,
      std::size_t expectedRegisterCount);

  /// The scale-invariant spectral-shape term the `singularValueRatio` mode uses
  /// as the whole-complex contribution to `rU`, in place of BOTH the
  /// single-output period residual and `nearKernelResidual`: the ratio of the
  /// sum of the lower half of the singular values of the METRIC `L_k` (the same
  /// signed operator `nearKernelResidual` reads) to the sum of the upper half.
  /// With `n` values descending and `h = n/2` (integer division), it is
  /// `(σ_{n−h+1} + … + σ_n) / (σ_1 + … + σ_h)`; an odd `n` leaves the median out
  /// of both halves. Each lower-half value is bounded by its upper-half
  /// counterpart, so the ratio lives in `[0, 1]`, and `L_k` is homogeneous of
  /// degree −1 in `l^2`, so a uniform rescale of the geometry scales every
  /// `σ` alike and cancels — degree 0, no conformal-inflation channel, no
  /// prescribed target: the term rewards a collapsing lower half of the
  /// spectrum, and WHAT the register comes to carry is read out afterwards.
  /// No `k`-cells at all returns 1 (the worst case — an empty complex must not
  /// score as a collapsed spectrum); `n = 1` and an identically-zero `L_k`
  /// return 0 (no pair of halves to compare / every mode already kernel).
  [[nodiscard]] static double singularValueHalfSumRatio(
      const std::shared_ptr<Spacetime> &st, int registerDegree);

  /// The number of registers the targets ask for: the largest component count
  /// over every input and output target vector (each component is carried by
  /// one register/hole, so a `[1, omega, omega^2]` target needs three).
  [[nodiscard]] std::size_t expectedRegisterCount() const;
  /// `F = reggeActionGradient (Regge extremization) + gamma * rU`.
  [[nodiscard]] double objective() const;
  /// Weight on each INPUT block's residual in `rU` (the output/whole term keeps
  /// weight 1). Raising it makes the optimizer prioritize keeping the input states
  /// represented, rather than only driving the whole to the output. Default 1.
  void setInputResidualWeight(double weight) { inputResidualWeight_ = weight; }

  // ---- the two stages + boundary-block construction ----
  /// Seed one INPUT block per seed vertex (region = the seed's cell-neighbourhood,
  /// tagged with its target). NOT grown here — runStage1's growBlockRegions grows
  /// it emergently under the objective.
  void seedInputs(const std::vector<std::uint64_t> &seeds);
  /// Seed one OUTPUT block per seed vertex (see seedInputs).
  void seedOutputs(const std::vector<std::uint64_t> &seeds);
  /// `growBoundaries` is the INITIALIZATION pass: while true the boundary regions
  /// grow to track the bulk until they carry their states (growBlockRegions);
  /// run the bulk EVOLUTION with it false, so ∂W stays frozen.
  /// `maxLookahead`: when a batch of single moves finds no improvement, the
  /// search deepens iteratively — 2-move sequences, then 3, up to this many
  /// moves — committing an F-lowering sequence as a whole. DEFAULT 1 (single
  /// moves): a deepened batch builds and scores many more candidates per
  /// iteration, so deepening is a caller's choice rather than a default — the
  /// proton animation passes its `--max-lookahead-depth`. Every depth scores
  /// the same way, unrelaxed (#714).
  std::vector<double> runStage1(int maxSteps = 200, int nCandidateMoves = 12,
                                bool growBoundaries = false,
                                int maxLookahead = 1);
  /// Stage 2 (geometric): relax every edge `ℓ²` along the **real signed-ℓ² manifold**
  /// toward a stationary point of `β‖∇S‖² + Γ·r_U`. The configuration space is real
  /// signed `ℓ²` (ordinary Lorentzian Regge; the complexified theory is unbuilt), so
  /// the descent direction is the exact gradient of `F` restricted to that manifold:
  /// for real `F` of a complex variable on the real axis `dF/dx = 2·Re(∂F/∂z̄)`, i.e.
  /// `Re(2β·H̄·g)` — the real part of the Wirtinger direction. Every trial is
  /// constructed exactly real, so **`Im ℓ² ≡ 0` holds for all time by construction**
  /// — no writer of `Im ℓ²` exists anywhere in the dynamics, nothing is enforced at
  /// runtime, and the invariant is proven by the suite tests. The line search accepts
  /// a step only when it lowers `F` by more than `tolerance·max(|F|,1)` — a RELATIVE
  /// stationarity test (an absolute floor of `tolerance` for `|F| < 1`), so the
  /// criterion scales with the objective rather than the absolute `convergenceTolerance_`
  /// the surgery stages use (for `F ≈ 100` that absolute `1e-9` accepted ~`1e-11` relative
  /// steps — the rounding floor). "No line-search step beats the threshold" is the
  /// stationary stop; `maxIters` is the safety budget cap. `lastStage2Stationary()` reports
  /// which of the two ended the run. Returns the `F` trace.
  ///
  /// Trials are UNBOUNDED on the real axis — fully Lorentzian, no clamp, no causal
  /// guard (#565): a trial `Re ℓ²` may land spacelike, timelike, or lightlike (either
  /// sign or inside any `(-ε, ε)` band). The objective is total on the real manifold,
  /// so no trial can fail to evaluate — there is no backoff and no rejection beyond
  /// the line search's own variational acceptance; a genuine error propagates loudly.
  /// Epic #559's rule still holds — nothing here seeds causal content; the whole
  /// timelike/lightlike range is merely admissible, so causal content may EMERGE from
  /// the dynamics (its absence is equally a finding).
  /// Default `tolerance` 1e-12: `runStage2` is the FINAL, precise relaxation of a
  /// drive (the combined `run` iterates its in-loop relaxations at the looser
  /// 10e-9 diminishing-returns cut and applies the same 1e-12 on its exit path).
  std::vector<double> runStage2(double beta = 1.0, int maxIters = 200,
                                  double alpha0 = 0.05, double tolerance = 1e-12);
  /// The combined drive. Each iteration takes ONE combinatorial stage-1 update —
  /// a best-ΔF move, deepening to `maxLookahead`-move sequences on a stall — and
  /// then relaxes the geometry FULLY: stage-2 updates repeat until the relative
  /// stationarity test at `tolerance` (default 10e-9) reports diminishing returns,
  /// so every move is proposed from, and leaves behind, relaxed geometry.
  ///
  /// Exit protocol: the loop wants to exit once the register is carried with the
  /// geometry stationary, or once the combinatorial moves have had no effect
  /// (nothing committed at any lookahead depth AND nothing left to relax) for a
  /// few consecutive iterations (one stalled batch is draw noise, not proof).
  /// The LAST geometric relaxation before exit then runs at the tight 1e-12: if
  /// it still finds descent, the exit was premature and the loop continues on
  /// the freshly relaxed geometry; only a state stationary at 1e-12 exits.
  /// `maxIters` remains the hard budget cap.
  ///
  /// `nCandidateMoves`/`growBoundaries`/`maxLookahead` parameterize the
  /// combinatorial half exactly as in `runStage1`; `beta`/`alpha0`/`tolerance` the
  /// geometric half exactly as in `runStage2`. NOTE with `beta != 1` the two
  /// halves weight `‖∇S‖²` differently (stage 1 books deltas of `objective()`,
  /// stage 2 descends `β‖∇S‖² + Γ·r_U`), so the shared trace mixes the two
  /// scales — the default `beta = 1` keeps one coherent `F`.
  /// `lastStage2Stationary()` reports the LAST geometric update's outcome.
  /// `relaxBudgetPerMove` caps the stage-2 updates that follow each committed
  /// move (and the tight exit re-check): the stationarity test is the real
  /// terminator, the cap only bounds slow descent tails where the line search
  /// accepts a near-unbounded run of threshold-sized micro-steps.
  /// Returns the combined `F` trace.
  std::vector<double> run(int maxIters = 200, int nCandidateMoves = 12,
                          bool growBoundaries = false,
                          double beta = 1.0, double alpha0 = 0.05,
                          double tolerance = 10e-9, int maxLookahead = 1,
                          int relaxBudgetPerMove = 10);

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
                 double stage2Beta = 1.0, int stage2MaxIters = 10,
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
  /// Whether the last `runStage2` ended on the relative-tolerance stationarity test (no
  /// line-search step lowered `F` by more than `tolerance·max(|F|,1)`) — `true` — versus
  /// hitting the `maxIters` budget cap — `false`. `true` means **real-manifold
  /// stationarity, `δF = 0` along real signed-ℓ² perturbations**: the exact
  /// on-manifold gradient direction `Re(2β·H̄·g)` buys no further descent (#589).
  /// Lets a caller report "stopped: stationary" vs "stopped: budget". `false` before
  /// the first `runStage2`/`run`; after `run`, reports the LAST geometric update's
  /// outcome (each update resets the flag, so an earlier stationary point that a
  /// later topology change reopened does not latch).
  [[nodiscard]] bool lastStage2Stationary() const { return lastStage2Stationary_; }

  /// The lookahead depth of the LAST stage-1 update's committed sequence: 1 = an
  /// ordinary single move, >1 = the search had to deepen (the single-move batch
  /// stalled and an F-lowering multi-move sequence was found at this depth), 0 =
  /// no F-lowering sequence found at ANY depth up to the update's `maxLookahead`
  /// (a stage-1 stall). 0 before the first update. Lets a driver/animation show
  /// WHEN the optimizer is looking more than one move into the future.
  [[nodiscard]] int lastStage1Lookahead() const { return lastStage1LookaheadDepth_; }

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

  // ---- the pieces of residualOfTargetStateAgainstHarmonic ----
  /// The target state as a dense complex vector — the `t` the harmonic is fitted to,
  /// and (as its squared norm) the full leak when no register has emerged.
  [[nodiscard]] static Eigen::VectorXcd targetStateVector(
      const std::vector<std::complex<double>> &targetState);
  /// The emergent holes that can carry `targetDimension` components: `emergentHoles`
  /// at this degree, truncated to at most one hole per target component. Empty when
  /// no register has emerged.
  [[nodiscard]] static std::vector<std::vector<std::uint64_t>> holesCarryingTheTarget(
      const Spacetime &spacetime, int registerDegree, std::size_t targetDimension);
  /// The period matrix \f$ P^{\top} \f$ of the degree's harmonics over `registerHoles`:
  /// `(targetDimension, b_k)`, row = hole, column = harmonic, zero-filled past the
  /// holes that emerged (a component with no hole to sit in leaks in full).
  [[nodiscard]] static Eigen::MatrixXcd holePeriodMatrix(
      const std::shared_ptr<Spacetime> &spacetime, int registerDegree,
      int degreeBettiNumber,
      const std::vector<std::vector<std::uint64_t>> &registerHoles,
      std::size_t targetDimension);
  /// The target's components reordered onto the holes by `relabeling`: component
  /// `relabeling[q]` sits in hole `q`. A relabeling is a bijection, so each hole
  /// takes exactly one component.
  [[nodiscard]] static Eigen::VectorXcd relabeledTargetVector(
      const Eigen::VectorXcd &targetVector, const std::vector<int> &relabeling);
  /// One register's winning relabeling: which target component each of its holes
  /// carries, and the least-squares residual \f$ \min_c \lVert P^{\top} c - t \rVert^2 \f$
  /// that matching leaves. `scored` is false when every relabeling was skipped as
  /// already claimed, so nothing was evaluated.
  struct RelabelingMatch {
    double residual = 0.0;
    std::vector<int> relabeling;
    bool scored = false;
  };
  /// The argmin over the `d!` relabelings of the target components onto the holes.
  /// With `skipClaimed` the relabelings in `claimedMatchings` — the ones registers
  /// scored earlier already won — are passed over, so this register is read against
  /// a matching of its own.
  [[nodiscard]] static RelabelingMatch bestRelabelingOfTarget(
      const Eigen::MatrixXcd &periodMatrixTransposed,
      const Eigen::VectorXcd &targetVector,
      const std::set<std::vector<int>> &claimedMatchings, bool skipClaimed);

  /// One boundary block's `r_U` term: the sum over the register degrees of
  /// `residualOfTargetStateAgainstHarmonic` evaluated on the block's own
  /// sub-complex (`subcomplexWithinVertexSet`) against the block's target. When the
  /// block has no full sub-complex yet, the full leak summed over the degrees.
  [[nodiscard]] double residualForBoundaryBlock(
      const BoundaryBlock &boundaryBlock,
      const std::shared_ptr<Spacetime> &spacetime) const;
  /// The same block term, sharing one `claimedMatchings` set with the rest of the
  /// `r_U` evaluation so this block's register degrees cannot re-use a relabeling
  /// another register already won (see
  /// `residualOfTargetStateAgainstHarmonicWithDistinctMatching`).
  [[nodiscard]] double residualForBoundaryBlockWithDistinctMatchings(
      const BoundaryBlock &boundaryBlock,
      const std::shared_ptr<Spacetime> &spacetime,
      std::set<std::vector<int>> &claimedMatchings) const;
  // Seed one boundary block per (seed, target) — region = the seed's cell-neighbourhood
  // — appended to `destinationBlocks` (shared by seedInputs/seedOutputs). The blocks are
  // grown later by growBlockRegions, not here.
  void seedBlocks(const std::vector<std::uint64_t> &seeds,
                  const std::vector<std::vector<std::complex<double>>> &targets,
                  std::vector<BoundaryBlock> &destinationBlocks);

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
  /// One best-ΔF batch: `nCandidateMoves` candidates, each a sequence of
  /// `lookaheadDepth` gated random moves applied successively (each drawn against
  /// the evolving candidate), committed as a whole iff the best sequence lowers
  /// F. EVERY depth scores by the same localized, UNRELAXED `deltaF` (#714):
  /// the combinatorial moves exist to leave a local minimum and the geometric
  /// update to descend within the region the complex then occupies, so scoring
  /// a candidate through a relaxation would mix the two — it would ask where a
  /// move lands after stage 2 rather than whether the move improves the state.
  /// A committed candidate is relaxed afterwards, bounded by the caller's
  /// `relaxBudgetPerMove`. Depth 1 pre-draws its batch and scores it in
  /// parallel; deeper searches stay serial, since each draw is made against the
  /// evolving candidate. Returns the committed ΔF, or 0.
  double step(int nCandidateMoves, int lookaheadDepth = 1);
  /// One iteration of `runStage1`'s loop: optional boundary growth plus one
  /// best-ΔF candidate-move step, booked into `objectiveTrace`. A batch with no
  /// improving move is NOT a stall — the batch is a random sample, so the next
  /// iteration simply redraws. Returns `false` only when the run should halt: no
  /// improving move AND the register already carried (converged); `true` to keep
  /// iterating.
  bool stage1Update(int nCandidateMoves, bool growBoundaries,
                    std::vector<double> &objectiveTrace, int maxLookahead = 1);
  /// One iteration of `runStage2`'s loop: exact gradient/Hessian, the on-manifold
  /// descent direction `Re(2β·H̄·g)`, and the backtracking line search. Appends the
  /// accepted objective to `objectiveTrace` and adapts `stepScale`. Returns `false`
  /// on the stationary stop (no line-search step beat the relative threshold —
  /// lengths restored and `lastStage2Stationary_` set), `true` to keep iterating.
  bool stage2Update(double beta, double tolerance,
                    std::vector<double> &objectiveTrace, double &stepScale);
  /// Grow each localized boundary block's region to track the bulk's growth: expand
  /// its vertex set by one shell (every top cell touching the current region), so the
  /// block gets room to develop the holes that carry its state — instead of staying
  /// frozen at the (too-small) construct-time region. Applies to every INPUT block
  /// and every localized OUTPUT block (a multi-output recombination); a single output
  /// reads off the whole and has no block here. Bounded: a block already carrying
  /// (residual < inputCarriedTolerance_) is left alone, so it stops growing once it
  /// represents its state. GATED per block: a shell that would RAISE the block's
  /// own r_U term is reverted (Δ <= 0 passes — the full-leak plateau of a region
  /// with no full cell yet is Δ == 0), so region growth can never raise F.
  /// Expand each not-yet-carrying block's SCORING REGION by one shell (the
  /// vertices of every top cell touching it), so the window a block's residual
  /// is read over gains room for the holes that carry its state.
  ///
  /// Two conditions bound it (#737). A shell is kept only when it STRICTLY
  /// lowers that block's residual, and growth happens only BEFORE the first
  /// committed combinatorial move — once the bulk is being linked, the states'
  /// read windows are settled. Without both, growth had no stopping point: a
  /// block that is not carrying scores the same constant full leak at any
  /// region size, so every shell was an exact tie, ties were kept, and the
  /// regions grew until they covered the whole complex and all blocks read one
  /// identical sub-complex.
  ///
  /// Creates no cells, edges, or vertices and never moves the cobordism's
  /// boundary: the only write is each block's vertex set.
  void growBlockRegions();
  /// Pre-grow the seed by `count` **gated cone-in moves** before any optimization
  /// (the constructor calls this once when `precone > 0`): each cones a fresh apex
  /// onto a random codim-1 facet of a random top cell and is committed only through
  /// `applyMoveSpecification`'s `dualComplexValid` gate — the same gate stage 1
  /// uses, so nothing is inserted by fiat. It enlarges the complex so
  /// surgery has room to act — the emergent analogue of a prebuilt host refinement.
  /// `count <= 0` is a no-op (RNG untouched). Best-effort: a draw onto an already-
  /// saturated facet is rejected by the gate and retried; if no valid cone-in is
  /// found for a cell, it stops early.
  /// `timelike` draws every cone timelike; `alternate` interleaves
  /// timelike/spacelike (and wins over `timelike`); default all-spacelike.
  void preconeCells(int count, bool timelike = false, bool alternate = false);

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
  /// #690: propagated to every spacetime this node constructs
  /// (host before precone, and each candidate snapshot rebuild).
  bool balancedEdgeWiring_{false};
  /// #697: `rU`'s whole-complex term is `singularValueHalfSumRatio` instead of
  /// the period residual + `nearKernelResidual` pair (see the constructor).
  bool singularValueRatio_{false};
  /// #724: false drops `‖∇S_Regge‖²` from every objective site (see the ctor).
  bool einsteinHilbert_{true};
  /// #737: latched by the first committed combinatorial move. Block regions
  /// grow only BEFORE the bulk is connected, so once a move has linked the
  /// complex up the boundary states' read windows are settled.
  bool bulkConnected_{false};
  /// The Einstein-Hilbert term of the objective, or 0 when it is switched off.
  /// One place, so `objective`, the stage-2 acceptance test, and `deltaF`
  /// cannot come to disagree about what F is.
  [[nodiscard]] double einsteinHilbertTerm(double beta = 1.0) const;
  /// Weight on the input-block residual terms in `rU` (see setInputResidualWeight).
  double inputResidualWeight_ = 1.0;
  /// An input region stops growing (growInputRegions) once its residual drops below
  /// this — i.e. once it carries its state.
  double inputCarriedTolerance_ = 1e-12;
  /// The move/restart random source driving stage 1 and block construction.
  std::mt19937_64 randomNumberGenerator_;
  /// #613: whether the move draw offers the disposition moves. See the accessor.
  bool shouldProposeDispositions_{true};
  double convergenceTolerance_ = 1e-9;
  /// Set by `runStage2`: `true` iff its last call stopped on the relative-tolerance
  /// stationarity test, `false` iff it hit the `maxIters` budget. See lastStage2Stationary.
  bool lastStage2Stationary_ = false;
  /// Set by `stage1Update`: the committed sequence's lookahead depth (see
  /// `lastStage1Lookahead`). 0 = the update committed nothing.
  int lastStage1LookaheadDepth_ = 0;
  std::vector<BoundaryBlock> inputBlocks_;
  std::vector<BoundaryBlock> outputBlocks_;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_MULTICOBORDISM_H
