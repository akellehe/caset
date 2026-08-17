// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_PROTON_H
#define TESSERA_COBORDISM_PROTON_H

#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

namespace tessera::spacetime { class Spacetime; }

namespace tessera::cobordism {
using ::tessera::spacetime::Spacetime;

class MultiCobordism;  // returned (seeded, not run) by the node factories below

/// # Proton
///
/// A high-level, footgun-free builder for **the** emergent proton, composing
/// `MultiCobordism` (which it does not modify). A proton is **three quarks** in a
/// colorless bound state, so it is built in **two steps** — a single
/// `MultiCobordism` merge would be physically invalid:
///
///   * **Step A — recombination** (one co-optimized 2→2 node): two neutral q-q̄
///     pairs `{1,-1,0}` ⊔ `{1,0,-1}` → a **diquark** `{1,ω}` ⊔ an **antidiquark**
///     `{1,ω²}`. A diquark is **colored** (an SU(3) `3̄`), so its target is a
///     2-vector — emphatically *not* the singlet.
///   * **Step B — formation** (a separate 2→1 node): the diquark `{1,ω}` + the
///     third quark `{ω²}` → the **proton** `{1,ω,ω²}`, the colorless 3-vector
///     color singlet (ω = `exp(2πi/3)`). Mixed target dimensions are fine: each
///     boundary block's `r_state` fits its own target dimension.
///
/// Step B's output block is the proton "at a point in time" — its spatial slice,
/// with the **relaxed metric copied in**, is where downstream tickets read the
/// physical observables (charge/mass/radius/spin) **off** `block()`; those reads
/// are out of scope here.
///
/// `build()` grows each step out of a **single Δ⁴ simplex seed** (one pentatope — the
/// proton pre-builds nothing; all topology emerges from one simplex through stage 1's
/// F-lowering candidate draw),
/// runs **A then B**, and **restarts across distinct seeds** (the two-step converges
/// less often than a single merge) until step B's whole cobordism carries the 3-vector
/// singlet with at least `minQuarkHoles` (default 3) emergent color holes. The accessors
/// lazily trigger `build()` on first use, so `Proton p; auto b = p.block();` just works.
class Proton {
 public:
  /// ω = `exp(2πi/3)`, the unit color-charge phase.
  [[nodiscard]] static std::complex<double> omega();
  /// The proton color singlet `{1, ω, ω²}` — the colorless 3-vector that step B
  /// drives the proton block to carry.
  [[nodiscard]] static std::vector<std::complex<double>> singlet();

  /// Configure a proton build. Physics (the targets, the two-step structure) and the
  /// single Δ⁴ simplex seed are fixed; only the optimization knobs are exposed.
  ///   * `seed`           — base RNG seed; restart `i` uses A-seed `seed+2i`,
  ///                        B-seed `seed+2i+1` (A and B always distinct).
  ///   * `registerDegree` — the color register degree `k` (3 on a 4-manifold,
  ///                        where `ker L_{d-1}` is the register holes).
  ///   * `gamma`          — Γ in `F = ‖∇S_Regge‖² + Γ·r_U`, chosen so Γ·r_U sits on
  ///                        the same order as ‖∇S‖² (else ∇S trumps r_U and the
  ///                        register is never driven to carry).
  ///   * `inputWeight`    — weight on the input residuals so the diquark/quark
  ///                        inputs are driven to carry rather than dissolve.
  ///   * `precone`        — pre-grow each step's single-Δ⁴ seed by this many gated
  ///                        cone-in moves before optimization (forwarded to the
  ///                        `MultiCobordism` ctor of every node), giving surgery
  ///                        room to act without prebuilding a host. Default 0.
  ///   * `shouldUseDirectedSurgery` — when true, `build()` augments each step's drive with
  ///                        the engine's DIRECTED cone-out/cone-in probes
  ///                        (`MultiCobordism::directedConeOut`/`directedConeIn`): a gated,
  ///                        score-guided search for the cells
  ///                        to open the register holes (and the holes to cap), instead of
  ///                        relying on `runStage1`'s random cone draws. Default false keeps
  ///                        `build()` byte-identical to its existing drive.
  ///   * `preconeTimelike` — draw every precone cone-in as the TIMELIKE
  ///                        disposition (#613).
  ///   * `preconeAlternate` — instead alternate the cone-ins timelike/spacelike
  ///                        for balanced causal content at one uniform
  ///                        edge-length magnitude (wins over `preconeTimelike`).
  ///   * `singularValueRatio` — every node scores the whole-complex term of
  ///                        `rU` with the scale-invariant singular-value
  ///                        half-sum ratio instead of the singlet period
  ///                        residual + near-kernel pair
  ///                        (`MultiCobordism::singularValueHalfSumRatio`); the
  ///                        singlet stays the after-the-fact readout verdict.
  explicit Proton(std::uint64_t seed = 0, int registerDegree = 3,
                  double gamma = 50.0, double inputWeight = 20.0,
                  int precone = 0, bool shouldUseDirectedSurgery = false,
                  bool preconeTimelike = false, bool preconeAlternate = false,
                  bool balancedEdges = false, bool singularValueRatio = false);

  /// Build the proton, restarting across seeds until step B's whole cobordism
  /// carries the singlet with `≥ minQuarkHoles` holes (or `maxRestarts` is
  /// exhausted, keeping the best attempt; `converged()` is then false). Each step
  /// runs an INITIALIZATION pass (`initSteps`, `grow_boundaries=true` — establishes
  /// the carrying input regions) then an EVOLUTION pass (`evolveSteps`,
  /// `grow_boundaries=false` — ∂W frozen) then `runStage2`. Idempotent.
  void build(int maxRestarts = 16, int initSteps = 180,
             int evolveSteps = 60, int stage1CandidateMoves = 8,
             double stage2Beta = 1.0, int stage2MaxIters = 10,
             double colorTolerance = 0.5, int minQuarkHoles = 3);

  /// EXPERIMENTAL one-step build: drive `directNode` — three q-q̄ pairs in, the
  /// singlet out, in ONE `MultiCobordism` — with the combined `MultiCobordism::run`
  /// drive, which interleaves the stage-1 surgery update and the stage-2 geometric
  /// relaxation in one loop (the drive `multicobordism_animation.py` uses): an init
  /// pass (`initSteps`, `grow_boundaries=true`) then an evolution pass (`evolveSteps`,
  /// `grow_boundaries=false` — ∂W frozen). Restarts across seeds (restart `i` uses
  /// `seed + i`) until the whole cobordism carries the singlet with `≥ minQuarkHoles`
  /// holes, or `maxRestarts` is exhausted keeping the best attempt. Populates the same
  /// accessors as `build()` (`diquarkResidual()` stays 0 — there is no step A).
  /// Idempotent, and shared with `build()`: whichever runs first claims the build
  /// state, so call this BEFORE any accessor triggers the lazy two-step `build()`.
  void buildDirect(int maxRestarts = 16, int initSteps = 180, int evolveSteps = 60,
                   int stage1CandidateMoves = 8, double stage2Beta = 1.0,
                   double colorTolerance = 0.5, int minQuarkHoles = 3);

  /// A fresh, seeded — but NOT yet run — Step A node (recombination, 2→2): two neutral
  /// q-q̄ pairs `{1,-1,0}` ⊔ `{1,0,-1}` → a diquark `{1,ω}` ⊔ antidiquark `{1,ω²}`, on a
  /// single Δ⁴ seed (inputs at v0,v1; outputs at v2,v3; input weight set). `build()` and
  /// the animation both drive this exact setup via `runStage1`/`runStage2` — the single
  /// source of truth for the recombination step.
  [[nodiscard]] std::shared_ptr<MultiCobordism> recombinationNode(std::uint64_t seed) const;
  /// A fresh, seeded — but NOT yet run — Step B node (formation, 2→1): the diquark
  /// `{1,ω}` + the third quark `{ω²}` → the proton singlet `{1,ω,ω²}`, on a single Δ⁴
  /// seed (inputs at v0,v1; the single output is read off the WHOLE, so no `seedOutputs`).
  [[nodiscard]] std::shared_ptr<MultiCobordism> formationNode(std::uint64_t seed) const;
  /// A fresh, seeded — but NOT yet run — ONE-STEP node (6→1): the three bare quarks
  /// `{1}`, `{ω}`, `{ω²}` AND their three anti-quarks `{1}`, `{ω̄}`, `{ω̄²}` (the
  /// elementwise conjugates — the antiparticle convention here, as antidiquark =
  /// conj(diquark)) as inputs on a single Δ⁴ seed, so the prepared content is three
  /// q-q̄ pairs. The proton singlet `{1,ω,ω²}` is the single output, read off the
  /// WHOLE cobordism (no `seedOutputs`) — the anti-baryon partner is left to emerge
  /// unpinned. The experimental single-merge alternative to the two-step build.
  [[nodiscard]] std::shared_ptr<MultiCobordism> directNode(std::uint64_t seed) const;

  /// True iff the whole step-B cobordism carries the singlet (`colorResidual() <
  /// colorTolerance`) with `≥ minQuarkHoles` holes. Triggers `build()`.
  [[nodiscard]] bool converged();
  /// The base seed of the converged (or best) attempt. Triggers `build()`.
  [[nodiscard]] std::uint64_t seed();
  /// The full relaxed emergent complex of step B (proton formation), grown from the
  /// single Δ⁴ seed. Triggers `build()`.
  [[nodiscard]] std::shared_ptr<Spacetime> spacetime();
  /// The proton itself: the relaxed WHOLE step-B cobordism. The single output IS the
  /// whole's harmonic (the inputs are held by their residual, the bulk evolves to
  /// carry the singlet), so there is no sub-block — this is what downstream
  /// observable readers consume. Triggers `build()`.
  [[nodiscard]] std::shared_ptr<Spacetime> block();
  /// The emergent color holes (`(k+2)`-vertex tuples) on the proton (the whole) —
  /// `≥ minQuarkHoles` when converged. Triggers `build()`.
  [[nodiscard]] std::vector<std::vector<std::uint64_t>> quarkHoles();
  /// The proton singlet residual — the relabeling-invariant, zero-filled `r_state`
  /// of `singlet()` against the WHOLE cobordism's `L_k` harmonic (`≈0` ⇒ carried).
  /// Triggers `build()`.
  [[nodiscard]] double colorResidual();
  /// Step A's realizability residual `r_U` — small ⇒ the diquark recombination
  /// converged (a separate physical claim from the proton's formation). Triggers
  /// `build()`.
  [[nodiscard]] double diquarkResidual();

 private:
  /// Lazily run `build()` with default parameters on first accessor use.
  void ensureBuilt();
  /// The minimal seed: a single `Δ⁴` simplex (one pentatope — 5 vertices, 1 top cell,
  /// Betti `[1,0,0,0,0]`, a contractible 4-ball) with a uniform metric. The proton grows
  /// ALL of its topology out of this one simplex through stage 1's F-lowering candidate
  /// draw, and the geometry out of the relaxation — nothing is pre-built (no host
  /// refinement, no metric jitter).
  [[nodiscard]] static std::shared_ptr<Spacetime> buildMinimalSeed(
      bool balancedEdges = false);

  // ---- configuration ----
  std::uint64_t baseSeed_;
  int registerDegree_;
  double gamma_;
  bool balancedEdges_{false};
  bool singularValueRatio_{false};  // forwarded to every node (see the ctor)
  double inputResidualWeight_;
  int precone_;  // gated cone-ins pre-grown into each node's seed (ctor → nodes)
  bool shouldUseDirectedSurgery_;  // build() uses the directed cone-out/cone-in probes
  bool preconeTimelike_;  // precone cone-ins drawn as the timelike disposition
  bool preconeAlternate_;  // precone cone-ins alternate timelike/spacelike

  // ---- build state (populated by build()) ----
  bool attempted_ = false;
  bool converged_ = false;
  std::uint64_t convergedSeed_ = 0;
  std::shared_ptr<Spacetime> spacetime_;  // step B's full relaxed complex
  std::shared_ptr<Spacetime> block_;      // proton sub-complex, relaxed metric
  std::vector<std::vector<std::uint64_t>> quarkHoles_;
  double colorResidual_ = 0.0;
  double diquarkResidual_ = 0.0;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_PROTON_H
