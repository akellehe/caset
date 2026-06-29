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
/// `build()` grows each step out of a **bare ∂Δ⁵ minimal seed** (the proton never
/// pre-refines its own host — all topology emerges via the trap door), runs **A then
/// B**, and **restarts across distinct seeds** (the two-step converges less often than
/// a single merge) until step B's whole cobordism carries the 3-vector singlet with at
/// least `minQuarkHoles` (default 3) emergent color holes. The accessors lazily trigger
/// `build()` on first use, so `Proton p; auto b = p.block();` just works.
class Proton {
 public:
  /// ω = `exp(2πi/3)`, the unit color-charge phase.
  [[nodiscard]] static std::complex<double> omega();
  /// The proton color singlet `{1, ω, ω²}` — the colorless 3-vector that step B
  /// drives the proton block to carry.
  [[nodiscard]] static std::vector<std::complex<double>> singlet();

  /// Configure a proton build. Physics (the targets, the two-step structure) and the
  /// bare ∂Δ⁵ minimal seed are fixed; only the optimization knobs are exposed.
  ///   * `seed`           — base RNG seed; restart `i` uses A-seed `seed+2i`,
  ///                        B-seed `seed+2i+1` (A and B always distinct).
  ///   * `registerDegree` — the color register degree `k` (3 on a 4-manifold,
  ///                        where `ker L_{d-1}` is the register holes).
  ///   * `gamma`          — Γ in `F = ‖∇S_Regge‖² + Γ·r_U`, chosen so Γ·r_U sits on
  ///                        the same order as ‖∇S‖² (else ∇S trumps r_U and the
  ///                        register is never driven to carry).
  ///   * `inputWeight`    — weight on the input residuals so the diquark/quark
  ///                        inputs are driven to carry rather than dissolve.
  explicit Proton(std::uint64_t seed = 0, int registerDegree = 3,
                  double gamma = 50.0, double inputWeight = 20.0);

  /// Build the proton, restarting across seeds until step B's whole cobordism
  /// carries the singlet with `≥ minQuarkHoles` holes (or `maxRestarts` is
  /// exhausted, keeping the best attempt; `converged()` is then false). Each step
  /// runs an INITIALIZATION pass (`initSteps`, `grow_boundaries=true` — establishes
  /// the carrying input regions) then an EVOLUTION pass (`evolveSteps`,
  /// `grow_boundaries=false` — ∂W frozen) then `runStage2`. Idempotent.
  void build(int maxRestarts = 16, int initSteps = 180,
             int evolveSteps = 60, int stage1CandidateMoves = 8, int stage1Patience = 15,
             double stage2Beta = 1.0, int stage2MaxIters = 10,
             double colorTolerance = 0.5, int minQuarkHoles = 3);

  /// True iff the whole step-B cobordism carries the singlet (`colorResidual() <
  /// colorTolerance`) with `≥ minQuarkHoles` holes. Triggers `build()`.
  [[nodiscard]] bool converged();
  /// The base seed of the converged (or best) attempt. Triggers `build()`.
  [[nodiscard]] std::uint64_t seed();
  /// The full relaxed closed-S⁴ complex of step B (proton formation). Triggers
  /// `build()`.
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
  /// The minimal closed seed: a bare `∂Δ⁵` sphere (S⁴, Betti `[1,0,0,0,1]`, six
  /// pentatopes) with a mild deterministic non-uniform metric. The proton grows all of
  /// its topology out of this via the trap door — there is deliberately no host
  /// refinement (that would pre-build topology that must instead emerge).
  [[nodiscard]] static std::shared_ptr<Spacetime> buildMinimalSeed();

  // ---- configuration ----
  std::uint64_t baseSeed_;
  int registerDegree_;
  double gamma_;
  double inputResidualWeight_;

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
