// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_PROTON_H
#define TESSERA_COBORDISM_PROTON_H

#include <complex>
#include <cstdint>
#include <memory>
#include <set>
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
/// `build()` builds the closed-S⁴ hosts internally (a port of
/// `examples/cobordism/emergent_optimizer.build_closed_s4`), runs **A then B**,
/// and **restarts across distinct seeds** (the two-step converges less often than
/// a single merge) until step B's proton block carries the 3-vector singlet with
/// at least `minQuarkHoles` (default 3) emergent color holes. The accessors
/// lazily trigger `build()` on first use, so `Proton p; auto b = p.block();`
/// just works.
class Proton {
 public:
  /// ω = `exp(2πi/3)`, the unit color-charge phase.
  [[nodiscard]] static std::complex<double> omega();
  /// The proton color singlet `{1, ω, ω²}` — the colorless 3-vector that step B
  /// drives the proton block to carry.
  [[nodiscard]] static std::vector<std::complex<double>> singlet();

  /// Configure a proton build. Physics (the targets, the two-step structure) is
  /// fixed; only the substrate/optimization knobs are exposed.
  ///   * `seed`           — base RNG seed; restart `i` uses A-seed `seed+2i`,
  ///                        B-seed `seed+2i+1` (A and B always distinct).
  ///   * `hostRefinement` — PreGeometric Pachner refinements of each closed-S⁴
  ///                        host (surgery room).
  ///   * `registerDegree` — the color register degree `k` (3 on a 4-manifold,
  ///                        where `ker L_{d-1}` is the register holes).
  ///   * `gamma`          — Γ in `F = ‖∇S_Regge‖² + Γ·r_U`.
  explicit Proton(std::uint64_t seed = 0, int hostRefinement = 14,
                  int registerDegree = 3, double gamma = 1.0);

  /// Build the proton: run step A then step B, restarting across seeds until step
  /// B's proton block carries the singlet with `≥ minQuarkHoles` holes (or
  /// `maxRestarts` is exhausted, in which case the best attempt is kept and
  /// `converged()` is false). Idempotent — a second call is a no-op. The stage
  /// parameters mirror `MultiCobordism::runStage1`/`runStage2`.
  void build(int maxRestarts = 16, int constructRounds = 12,
             int stage1MaxSteps = 30, int stage1Candidates = 8,
             int stage1Patience = 8, double stage2Beta = 1.0,
             int stage2MaxIters = 20, double colorTolerance = 0.5,
             int minQuarkHoles = 3);

  /// True iff step B's proton block carries the singlet (`colorResidual() <
  /// colorTolerance`) with `≥ minQuarkHoles` holes. Triggers `build()`.
  [[nodiscard]] bool converged();
  /// The base seed of the converged (or best) attempt. Triggers `build()`.
  [[nodiscard]] std::uint64_t seed();
  /// The full relaxed closed-S⁴ complex of step B (proton formation). Triggers
  /// `build()`.
  [[nodiscard]] std::shared_ptr<Spacetime> spacetime();
  /// Step B's proton sub-complex — the top cells of the formation block carved
  /// out with the **relaxed metric copied in** (not the unit-metric `subOf`).
  /// This is what downstream observable readers consume. Triggers `build()`.
  [[nodiscard]] std::shared_ptr<Spacetime> block();
  /// The emergent color holes (`(k+2)`-vertex tuples) on the proton block —
  /// `≥ minQuarkHoles` when converged. Triggers `build()`.
  [[nodiscard]] std::vector<std::vector<std::uint64_t>> quarkHoles();
  /// Step B's proton singlet residual — the relabeling-invariant, zero-filled
  /// `r_state` of `singlet()` against the block's `L_k` harmonic (`≈0` ⇒
  /// carried). Triggers `build()`.
  [[nodiscard]] double colorResidual();
  /// Step A's realizability residual `r_U` — small ⇒ the diquark recombination
  /// converged (a separate physical claim from the proton's formation). Triggers
  /// `build()`.
  [[nodiscard]] double diquarkResidual();

 private:
  void ensureBuilt();

  // ---- configuration ----
  std::uint64_t baseSeed_;
  int hostRefinement_;
  int registerDegree_;
  double gamma_;

  // ---- build state (populated by build()) ----
  bool attempted_ = false;
  bool converged_ = false;
  std::uint64_t convergedSeed_ = 0;
  std::shared_ptr<Spacetime> spacetime_;  // step B's full relaxed complex
  std::shared_ptr<Spacetime> block_;      // proton sub-complex, relaxed metric
  std::vector<std::vector<std::uint64_t>> quarkHoles_;
  double colorResidual_ = 0.0;
  double diquarkResidual_ = 0.0;

  // Stored stage parameters so the lazy accessors can run build() with the
  // configuration the first caller (if any) chose.
  int maxRestarts_ = 16;
  int constructRounds_ = 12;
  int stage1MaxSteps_ = 30;
  int stage1Candidates_ = 8;
  int stage1Patience_ = 8;
  double stage2Beta_ = 1.0;
  int stage2MaxIters_ = 20;
  double colorTolerance_ = 0.5;
  int minQuarkHoles_ = 3;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_PROTON_H
