// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_MODULARITYOPTIMIZER_H
#define TESSERA_OBSERVABLES_MODULARITYOPTIMIZER_H

#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "observables/PersistentModularity.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
// === cross-subsystem fwd-decls ===
namespace tessera::simulations {
  class CDT;
}
namespace tessera::spacetime {
  class PachnerMove;
  class Spacetime;
}
namespace tessera::observables {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;


/// One recorded measurement on the (Q, D_S) trajectory.
///
/// Mirrors ``examples/modularity.py:Measurement``.
struct ModularityMeasurement {
  double Q;            ///< Newman-Girvan modularity at the time of measurement.
  double dsSmall;      ///< Spectral dimension at small diffusion times.
  double dsLarge;      ///< Spectral dimension at large diffusion times.
  std::size_t nVertices;
  std::size_t nEdges;
  std::size_t nSimplices;
  int iter;            ///< Sweep iteration index.
  std::string direction;  ///< "up" | "down"
};

/// Configuration for :class:`ModularityOptimizer`.
struct ModularityOptimizerConfig {
  double targetDq = 0.05;          ///< Q increment between measurements.
  int maxIterations = 400;         ///< Hard cap per sweep direction.
  int nDiffusionWalks = 80;        ///< Diffusion start nodes per measurement.
  double maxSigma = 200.0;         ///< Upper bound of log-spaced t grid.
  int negativeRetryMax = 10;       ///< Max retries when D_S comes back negative.
  double epsilonQMax = 0.01;       ///< Up-sweep early-exit tolerance.
  int krylovDim = 30;              ///< Krylov subspace dim for heat kernel.
  int targetNModules = 4;          ///< M (modulo partition).
};

/// Modularity sweep on a CDT spacetime, driven by transactional
/// Pachner moves with Q-direction acceptance.
///
/// **Score domain and status.**  Every modularity number this class
/// produces — the sweep's Q trajectory (via
/// ``Spacetime::modularityOnSkeleton``) and the label-free discovery
/// reachable through :func:`discoverComponents` — is a Newman-Girvan /
/// generalized-modularity score evaluated on a combinatorial /
/// nonnegative one-skeleton.  It is blind to signed and complex Hodge
/// weights, so it is a heuristic proposal generator only: it may
/// propose candidate component supports, it never enters the emergence
/// objective, and it may not veto an otherwise certified fiber.  Fiber
/// acceptance rests solely on the independent weight-aware
/// gap/localization/leakage/persistence/anchor certificates.
///
/// Algorithm (per iteration):
///   1. Pick a random move type from {add, remove, flip, iflip, shift}.
///   2. ``cdt.proposeXxx()`` — read-only target selection.  If no
///      eligible target, try another move type (one fallback each
///      iteration).
///   3. Snapshot Q on the spacetime 1-skeleton.
///   4. ``move.apply()`` — commit the move.
///   5. Compute Q after.  If direction matches (up: Q rose; down: Q
///      fell), keep the move.  Otherwise ``move.rollback()``.
///   6. If Q crossed the next ``target_dq`` threshold, build the dual
///      graph and measure D_S; record a Measurement.
///
/// The sweep's fixed-partition read (community = vertex id mod M) is
/// unchanged and remains available; label-free discovery is the
/// :class:`PersistentModularity` extension below.
///
/// The "informed proposal" hook in ``selectMoveType`` can bias the
/// move-type distribution toward those most likely to move Q in the
/// target direction.  Default: uniform over all 5 types.
class ModularityOptimizer {
public:
  /// Progress callback signature: (iter, maxIter, currentQ, n_meas).
  using ProgressCallback =
      std::function<void(int, int, double, std::size_t)>;

  ModularityOptimizer(ModularityOptimizerConfig cfg, std::uint64_t seed)
      : cfg_(cfg), rng_(seed) {}

  /// Drive the spacetime via ``cdt`` Pachner moves to walk Q in the
  /// given ``direction`` ("up" or "down").  Records (Q, D_S) at every
  /// ``targetDq`` threshold crossing.  Mutates ``cdt``'s spacetime
  /// in place.  Resets the per-sweep counters before running.
  std::vector<ModularityMeasurement> sweep(
      CDT &cdt,
      const std::string &direction,
      ProgressCallback progress = nullptr);

  /// Label-free discovery of persistent modular components on the CURRENT
  /// spacetime one-skeleton (ticket #765, design spec section 8): builds
  /// the nonnegative similarity graph under ``map`` and delegates to
  /// :func:`PersistentModularity::scanResolutions`.  Read-only — never
  /// mutates the spacetime, never proposes or applies moves, and its
  /// result must never feed the emergence objective (heuristic proposal
  /// generator; see the class documentation).  Deterministic for a fixed
  /// ``cfg`` seed sequence; the optimizer's own RNG is untouched.
  ScanReport discoverComponents(
      const Spacetime &st, const PersistentModularityConfig &cfg,
      PersistentModularity::WeightMap map =
          PersistentModularity::WeightMap::ExpNegAbsLength) const;

  // Per-sweep counters.  Reset at the top of each ``sweep()`` call.
  /// Number of moves applied + kept (Q moved in the desired direction).
  std::int64_t getNAccepted() const noexcept { return nAccepted_; }
  /// Number of moves applied then rolled back (Q moved the wrong way).
  std::int64_t getNRolledBack() const noexcept { return nRolledBack_; }
  /// Number of iterations with no eligible Pachner-move proposal.
  std::int64_t getNNoMove() const noexcept { return nNoMove_; }
  /// Number of D_S measurements taken (equal to len(sweep result)).
  std::int64_t getNMeasurements() const noexcept { return nMeasurements_; }

private:
  ModularityOptimizerConfig cfg_;
  std::mt19937 rng_;
  std::int64_t nAccepted_ = 0;
  std::int64_t nRolledBack_ = 0;
  std::int64_t nNoMove_ = 0;
  std::int64_t nMeasurements_ = 0;

  // Try each move type in turn (random order) until one validates
  // via propose().  Returns nullptr if all 5 fail.
  std::unique_ptr<PachnerMove> proposeAny(CDT &cdt);

  // Measure D_S on the dual graph; recompute Q on the 1-skeleton.
  // The "negative D_S retry" safety net is built in.
  ModularityMeasurement measure(
      CDT &cdt, int iter, const std::string &direction);
};

}  // namespace tessera

#endif  // TESSERA_OBSERVABLES_MODULARITYOPTIMIZER_H
