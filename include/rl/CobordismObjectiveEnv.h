// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.
//
// A Gym-style environment over the `MultiCobordism` objective search — the C++/libtorch
// port of `examples/cobordism/rl/objective_env.py` (#537/#546, ported in #551).
//
// This is a HARNESS, not a builder: MultiCobordism + Proton are the sole source of truth for
// proton construction, and this env only DRIVES them. Every macro-action is one
// `MultiCobordism::buildStep` (and, when `directedGrow`, the CANONICAL `directedConeOut`/
// `directedConeIn` probe, #550 — never a reimplementation); the observation and reward only
// READ published engine quantities (r_U, reggeActionGradient, emergentHoles, r_state, betti).
// The fixed `Proton.build()` schedule (init -> evolve -> relax) becomes the RL problem: the
// agent learns WHICH macro-action to take and with what parameters.

#ifndef TESSERA_RL_COBORDISM_OBJECTIVE_ENV_H
#define TESSERA_RL_COBORDISM_OBJECTIVE_ENV_H

#include <array>
#include <complex>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace tessera::cobordism { class MultiCobordism; }

namespace tessera::rl {

/// The discrete macro-moves (the policy's categorical head): GROW/EVOLVE are stage-1 surgery
/// passes (boundary-growing vs frozen-boundary evolution); RELAX is a stage-2 relaxation.
enum class Move { Grow = 0, Evolve = 1, Relax = 2 };
inline constexpr int kNumMoves = 3;
/// Continuous params per action (the Gaussian head): [intensity, knob], each clipped to [0,1].
inline constexpr int kParamDim = 2;
inline constexpr int kBettiSlots = 5;
/// Observation layout: 4 slog scalars + 5 Betti slots + hole count + 3 size counts + budget
/// fraction + kNumMoves last-move one-hot.
inline constexpr int kObsDim = 4 + kBettiSlots + 1 + 3 + 1 + kNumMoves;  // 17

/// `seed -> node` factory (Step A recombination or Step B formation).
using NodeFactory =
    std::function<std::shared_ptr<cobordism::MultiCobordism>(std::uint64_t)>;

/// Knobs mirroring the Python `CobordismObjectiveEnv`. The Python-probe-only knobs
/// (cone_strategy / max_candidates / overshoot / probe_openers) are gone: directed surgery
/// is now the canonical engine probe (`MultiCobordism::directedConeOut`/`directedConeIn`).
struct EnvConfig {
  int registerDegree = 3;
  double gamma = 50.0;
  int maxActions = 8;
  std::pair<int, int> growSteps = {2, 8};
  std::pair<int, int> evolveSteps = {2, 8};
  std::pair<int, int> relaxIters = {1, 4};
  std::pair<double, double> betaRange = {0.25, 2.0};
  std::pair<double, double> alphaRange = {0.02, 0.2};
  int nCandidateMoves = 6;
  int patience = 15;
  double carryTol = 0.5;
  int targetHoles = 3;
  double carryBonus = 3.0;
  double rewardScale = 1.0;
  double holeRewardWeight = 0.0;
  double rstateRewardWeight = 0.0;
  bool terminateOnCarry = true;
  bool directedGrow = false;
};

/// One env transition's outcome — the Python `(obs, reward, done, info)` tuple flattened.
struct StepResult {
  std::vector<float> obs;  // kObsDim
  double reward = 0.0;
  bool done = false;
  // --- info ---
  int move = -1;
  double F = 0.0;
  double deltaF = 0.0;
  double rU = 0.0;
  double rstate = 0.0;
  int holes = 0;
  bool carried = false;
  bool terminated = false;
  bool truncated = false;
  bool engineError = false;
};

/// A Gym-style RL environment over one `MultiCobordism` node's objective search. Faithful
/// port of `objective_env.py`: same 17-dim observation, hybrid action (move + 2 params), and
/// reward (slog ΔF + hole/r_state shaping + carry bonus + error penalty). Deterministic in
/// the reset seed.
class CobordismObjectiveEnv {
 public:
  /// `target` empty = recombination (success = r_U → 0); non-empty = a whole-cobordism target
  /// color state (e.g. the proton singlet), success = it is carried over ≥ targetHoles holes.
  CobordismObjectiveEnv(NodeFactory nodeFactory,
                        std::vector<std::complex<double>> target, EnvConfig config);

  /// Seed a fresh node on a single Δ⁴ simplex; return the initial observation.
  std::vector<float> reset(std::uint64_t seed);
  /// Apply ONE macro-action (move + params, clipped to [0,1]); return the transition.
  StepResult step(Move move, std::array<float, kParamDim> params);

  [[nodiscard]] int obsDim() const { return kObsDim; }
  [[nodiscard]] int numMoves() const { return kNumMoves; }
  [[nodiscard]] int paramDim() const { return kParamDim; }
  [[nodiscard]] double currentF() const { return currentF_; }
  [[nodiscard]] const std::shared_ptr<cobordism::MultiCobordism> &node() const {
    return node_;
  }

 private:
  struct Metrics {
    double F = 0.0, gradN2 = 0.0, rU = 0.0, rstate = 0.0;
    int holes = 0;
    std::vector<int> betti;
    int nVertices = 0, nEdges = 0, nTopCells = 0;
  };
  [[nodiscard]] Metrics computeMetrics() const;
  [[nodiscard]] bool isCarried(const Metrics &m) const;
  [[nodiscard]] std::vector<float> observation(const Metrics &m) const;

  NodeFactory nodeFactory_;
  std::vector<std::complex<double>> target_;  // empty = no whole-cobordism target
  EnvConfig config_;

  std::shared_ptr<cobordism::MultiCobordism> node_;
  std::uint64_t seed_ = 0;
  int stepsTaken_ = 0;
  double currentF_ = 0.0;
  int lastMove_ = -1;
  bool carried_ = false;
  Metrics lastMetrics_;
};

/// `seed -> node` factories mirroring `formation_node_factory` / `recombination_node_factory`:
/// Step B formation (2→1, the proton singlet carried by the whole) and Step A recombination
/// (2→2, a colored diquark ⊔ antidiquark; no whole-cobordism target). Both build the node via
/// `Proton` — the source of truth — so the RL drives the exact same setup `Proton.build()` does.
[[nodiscard]] NodeFactory formationNodeFactory(int registerDegree = 3, double gamma = 50.0,
                                               double inputWeight = 20.0);
[[nodiscard]] NodeFactory recombinationNodeFactory(int registerDegree = 3, double gamma = 50.0,
                                                   double inputWeight = 20.0);

/// Convenience env builders mirroring `make_formation_env` / `make_recombination_env`:
/// formation carries the proton singlet on the whole cobordism; recombination has no
/// whole-cobordism target (success = r_U → 0). Both wire the `Proton`-backed factory +
/// target internally, so the RL drives the exact node `Proton.build()` uses.
[[nodiscard]] CobordismObjectiveEnv makeFormationEnv(EnvConfig config, double inputWeight = 20.0);
[[nodiscard]] CobordismObjectiveEnv makeRecombinationEnv(EnvConfig config,
                                                         double inputWeight = 20.0);

}  // namespace tessera::rl

#endif  // TESSERA_RL_COBORDISM_OBJECTIVE_ENV_H
