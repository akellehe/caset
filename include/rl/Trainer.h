// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.
//
// The C++ training loop + benchmark + checkpoint for the libtorch RL — the port of
// `examples/cobordism/rl/train.py` (#551). Trains the PPO policy over the RL harness
// (CobordismObjectiveEnv, which drives MultiCobordism + Proton) and compares it to the
// random + grow-only baselines on the proton carry criterion. Nothing here touches proton
// construction — it only orchestrates the env + agent.

#ifndef TESSERA_RL_TRAINER_H
#define TESSERA_RL_TRAINER_H

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "rl/CobordismObjectiveEnv.h"
#include "rl/PpoAgent.h"

namespace tessera::rl {

/// Training hyperparameters (train.py's benchmark args; defaults = the CARRY_PROFILE run).
struct TrainConfig {
  int iterations = 8;
  int episodesPerIter = 3;
  int evalSeeds = 6;
  int hidden = 64;
  double lr = 7e-4;
  int updateEpochs = 8;
  double entropyCoef = 0.03;
  double entropyCoefFinal = 0.005;  // < 0 disables annealing
  std::uint64_t agentSeed = 0;
  bool evalDeterministic = true;
};

/// Aggregate eval stats over held-out seeds (the proton criterion leads).
struct EvalSummary {
  double carryRate = 0.0;
  double meanHoles = 0.0;
  double meanRstate = 0.0;
  double meanFinalF = 0.0;
  double meanReward = 0.0;
};

/// One training iteration's diagnostics.
struct IterStat {
  int iteration = 0;
  double meanReturn = 0.0;
  double policyLoss = 0.0;
  double valueLoss = 0.0;
  double entropy = 0.0;
  double entropyCoef = 0.0;
};

/// The full benchmark: RL (learned) vs the random + grow-only baselines.
struct BenchmarkResult {
  std::vector<IterStat> history;
  EvalSummary rl;
  EvalSummary randomBaseline;
  EvalSummary growOnly;
  double trainTimeS = 0.0;
};

/// A macro-level policy: obs -> (move, params).
using PolicyFn = std::function<std::pair<Move, std::array<float, kParamDim>>(
    const std::vector<float> &)>;

/// The proton-carry env + train configs (train.py's CARRY_PROFILE).
[[nodiscard]] EnvConfig carryProfileEnv();
[[nodiscard]] TrainConfig carryProfileTrain();

/// Aggregate eval stats over `seeds` under `policy` (fresh node per seed).
[[nodiscard]] EvalSummary evaluate(CobordismObjectiveEnv &env, const PolicyFn &policy,
                                   const std::vector<std::uint64_t> &seeds);

/// Train PPO on the target (formation | recombination) + evaluate vs the random and
/// grow-only baselines. Saves the trained policy to `checkpointPath` (torch::save) if given.
[[nodiscard]] BenchmarkResult benchmark(EnvConfig envConfig, TrainConfig trainConfig,
                                        bool formation, const std::string &checkpointPath = "");

/// Load a policy checkpoint into a fresh HybridActorCritic (for eval / the animation).
[[nodiscard]] HybridActorCritic loadPolicy(const std::string &checkpointPath, int obsDim,
                                           int nMoves, int paramDim, int hidden = 64);

}  // namespace tessera::rl

#endif  // TESSERA_RL_TRAINER_H
