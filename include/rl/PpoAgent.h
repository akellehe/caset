// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.
//
// A libtorch PPO actor-critic for the cobordism objective-search policy — the C++ port of
// `examples/cobordism/rl/ppo_agent.py` (#551). Engine-agnostic: it only sees observation
// vectors and the `(obs, reward, done, info)` contract of `CobordismObjectiveEnv` (which is
// the harness that drives MultiCobordism + Proton). libtorch has no `torch::distributions`,
// so Categorical/Normal log_prob/entropy/sample are computed explicitly.

#ifndef TESSERA_RL_PPO_AGENT_H
#define TESSERA_RL_PPO_AGENT_H

#include <array>
#include <cstdint>
#include <random>
#include <tuple>
#include <vector>

#include <torch/torch.h>

#include "rl/CobordismObjectiveEnv.h"

namespace tessera::rl {

/// Seed torch (network init + action sampling) for reproducibility. The env's engine RNG is
/// seeded separately (per `reset(seed)`).
void setSeed(std::uint64_t seed);

/// One sampled action: the categorical move + Gaussian params, with their joint log-prob and
/// the critic value (the rollout path, computed under no_grad).
struct ActOutput {
  int move = 0;
  std::array<float, kParamDim> params{};
  double logp = 0.0;
  double value = 0.0;
};

/// Shared-trunk actor-critic: MLP trunk (Tanh) -> categorical move head + diagonal-Gaussian
/// param head (state-independent log-σ, init -0.5) + value head. Faithful to
/// `ppo_agent.py`'s HybridActorCritic.
struct HybridActorCriticImpl : torch::nn::Module {
  HybridActorCriticImpl(int obsDim, int nMoves, int paramDim, int hidden = 64);

  /// (moveLogits, paramMean, paramStd, value) for a batch of observations.
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(
      torch::Tensor obs);
  /// Sample (or, if deterministic, take the mode of) one action for a single [1, obsDim] obs.
  ActOutput act(torch::Tensor obs, bool deterministic = false);
  /// The critic value for a single [1, obsDim] obs.
  double valueOf(torch::Tensor obs);
  /// Joint log-prob, joint entropy, and value for a batch — the PPO update path.
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> evaluateActions(
      torch::Tensor obs, torch::Tensor moves, torch::Tensor params);

  int nMoves_ = 0;
  int paramDim_ = 0;
  torch::nn::Linear trunk1{nullptr}, trunk2{nullptr}, moveHead{nullptr},
      paramMeanHead{nullptr}, valueHead{nullptr};
  torch::Tensor paramLogStd;
};
TORCH_MODULE(HybridActorCritic);

/// One rollout transition (GAE advantage + return are filled in after the episode).
struct Transition {
  std::vector<float> obs;
  int move = 0;
  std::array<float, kParamDim> params{};
  double logp = 0.0;
  double value = 0.0;
  double reward = 0.0;
  double advantage = 0.0;
  double ret = 0.0;
};

/// One PPO update's last-minibatch diagnostics.
struct UpdateStats {
  double policyLoss = 0.0;
  double valueLoss = 0.0;
  double entropy = 0.0;
};

/// Proximal Policy Optimization for the hybrid policy — GAE, clipped surrogate, value +
/// entropy terms, minibatch SGD. Faithful to `ppo_agent.py`'s PPO.
class PPO {
 public:
  PPO(int obsDim, int nMoves, int paramDim, int hidden = 64, double lr = 3e-4,
      double gamma = 0.99, double lam = 0.95, double clip = 0.2, double valueCoef = 0.5,
      double entropyCoef = 0.01, int updateEpochs = 6, int minibatchSize = 64,
      double maxGradNorm = 0.5);

  /// Run one full episode under the current policy; fill GAE advantages + returns; return the
  /// transitions and (out) the final `StepResult` (for benchmark metrics + terminal bootstrap).
  std::vector<Transition> collectEpisode(CobordismObjectiveEnv &env, std::uint64_t seed,
                                         StepResult &finalInfo);
  /// One PPO update over a batch of transitions (updateEpochs of minibatch SGD).
  UpdateStats update(std::vector<Transition> &transitions);
  /// The greedy (or sampled) action for evaluation.
  ActOutput selectAction(const std::vector<float> &obs, bool deterministic = true);

  HybridActorCritic policy;
  double entropyCoef;  // public: the training loop anneals it across a run

 private:
  void finishGae(std::vector<Transition> &transitions, double lastValue) const;

  double gamma_, lam_, clip_, valueCoef_, maxGradNorm_;
  int updateEpochs_, minibatchSize_;
  torch::optim::Adam optimizer_;
  std::mt19937 shuffleRng_;
};

/// The greedy/sampled action for a (possibly loaded) bare policy given a raw observation
/// vector — the eval / animation path (no PPO instance needed).
[[nodiscard]] ActOutput selectPolicyAction(HybridActorCritic policy,
                                           const std::vector<float> &obs,
                                           bool deterministic = true);

}  // namespace tessera::rl

#endif  // TESSERA_RL_PPO_AGENT_H
