// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.
//
// The `_tessera_rl` Python module: the libtorch C++ reinforcement-learning policy that
// drives the canonical `MultiCobordism::buildStep` engine to assemble the proton (#551).
//
// This module links libtorch, so it is built ONLY when torch is importable in the build
// interpreter (the `rl` extra); the core `_tessera` stays libtorch-free. Imported as
// `tessera._tessera_rl` (a `tessera.rl` facade follows with the env + agent).

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>

// MultiCobordism is only forward-declared in CobordismObjectiveEnv.h, but the env exposes
// `node()` (a shared_ptr<MultiCobordism>) to Python, so pybind11 needs the complete type here.
#include "cobordism/MultiCobordism.h"
#include "rl/CobordismObjectiveEnv.h"
#include "rl/PpoAgent.h"
#include "rl/Trainer.h"

namespace py = pybind11;
using namespace tessera::rl;

PYBIND11_MODULE(_tessera_rl, m) {
  m.doc() = "tessera RL (libtorch PPO over MultiCobordism::buildStep) — #551.";

  // Foundation smoke: a tiny tensor op that proves libtorch is linked and runs at
  // import time. Returns 4.0 (the sum of a 2x2 ones tensor). The env + PPO agent land
  // on top of this in the following tasks.
  m.def(
      "torch_smoke",
      []() { return torch::ones({2, 2}).sum().item<double>(); },
      "Returns 4.0 (sum of a 2x2 ones tensor) — confirms libtorch links + runs.");

  m.attr("cxx11_abi") = static_cast<bool>(
#ifdef _GLIBCXX_USE_CXX11_ABI
      _GLIBCXX_USE_CXX11_ABI
#else
      true
#endif
  );

  // ---- The env: a HARNESS driving MultiCobordism + Proton (never a reimplementation) ----
  py::enum_<Move>(m, "Move", "The discrete macro-moves the policy chooses among.")
      .value("GROW", Move::Grow)
      .value("EVOLVE", Move::Evolve)
      .value("RELAX", Move::Relax);

  py::class_<EnvConfig>(m, "EnvConfig",
      "Knobs mirroring the Python CobordismObjectiveEnv (directed surgery is now the "
      "canonical engine probe, so cone_strategy/overshoot/etc. are gone).")
      .def(py::init<>())
      .def_readwrite("register_degree", &EnvConfig::registerDegree)
      .def_readwrite("gamma", &EnvConfig::gamma)
      .def_readwrite("max_actions", &EnvConfig::maxActions)
      .def_readwrite("grow_steps", &EnvConfig::growSteps)
      .def_readwrite("evolve_steps", &EnvConfig::evolveSteps)
      .def_readwrite("relax_iters", &EnvConfig::relaxIters)
      .def_readwrite("beta_range", &EnvConfig::betaRange)
      .def_readwrite("alpha_range", &EnvConfig::alphaRange)
      .def_readwrite("n_candidate_moves", &EnvConfig::nCandidateMoves)
      .def_readwrite("patience", &EnvConfig::patience)
      .def_readwrite("carry_tol", &EnvConfig::carryTol)
      .def_readwrite("target_holes", &EnvConfig::targetHoles)
      .def_readwrite("carry_bonus", &EnvConfig::carryBonus)
      .def_readwrite("reward_scale", &EnvConfig::rewardScale)
      .def_readwrite("hole_reward_weight", &EnvConfig::holeRewardWeight)
      .def_readwrite("rstate_reward_weight", &EnvConfig::rstateRewardWeight)
      .def_readwrite("terminate_on_carry", &EnvConfig::terminateOnCarry)
      .def_readwrite("directed_grow", &EnvConfig::directedGrow);

  py::class_<StepResult>(m, "StepResult", "One env transition (obs, reward, done + info).")
      .def_readonly("obs", &StepResult::obs)
      .def_readonly("reward", &StepResult::reward)
      .def_readonly("done", &StepResult::done)
      .def_readonly("move", &StepResult::move)
      .def_readonly("F", &StepResult::F)
      .def_readonly("delta_F", &StepResult::deltaF)
      .def_readonly("r_u", &StepResult::rU)
      .def_readonly("rstate", &StepResult::rstate)
      .def_readonly("holes", &StepResult::holes)
      .def_readonly("carried", &StepResult::carried)
      .def_readonly("terminated", &StepResult::terminated)
      .def_readonly("truncated", &StepResult::truncated)
      .def_readonly("engine_error", &StepResult::engineError);

  py::class_<CobordismObjectiveEnv>(m, "CobordismObjectiveEnv",
      "A Gym-style RL env over one MultiCobordism node's objective search. Every macro-action "
      "is one MultiCobordism::buildStep (+ the canonical directedConeOut/In when directed_grow); "
      "obs/reward only READ published engine quantities. Constructed via make_formation_env / "
      "make_recombination_env.")
      .def("reset", &CobordismObjectiveEnv::reset, py::arg("seed"))
      // step() drives one buildStep (long pure-C++ compute); release the GIL so a background
      // thread can run the RL-driven build without blocking the GUI (animation --rl --live).
      .def("step", &CobordismObjectiveEnv::step, py::arg("move"), py::arg("params"),
           py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("obs_dim", &CobordismObjectiveEnv::obsDim)
      .def_property_readonly("num_moves", &CobordismObjectiveEnv::numMoves)
      .def_property_readonly("param_dim", &CobordismObjectiveEnv::paramDim)
      .def_property_readonly("current_F", &CobordismObjectiveEnv::currentF)
      .def_property_readonly("node", &CobordismObjectiveEnv::node,
          "The env's current MultiCobordism node (for drawing / metrics — the animation "
          "reads node.st, node.objective(), emergent_holes(node.st, k), etc.).");

  m.def("make_formation_env", &makeFormationEnv, py::arg("config"),
        py::arg("input_weight") = 20.0,
        "The formation (2->1) env: the whole cobordism carries the proton singlet.");
  m.def("make_recombination_env", &makeRecombinationEnv, py::arg("config"),
        py::arg("input_weight") = 20.0,
        "The recombination (2->2) env: colored diquark + antidiquark; success = r_U -> 0.");

  // ---- PPO policy + training + checkpoint ----
  m.def("set_seed", &setSeed, py::arg("seed"),
        "Seed torch (network init + action sampling).");

  py::class_<ActOutput>(m, "ActOutput")
      .def_readonly("move", &ActOutput::move)
      .def_readonly("params", &ActOutput::params)
      .def_readonly("logp", &ActOutput::logp)
      .def_readonly("value", &ActOutput::value);

  // Opaque trained-policy handle (produced by load_policy; consumed by select_action). The
  // torch ModuleHolder is itself a copyable value type, so bind it as an opaque class (a
  // pybind holder template param would be a torch ModuleHolder, which pybind rejects).
  py::class_<HybridActorCritic>(m, "Policy");
  m.def("select_action", &selectPolicyAction, py::arg("policy"), py::arg("obs"),
        py::arg("deterministic") = true,
        "The greedy/sampled action for a (loaded) policy given a raw observation vector.");
  m.def("load_policy", &loadPolicy, py::arg("checkpoint_path"), py::arg("obs_dim"),
        py::arg("n_moves"), py::arg("param_dim"), py::arg("hidden") = 64,
        "Load a torch::save'd policy checkpoint into a fresh actor-critic.");

  py::class_<TrainConfig>(m, "TrainConfig")
      .def(py::init<>())
      .def_readwrite("iterations", &TrainConfig::iterations)
      .def_readwrite("episodes_per_iter", &TrainConfig::episodesPerIter)
      .def_readwrite("eval_seeds", &TrainConfig::evalSeeds)
      .def_readwrite("hidden", &TrainConfig::hidden)
      .def_readwrite("lr", &TrainConfig::lr)
      .def_readwrite("update_epochs", &TrainConfig::updateEpochs)
      .def_readwrite("entropy_coef", &TrainConfig::entropyCoef)
      .def_readwrite("entropy_coef_final", &TrainConfig::entropyCoefFinal)
      .def_readwrite("agent_seed", &TrainConfig::agentSeed)
      .def_readwrite("eval_deterministic", &TrainConfig::evalDeterministic);

  py::class_<EvalSummary>(m, "EvalSummary")
      .def_readonly("carry_rate", &EvalSummary::carryRate)
      .def_readonly("mean_holes", &EvalSummary::meanHoles)
      .def_readonly("mean_rstate", &EvalSummary::meanRstate)
      .def_readonly("mean_final_F", &EvalSummary::meanFinalF)
      .def_readonly("mean_reward", &EvalSummary::meanReward);

  py::class_<IterStat>(m, "IterStat")
      .def_readonly("iteration", &IterStat::iteration)
      .def_readonly("mean_return", &IterStat::meanReturn)
      .def_readonly("policy_loss", &IterStat::policyLoss)
      .def_readonly("value_loss", &IterStat::valueLoss)
      .def_readonly("entropy", &IterStat::entropy)
      .def_readonly("entropy_coef", &IterStat::entropyCoef);

  py::class_<BenchmarkResult>(m, "BenchmarkResult")
      .def_readonly("history", &BenchmarkResult::history)
      .def_readonly("rl", &BenchmarkResult::rl)
      .def_readonly("random", &BenchmarkResult::randomBaseline)
      .def_readonly("grow_only", &BenchmarkResult::growOnly)
      .def_readonly("train_time_s", &BenchmarkResult::trainTimeS);

  m.def("carry_profile_env", &carryProfileEnv,
        "The proton-carry EnvConfig (train.py's CARRY_PROFILE).");
  m.def("carry_profile_train", &carryProfileTrain,
        "The proton-carry TrainConfig (train.py's CARRY_PROFILE benchmark args).");
  m.def("benchmark", &benchmark, py::arg("env_config"), py::arg("train_config"),
        py::arg("formation") = true, py::arg("checkpoint_path") = "",
        // Release the GIL for the whole call. benchmark runs the PPO training loop, whose
        // backward pass drives libtorch's autograd engine — and libtorch (2.x) aborts if the
        // autograd engine is entered while the GIL is held. The call is pure C++ (it drives the
        // MultiCobordism engine + libtorch and never calls back into Python), so dropping the
        // GIL for its duration is safe and also lets other Python threads run during training.
        py::call_guard<py::gil_scoped_release>(),
        "Train PPO on the target + evaluate vs random/grow-only baselines; save the trained "
        "policy to checkpoint_path if given.");
}
