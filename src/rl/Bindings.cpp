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

#include "rl/CobordismObjectiveEnv.h"

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
      .def("step", &CobordismObjectiveEnv::step, py::arg("move"), py::arg("params"))
      .def_property_readonly("obs_dim", &CobordismObjectiveEnv::obsDim)
      .def_property_readonly("num_moves", &CobordismObjectiveEnv::numMoves)
      .def_property_readonly("param_dim", &CobordismObjectiveEnv::paramDim)
      .def_property_readonly("current_F", &CobordismObjectiveEnv::currentF);

  m.def("make_formation_env", &makeFormationEnv, py::arg("config"),
        py::arg("input_weight") = 20.0,
        "The formation (2->1) env: the whole cobordism carries the proton singlet.");
  m.def("make_recombination_env", &makeRecombinationEnv, py::arg("config"),
        py::arg("input_weight") = 20.0,
        "The recombination (2->2) env: colored diquark + antidiquark; success = r_U -> 0.");
}
