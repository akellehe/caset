// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.
//
// The `_tessera_rl` Python module: the libtorch C++ reinforcement-learning policy that
// drives the canonical `MultiCobordism::buildStep` engine to assemble the proton (#551).
//
// This module links libtorch, so it is built ONLY when torch is importable in the build
// interpreter (the `rl` extra); the core `_tessera` stays libtorch-free. Imported as
// `tessera._tessera_rl` (a `tessera.rl` facade follows with the env + agent).

#include <pybind11/pybind11.h>
#include <torch/torch.h>

namespace py = pybind11;

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
}
