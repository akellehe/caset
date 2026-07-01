# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The libtorch reinforcement-learning harness — PPO over ``MultiCobordism.buildStep``.

Optional: requires the ``rl`` extra (libtorch), which builds the ``_tessera_rl`` extension.
Everything here is a HARNESS that DRIVES the canonical ``MultiCobordism`` + ``Proton`` engine
(the sole source of truth for proton construction) — it never reimplements construction.

Import as ``tessera.rl``::

    import tessera
    cfg = tessera.rl.carry_profile_env()
    res = tessera.rl.benchmark(cfg, tessera.rl.carry_profile_train(), formation=True,
                               checkpoint_path="proton_policy.pt")
"""
import os

# tessera_core (loaded by `import tessera`) links Homebrew's LLVM libomp; libtorch ships its
# own copy of the *same* LLVM OpenMP runtime. Loading `_tessera_rl` pulls libtorch's libomp
# into a process that already has the core's, so TWO OpenMP runtimes coexist. That breaks in
# two stages: the second runtime's init aborts the process ("OMP: Error #15 ... multiple
# copies of the OpenMP runtime"), and merely silencing that abort is not enough — the two
# runtimes then fight over a shared thread pool and DEADLOCK inside any threaded region (e.g.
# libtorch's log_softmax during PPO.update parks forever in __kmp_join_barrier).
#
# Both fixes below are needed and both use setdefault so an explicit environment override
# still wins. KMP_DUPLICATE_LIB_OK tolerates the duplicate init (benign: same implementation).
# OMP_NUM_THREADS=1 keeps every OpenMP region single-threaded, which sidesteps the cross-
# runtime barrier deadlock entirely. This is tessera's default scan policy anyway (the core's
# heat-kernel loop is bit-identical to serial at 1 thread), and this RL workload is bound by
# MultiCobordism.buildStep in the core, not by libtorch ops on the tiny actor-critic — so the
# single-thread cap costs effectively nothing here.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# `_tessera_rl` sets an RPATH to libtorch, so it imports without importing torch first.
from tessera._tessera_rl import *  # noqa: E402,F401,F403
