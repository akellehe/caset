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
# `_tessera_rl` sets an RPATH to libtorch, so it imports without importing torch first.
from tessera._tessera_rl import *  # noqa: F401,F403
