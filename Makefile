# Convenience build targets. The core library builds with a plain
# `pip install -e ".[dev]"` and needs nothing from this file; these targets exist
# for the optional subsystems whose build steps are less obvious.

.PHONY: rl

# Build the optional libtorch reinforcement-learning extension (`tessera.rl` /
# `_tessera_rl`, the PPO harness over MultiCobordism.buildStep).
#
# One command, standard build isolation. The extension is COMPILED against libtorch,
# so torch must be visible to the build interpreter — but torch cannot go in
# [build-system].requires (that would force a multi-GB torch download into every
# core-only build too) and an extra ([rl]) is invisible to the build backend. So the
# TESSERA_RL env var gates a scikit-build-core override (see pyproject.toml) that adds
# torch to the ISOLATED build environment on demand; find_package(Torch) then succeeds
# and CMake compiles _tessera_rl. Core builds stay torch-free.
#
# For rapid C++ iteration (recompiling _tessera_rl repeatedly), skip the per-build torch
# install by reusing an env that already has it:
#     pip install -e ".[dev,rl]"                      # once: build backend + torch
#     pip install -e ".[rl]" --no-build-isolation     # fast rebuilds thereafter
rl:
	TESSERA_RL=1 pip install -e ".[rl]"
	@python -c "import tessera, tessera.rl; print('tessera.rl OK (torch_smoke =', tessera.rl.torch_smoke(), ')')"
