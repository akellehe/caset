# Convenience build targets. The core library builds with a plain
# `pip install -e ".[dev]"` and needs nothing from this file; these targets exist
# for the optional subsystems whose build steps are less obvious.

.PHONY: rl

# Build the optional libtorch reinforcement-learning extension (`tessera.rl` /
# `_tessera_rl`, the PPO harness over MultiCobordism.buildStep).
#
# Why two commands: the extension is COMPILED against libtorch, so torch must be
# importable by the build interpreter. torch lives in the optional `[rl]` extra
# (it is heavy, and only the RL needs it) rather than in [build-system].requires,
# because a build requirement cannot be conditioned on which extra was requested
# — putting it there would force a ~2 GB torch download on every core-only build
# too. So we populate the environment first (step 1, an ordinary isolated build
# that skips the RL because the isolated build env can't see torch), then rebuild
# with --no-build-isolation (step 2) so the build interpreter sees the torch we
# just installed and compiles _tessera_rl.
rl:
	pip install -e ".[dev,rl]"
	pip install -e ".[rl]" --no-build-isolation
	@python -c "import tessera, tessera.rl; print('tessera.rl OK (torch_smoke =', tessera.rl.torch_smoke(), ')')"
