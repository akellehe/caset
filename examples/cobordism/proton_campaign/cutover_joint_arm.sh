#!/usr/bin/env bash
# Campaign cutover to the joint three-pair arm (#585): stop the running generation,
# preserve its data, rebuild the campaign worktree's engine at the generation
# boundary, and relaunch 16 workers on the joint drive with the SAME deadline.
#
# DELIBERATELY MANUAL: step 1 kills in-flight attempts (hours of compute each) —
# run this only on an explicit go. Everything it does is idempotent/resumable.
#
# Usage, from the campaign worktree (e.g. ~/feat-proton-ingredients):
#   bash examples/cobordism/proton_campaign/cutover_joint_arm.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
RUN=.overnight
UNIT=tessera-campaign-562.service

echo "== 1/6: stop the running generation (kills in-flight attempts)"
systemctl --user stop "$UNIT" 2>/dev/null || echo "   ($UNIT not running)"

echo "== 2/6: archive the old arm's data (nothing deleted)"
STAMP=$(date +%Y%m%d-%H%M%S)
mkdir -p "$RUN/arm-two-step-$STAMP"
mv "$RUN"/worker_*.jsonl "$RUN"/worker_*.progress.jsonl "$RUN"/worker_*.log \
   "$RUN/arm-two-step-$STAMP/" 2>/dev/null || true
cp -p "$RUN/supervisor.log" "$RUN/arm-two-step-$STAMP/" 2>/dev/null || true
echo "   archived to $RUN/arm-two-step-$STAMP (animations/ and geometry/ stay shared)"

echo "== 3/6: update the worktree to the joint-arm code (detached on main)"
# The campaign branch was squash-merged, so it cannot fast-forward; the campaign
# runs main directly from here on. Untracked .overnight/ data is untouched.
git fetch origin main
git checkout --detach origin/main

echo "== 4/6: rebuild the engine (generation boundary — sanctioned)"
OMP_NUM_THREADS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 \
  .venv-build/bin/python -m pip install -e ".[dev]" >/dev/null
.venv-build/bin/python -c "import tessera; n = tessera.cobordism.ProtonIngredients(seed=1).joint_node(1); assert len(n.inputs) == 3 and len(n.outputs) == 0" \
  && echo "   engine OK: joint_node = 3 fixed pair inputs, nothing pinned"

echo "== 5/6: install the joint-arm campaign scripts"
cp -p examples/cobordism/proton_campaign/{worker.py,renderer.py,launch_campaign.sh,analyze_attempt.py,aggregate.py} "$RUN/"

echo "== 6/6: relaunch the supervisor (same persisted deadline)"
systemd-run --user --unit "$UNIT" \
  /usr/bin/bash -c "bash $PWD/$RUN/launch_campaign.sh >> $PWD/$RUN/supervisor.log 2>&1"
sleep 8
systemctl --user is-active "$UNIT" && pgrep -cf "worker.py --worker" | xargs echo "   workers up:"
echo "done — the joint three-pair arm is live"
