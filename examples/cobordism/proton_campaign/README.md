# Proton campaign harness (#562)

The inputs-only sweep hunting a fully emergent proton: a `b₃ = 3` basin of the
two-step `ProtonIngredients` build that carries the color singlet `{1, ω, ω²}`
with nothing pinned downstream. Sixteen detached single-threaded workers run
attempts to genuine stationarity (stage-2 `relTol = 1e-9` + persistence passes)
until the calendar deadline, recording one verdict line per attempt, per-chunk
progress, animations for the interesting attempts, and — via the keep-policy
below — a drift-proof geometry dump of each interesting final complex.

These are the **canonical copies** of the campaign scripts. The live campaign
executes untracked copies in the campaign worktree's `.overnight/` directory;
this directory freezes those bytes so the exact drive that produced any verdict
stays recoverable forever. When the two diverge, the running generation defines
the physics until its workers exit; deploys land at generation boundaries only.

## Files

- `worker.py` — one attempt per base seed: build `ProtonIngredients` nodes,
  init → evolve → stage-2 to stationarity per node, persistence passes on the
  formation node, verdict + progress records, GIF + geometry-dump keep-policy.
- `renderer.py` — per-attempt animation recorder (reuses the emergent-proton
  panels; frames at real pass/chunk boundaries).
- `aggregate.py` — sweep statistics over the verdict files.
- `launch_campaign.sh` — the supervisor: waits for prior-generation workers to
  drain, launches 16 single-threaded workers against the persisted deadline
  file. Run it under `systemd-run --user` so the unit (and the workers' cgroup)
  survives the launching session.
- `analyze_attempt.py` — out-of-band rebuild + characterization of a recorded
  attempt: exact replay from its seed, verification against the recorded
  verdict (and geometry dump when present), then the ready observables.

## Exact replay (the rebuild contract)

Attempts are **bit-for-bit reproducible** from their recorded `base_seed`: a
fresh single-threaded process running the same scripts against the same
engine build reproduces the recorded snapshots exactly (verified: replayed
campaign snapshots match `F` to all digits, plus `gradN2`, `rU`, `holes`,
`b3`, `cells`, `edges`). The contract requires:

- the campaign worktree's editable engine build (its `.venv-build` python);
- single-threaded numerics: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1`;
- the generation's own `worker.py` drive (this directory, for the generation
  recorded in the provenance manifest).

The geometry dumps exist so that an interesting find survives even a broken
contract: they record the final complex itself, not a recipe for rebuilding it.

## Provenance manifest

Generation boundaries — a verdict replays only under the scripts of the
generation that recorded it:

| generation | window (MDT) | replayable | notes |
|---|---|---|---|
| pre-campaign | 2026-07-01 evening → 2026-07-02 08:57 | **no** (superseded scripts) | 16 verdicts: 15× b₃=1, 1× b₃=2; killed by the box crash |
| campaign | launched 2026-07-02 06:44 (workers from 09:21) | **yes** (these bytes) | deadline 2026-08-01 06:44 MDT |

sha256 of the frozen (= running campaign generation) scripts:

```
3641c06ca525b0cf43aa11cea42b092d709e531683c30fa64b1f63120bb522cf  worker.py   (as launched; before the geometry-dump addition)
9d75f11d1d79103e50660b1ae362db623ce8c4d7dc6ba015a1c8aaf7e4e8929c  renderer.py
198c263707aca4eeffffcbd2946378befbe3274824abc1b06a9599899994d13f  aggregate.py
73474a87d000afddefee589ed39ce0156de00b836764e72afadb3793f0deec55  launch_campaign.sh
```

The geometry-dump addition to `worker.py` executes only after an attempt
completes (it reads the final state; the drive and its RNG draws are
untouched), so it does not open a new replay generation: seeds recorded by the
as-launched bytes replay identically under the dump-enabled bytes.

## Analyzing an attempt

```bash
cd <campaign worktree>
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
.venv-build/bin/python <this dir>/analyze_attempt.py \
    --seed <base_seed> --run-dir .overnight
```

The analyzer replays the attempt (single-threaded; expect the attempt's
original wall time), verifies the final state against the recorded verdict
line and the geometry dump if one exists, runs the ready observables, and
writes `analysis_seed_<base_seed>.json` into the run directory. Use
`--skip-replay` to run the dump-only observables (no engine rebuild) on a
recorded geometry dump.
