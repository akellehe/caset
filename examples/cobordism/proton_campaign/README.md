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
  Streams and failure semantics (#591): the progress stream is written gzipped
  (`worker_N.progress.jsonl.gz` — inspect with `zcat`; 50 MB uncompressed cap
  per process); the verdict stream is capped at 50 MB (a `verdicts_capped`
  marker line, then the worker exits — a worker that cannot record must not
  run); a startup self-check constructs the drive's node once and exits
  non-zero on a broken environment (stale build, missing binding) instead of
  recording one instant failure per seed; and consecutive attempts failing
  with an identical error collapse into one summary record (`error_repeats`,
  `first_seed`/`last_seed` spanning the collapsed repeats), with a short sleep
  after every failed attempt. All harness-only — attempt physics untouched.
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

## The rebuild contract: dumps, not replays

The engine build is **not process-deterministic**: identical fresh single-
threaded processes running the identical node construction and drive on the
same seed can produce different complexes (measured: same seed, same calls,
`F=272.5`/78 cells vs `F=81.0`/43 cells). Occasional bit-exact replays happen
on stable paths, but they are luck, not a contract. A `base_seed` therefore
**labels** an attempt; it cannot reproduce one.

The **geometry dump is the attempt's only faithful record**: `worker.py`
writes one per attempt (`geometry/seed_<base>_geometry.json` — top cells in
intrinsic vertex order, every edge's complex ℓ², per-vertex times), canonically
ordered so the same state always serializes to the same bytes, written only
after the attempt completes. `Spacetime.fromCells` rebuilds the exact final
state from it in seconds; `analyze_attempt.py --replay` remains available as a
fresh *sample* of a seed's attempt distribution (its verdict comparison is a
divergence measurement, not a check).

## Provenance manifest

Generation boundaries — which scripts *drove* which verdicts (the drive
defines the physics of the recorded statistics; it cannot reproduce
individual attempts, see above):

| generation | window (MDT) | notes |
|---|---|---|
| pre-campaign | 2026-07-01 evening → 2026-07-02 08:57 | 16 verdicts: 15× b₃=1, 1× b₃=2; superseded scripts; killed by the box crash |
| campaign | launched 2026-07-02 06:44 (workers from 09:21) | two-step drive, dump-less at launch; deadline 2026-08-01 06:44 MDT |

sha256 of the campaign generation's scripts as launched:

```
3641c06ca525b0cf43aa11cea42b092d709e531683c30fa64b1f63120bb522cf  worker.py   (as launched; before dumps-per-attempt)
9d75f11d1d79103e50660b1ae362db623ce8c4d7dc6ba015a1c8aaf7e4e8929c  renderer.py
198c263707aca4eeffffcbd2946378befbe3274824abc1b06a9599899994d13f  aggregate.py
73474a87d000afddefee589ed39ce0156de00b836764e72afadb3793f0deec55  launch_campaign.sh
```

The dumps-per-attempt addition to `worker.py` executes only after an attempt
completes (it reads the final state; the drive and its RNG draws are
untouched), so deploying it mid-generation changes no attempt physics — it
only starts recording what was previously lost.

## Analyzing an attempt

```bash
cd <campaign worktree>
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
.venv-build/bin/python <this dir>/analyze_attempt.py \
    --seed <base_seed> --run-dir .overnight
```

The analyzer rebuilds the attempt's final state from its geometry dump
(seconds), verifies it against the dump's recorded metadata, runs the ready
observables (color, geometry/curvature in both deficit channels, spectral
dimension, deficit-angle Wilson loops), and writes
`analysis_seed_<base_seed>.json` into the run directory. `--replay` re-runs
the attempt with the frozen drive instead — a fresh sample of the seed's
attempt distribution whose verdict comparison measures divergence.
