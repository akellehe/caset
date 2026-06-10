# Quantum experiments

The project's progression toward an emergent 4D spacetime construction.

For orientation, start with the [intellectual lineage](overview/intellectual_lineage.md)
which describes the sequence of ideas — from van Raamsdonk to the
current state — and the [H_DS4 status page](overview/h_ds4_status.md)
which tracks where the central hypothesis stands across all
experiments.

The section has three parts. The [overview](overview/index.md) is the
orientation layer. [Earlier work](earlier-work/index.md) holds
everything that led here — the Schwinger-era charters and subsystem
reference, the experiments they produced, and the Charged Cartan
Monte Carlo programme. The
[State–Operation–Cobordism Correspondence](state-operation-cobordism/index.md)
is the current line.

```{toctree}
:maxdepth: 2

overview/index
earlier-work/index
state-operation-cobordism/index
```

## Reproducing

Every experiment writeup names the example script that produced its
numbers and the command-line arguments used. The scripts live in
``examples/quantum/`` and ``examples/cobordism/`` and write
reproducibility records (JSON or plain-text snapshots) into
``/tmp/interaction-history``; archive those alongside the writeup if
you want to pin a result.

## Current state at a glance

- **Current line**: the
  [State–Operation–Cobordism Correspondence](state-operation-cobordism/index.md)
  — see the [results companion](state-operation-cobordism/cobordism-results.md).
- **H_DS4 status (Charged Cartan)**: near-pass with ~1.5% offset; peak
  D_S = 4.245 at T = 20k, extrapolates to ≈ 4.07 at T → ∞.
- **Latest Charged Cartan version**: v0.2 (qudit basis + Choi-state Σ_AB).
- **Open milestone**:
  [v0.2](https://github.com/akellehe/tessera/milestone/1) (plateau
  investigation + qudit-basis follow-ups),
  [v0.3](https://github.com/akellehe/tessera/milestone/2) (gauge
  mediation).

See the [H_DS4 status page](overview/h_ds4_status.md) for the
detailed table.
