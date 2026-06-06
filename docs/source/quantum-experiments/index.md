# Quantum experiments

The project's progression toward an emergent 4D spacetime construction.

For orientation, start with the [intellectual lineage](overview/intellectual_lineage.md)
which describes the sequence of ideas — from van Raamsdonk to the
current state — and the [H_DS4 status page](overview/h_ds4_status.md)
which tracks where the central hypothesis stands across all
experiments.

```{toctree}
:maxdepth: 2

overview/index
earlier-work/index
charged-cartan/index
cobordism
cobordism-plan
```

## Reproducing

Every experiment writeup names the example script that produced its
numbers and the command-line arguments used. The scripts live in
``examples/quantum/`` and write reproducibility records (JSON or
plain-text snapshots) into ``/tmp/interaction-history``; archive
those alongside the writeup if you want to pin a result.

## Current state at a glance

- **H_DS4 status**: near-pass with ~1.5% offset; peak D_S = 4.245 at
  T = 20k, extrapolates to ≈ 4.07 at T → ∞.
- **Latest version**: v0.2 (qudit basis + Choi-state Σ_AB).
- **Open milestone**:
  [v0.2](https://github.com/akellehe/tessera/milestone/1) (plateau
  investigation + qudit-basis follow-ups),
  [v0.3](https://github.com/akellehe/tessera/milestone/2) (gauge
  mediation).

See the [H_DS4 status page](overview/h_ds4_status.md) for the
detailed table.
