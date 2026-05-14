# Mutual-information connectivity — H_4D convergence tests

Reproducibility scripts for the experiment described in
[../../docs/source/quantum-experiments/temporally_connected_entangled_spacetime_writeup.md](../../docs/source/quantum-experiments/temporally_connected_entangled_spacetime_writeup.md).

The hypothesis under test (**H_4D**): the peak spectral dimension of
the MI-driven dual lattice approaches `D_S = 4` from below as the
chain length `N` and snapshot count `K` grow.

This directory contains the runners and plotters for the three
convergence tests that either validate or falsify H_4D.

## Layout

| File | Purpose |
|------|---------|
| `run_k_scan.sh` | Test 1 — K-only scan at fixed N=40. Cheapest. Isolates whether the N=60 K=9 overshoot was a K-effect. |
| `run_epsilon_scan.sh` | Test 2 — ε-scan at fixed (N=60, K=9). Tests the "Goldilocks ε" claim. |
| `run_n_scan.sh` | Test 3 — N-only scan at fixed K=5. Most expensive. Tests the asymptote along the N-axis. |
| `plot_k_scan.py` | Render `figures/k_scan.png` from Test 1 outputs. |
| `plot_epsilon_scan.py` | Render `figures/epsilon_scan.png` from Test 2 outputs. |
| `plot_n_scan.py` | Render `figures/n_scan.png` from Test 3 outputs. |

The experiment binary itself lives at
`../../examples/quantum/temporally_connected_entangled_spacetime.py`
and is invoked by every runner. The runners only set the scan grid
and the output directory.

## Reproducing the figures

```bash
# from the tessera repository root, on a machine that has built tessera with
# TESSERA_QUANTUM=1 set during pip install.

OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 \
MKL_NUM_THREADS=10 BLIS_NUM_THREADS=10 \
    bash examples/mutual-information-connectivity/run_k_scan.sh
python examples/mutual-information-connectivity/plot_k_scan.py

OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 \
MKL_NUM_THREADS=10 BLIS_NUM_THREADS=10 \
    bash examples/mutual-information-connectivity/run_epsilon_scan.sh
python examples/mutual-information-connectivity/plot_epsilon_scan.py

OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 \
MKL_NUM_THREADS=10 BLIS_NUM_THREADS=10 \
    bash examples/mutual-information-connectivity/run_n_scan.sh
python examples/mutual-information-connectivity/plot_n_scan.py
```

The runners write JSON records to `/tmp/temporal-entangled/{k_scan,
epsilon_scan, n_scan}/` and the plotters write PNGs to
`docs/source/quantum-experiments/figures/`.

## How the tests decide H_4D

| Test | H_4D passes if | H_4D fails if |
|------|----------------|---------------|
| K-scan | peak `D_S` plateaus at ~4 as K grows at fixed N | peak `D_S` plateaus below 4, or keeps rising past 4 |
| ε-scan | there is an intermediate ε window where peak `D_S` ≈ 4 and hop diameter grows toward `O(N)` | peak `D_S` slides monotonically from 4+ to ~1 with no plateau |
| N-scan | peak `D_S` plateaus near 4 from below as N grows at fixed K | peak `D_S` plateaus below 4, or keeps rising past 4 |

A clean H_4D validation requires **all three** tests to converge on
~4. A single plateau-below-4 refutes H_4D in favour of some lower
emergent dimension; continued rise past 4 refutes H_4D in favour of
small-world saturation.

## Costs (10-thread Schwinger TDVP)

| Test | Cells | Est. total runtime |
|------|-------|--------------------|
| K-scan (N=40, K∈{5,7,9,11,13}, m/g∈{0.125,0.25,0.5}) | 15 | 45–90 min |
| ε-scan (N=60, K=9, ε∈{1e-8,1e-6,1e-4,1e-3,1e-2}, m/g∈{0.125,0.25,0.5}) — one TDVP per m/g, multi-ε | 3 TDVPs / 15 graph builds | ~40 min |
| N-scan (N∈{50,60,80,100}, K=5, m/g=0.5) | 4 | ~2–3 hours |

## See also

- `../quantum/temporally_connected_entangled_spacetime.py` — the experiment script.
- `../quantum/plot_temporally_connected.py` — the existing N×m/g overview plotter (referenced by the dual-lattice writeup).
- `../quantum/plot_mi_vs_separation.py` — d_MI vs bond separation; confinement-scaling check.
