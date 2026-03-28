# caset

[![Build & Deploy Docs](https://github.com/akellehe/caset/actions/workflows/pages.yml/badge.svg)](https://github.com/akellehe/caset/actions/workflows/pages.yml)
[![Deploy static content](https://github.com/akellehe/caset/actions/workflows/static.yml/badge.svg)](https://github.com/akellehe/caset/actions/workflows/static.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

**[Documentation](https://akellehe.github.io/caset/)**

A computational test bed for non-perturbative quantum gravity. Build triangulated spacetimes, run Monte Carlo simulations, solve the discrete Einstein equations, and visualize the results -- all from Python, with C++/CUDA under the hood.

caset supports multiple approaches to quantum gravity on the same simplicial mesh infrastructure:

- **Causal Dynamical Triangulations (CDT)** -- Monte Carlo path integral over geometries with a causal foliation
- **Regge Calculus** -- discrete general relativity via deficit angles and edge-length dynamics
- **Spin Foam / GFT**, **Coset**, and **Ricci Flow** formulations (scaffolding in place)

## Installation

Requires Python 3.9+, a C++20 compiler, and CMake 3.18+.

```bash
pip install -e ".[dev]"
```

Verify:

```bash
python -c "import caset; print('caset OK')"
```

CUDA GPU acceleration is auto-detected. To force it off: `CASET_CUDA=0 pip install -e .`

## Quick start

Build a 4D Lorentzian spacetime, thermalize it with CDT, and export a rotating GIF:

```python
import caset

sig   = caset.Signature(4, caset.Lorentzian)
metric = caset.Metric(True, sig)
st    = caset.Spacetime(metric, caset.CDT, 1.0, 1.0,
                        caset.PREFERRED, caset.Toroid())
st.build(500)

cdt = caset.CDTSimulation(st, k0=2.2, k4=0.5, delta=0.6,
                           epsilon=0.02, target_n41=st.getN41())
cdt.tune()
cdt.sweep(100)

st.save("spacetime.gif", tilt=25, spin=1, precession=1)
```

## Features

### CDT Monte Carlo

Full implementation of 4D Causal Dynamical Triangulations following [Ambjorn, Jurkiewicz & Loll (2005)](https://arxiv.org/abs/hep-th/0505154). Includes all four Pachner moves (add, remove, flip, shift), Metropolis acceptance, and automatic coupling-constant tuning.

The CDT phase structure is accessible out of the box:

| Phase | Geometry | Coupling regime |
|-------|----------|-----------------|
| **A** (branched polymer) | Fractal, elongated | Large k_0 |
| **B** (crumpled) | Collapsed to 1-2 slices | Small k_0, small delta |
| **C_dS** (de Sitter) | Extended 4D, matches S^4 | Moderate k_0, nonzero delta |

```bash
python examples/phase_diagram.py        # scan the (k0, delta) plane
python examples/volume_profile_phases.py # visualize blob/crumpled/polymer shapes
```

### Regge solver

Solve the discrete Einstein equations by minimizing the Regge action gradient. Add point-mass matter via the proper-time action and watch curvature concentrate around the source.

```python
matter = caset.MatterConfiguration()
matter.setWorldlineMass(center_vertex, mass=1.0, spacetime=st)

solver = caset.ReggeSolver(st, matter)
converged, F, iters = solver.solve(tol=1e-8, max_iters=5000)
```

```bash
python examples/regge_point_mass.py --mass 2.0 --n-simplices 200
```

This produces both a simplicial-complex GIF and a per-time-slice curvature heat map.

### Paper validation

The examples reproduce key figures from the CDT literature, primarily from:

> J. Ambjorn, J. Jurkiewicz, R. Loll, **"Reconstructing the Universe"**, Phys. Rev. D 72, 064014 (2005) [[hep-th/0505154]](https://arxiv.org/abs/hep-th/0505154)

| Example | Figures reproduced | What it shows |
|---------|-------------------|---------------|
| `volume_profile_phases.py` | Figs 4-6 | Volume profiles in phases A (polymer), B (crumpled), C (de Sitter) |
| `phase_diagram.py` | Fig 3 | Phase diagram scan over the (k_0, delta) coupling plane |
| `spectral_dimension.py` | Figs 9-10 | Spectral dimension D_S: ~1.8 at short scales, ~4.0 at long scales |
| `volume_scaling.py` | Figs 7-8, 12 | Hausdorff dimension D_H = 4 from volume-volume correlator collapse |
| `effective_action.py` | Figs 11-13 | Effective action, D_2 scaling dimension, minisuperspace comparison |
| `n32_distribution.py` | Fig 2 | N_32 distribution at fixed N_41 -- strong simplex-type coupling |

Each script includes the paper's coupling constants (k_0=2.2, delta=0.6) and prints reference values for comparison. Run any of them with `--help` to see the full parameter set.

### Topologies

Three spatial topologies for the foliated slices:

- **Toroid** (`T^3 x S^1`) -- periodic boundaries, default
- **Sphere** (`S^3 x S^1`) -- natural for de Sitter cosmology
- **Cylinder** (`Sigma x [0,T]`) -- open time boundaries for transition amplitudes

### Visualization

Export spacetimes in multiple formats:

```python
st.save("spacetime.gif")      # animated rotating GIF
st.save("spacetime.png")      # static 4-panel PNG
st.save("spacetime.graphml")  # import into Gephi, yEd, etc.
st.save("spacetime.dot")      # render with Graphviz
```

The curvature-slice visualization embeds each spatial slice in 3D with a heat map of the deficit-angle curvature:

```python
from curvature_slice_gif import render_curvature_gif
render_curvature_gif(st, solver, worldline, "curvature.gif")
```

### GPU acceleration

The Regge solver optionally offloads deficit-angle and gradient computation to CUDA. This is transparent -- the same Python API works with or without a GPU. Significant speedups on triangulations with 10k+ simplices.

## Running tests

```bash
pytest tests/ -v
```

Build options via environment variables:

```bash
CASET_CUDA=0       pip install -e .     # CPU-only build
CASET_ASAN=1       pytest tests/        # AddressSanitizer + UBSan
CASET_VERBOSE=1    pytest tests/        # C++ logging
CASET_ASSERTIONS=1 pytest tests/        # extra invariant checks
```

## Building documentation

```bash
sudo apt-get install doxygen   # or: brew install doxygen
cd docs && pip install -r requirements-docs.txt && make html
open _build/html/index.html
```

## License

MIT
