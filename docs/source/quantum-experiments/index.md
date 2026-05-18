# Quantum experiments

Experimental writeups of runs against the hypotheses laid out in
[../quantum-methodology.md](../quantum-methodology.md) and
[../holography-causal-ordering-emergent-dimension.md](../holography-causal-ordering-emergent-dimension.md).
Each writeup pairs numerical results with the falsification check the
corresponding spec specifies; cross-link the spec and the writeup to
trace claim → number → command-to-reproduce.

```{toctree}
:maxdepth: 1
:caption: Project-level overview

intellectual_lineage
```

```{toctree}
:maxdepth: 1
:caption: Earlier experiments

lightcone_vs_majorization_writeup
emergent_spectral_dimension_writeup
temporally_connected_entangled_spacetime_writeup
interaction_branching_simplex_writeup
interaction_history_monte_carlo_writeup
from_schwinger_to_lattice
```

```{toctree}
:maxdepth: 1
:caption: Charged Cartan Monte Carlo

charged_cartan_monte_carlo_v0.1
charged_cartan_v01_BplusIII_writeup
charged_cartan_monte_carlo_v0.2
charged_cartan_v02_beta_scan_writeup
v02_finite_size_investigation
charged_cartan_monte_carlo_v0.3
```

## Reproducing

Every writeup names the example script that produced its numbers and
the command-line arguments used. The scripts live in
``examples/quantum/`` and write reproducibility records (JSON or
plain-text snapshots) into ``/tmp``; archive those alongside the
writeup if you want to pin the result.
