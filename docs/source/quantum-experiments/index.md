# Quantum experiments

Experimental writeups of runs against the hypotheses laid out in
[../quantum-methodology.md](../quantum-methodology.md) and
[../holography-causal-ordering-emergent-dimension.md](../holography-causal-ordering-emergent-dimension.md).
Each writeup pairs numerical results with the falsification check the
corresponding spec specifies; cross-link the spec and the writeup to
trace claim → number → command-to-reproduce.

```{toctree}
:maxdepth: 1

lightcone_vs_majorization_writeup
emergent_spectral_dimension_writeup
temporally_connected_entangled_spacetime_writeup
```

## Reproducing

Every writeup names the example script that produced its numbers and
the command-line arguments used. The scripts live in
``examples/quantum/`` and write reproducibility records (JSON or
plain-text snapshots) into ``/tmp``; archive those alongside the
writeup if you want to pin the result.
