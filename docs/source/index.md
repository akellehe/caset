# tessera

Tessera is a personal sandbox for poking at simplicial manifolds and partially-ordered ("causal") sets in C++. It
provides a highly parallelizable interface for working with causally oriented simplicial Lorentzian (or Euclidean)
meshes. Optional CUDA GPU acceleration is available for the Regge solver.

The fun is computing things like holonomies and co-chain observables on those meshes, and seeing how a few different
discrete-geometry constructions compare side by side.

Currently there is a C++ interface with bindings to python (which we use for unit testing). The C++ documentation 
extends to the python documentation, even though it doesn't appear there natively. We provide a very tight set of 
bindings to move back and forth.


```{toctree}
:maxdepth: 2

getting_started
theory
examples
benchmarks
quantum
quantum-methodology
holography-causal-ordering-emergent-dimension
quantum-experiments/index
cpp_api
```
