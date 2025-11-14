# caset

Caset is a package for experimenting with CAusal SETs and simplicial manifolds in C++ with libtorch for GPU and hardware 
acceleration. It provides a highly parallelizable interface to execute methods against a causally oriented simplicial 
Lorentzian(or Euclidean) manifold.

The intention is to calculate things like holonomies and observables as co-chains over chain complexes within several of
the most mature theories of discretized causal spacetime.

Currently there is a C++ interface with bindings to python (which we use for unit testing). The C++ documentation 
extends to the python documentation, even though it doesn't appear there natively. We provide a very tight set of 
bindings to move back and forth.


```{toctree}
:maxdepth: 2

getting_started
theory
cpp_api
```
