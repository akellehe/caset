"""tessera.quantum.cuda — torch / libtorch backend mirror of the Schwinger pipeline.

Compute lives in the C++ extension built when ``TESSERA_QUANTUM_CUDA=1``;
this module is a thin re-export shim that surfaces those bindings under
their natural import path.

Tensors live on a single ``torch.device`` (default ``cuda:0`` when
CUDA is available, ``cpu`` otherwise) at ``torch.complex128``. Class
names mirror the C++ ITensor pipeline in :mod:`tessera.quantum` so
cross-validation tests can be written without translation.

The user-facing classes (built up over successive work units) are:

* :class:`MPS`                — matrix product state on torch tensors
* :class:`MutualInformation`  — entropy + site-pair MI helpers

Site indices are 0-based throughout this module (idiomatic Python),
contrast with the 1-based indexing of the ITensor binding.

Availability
------------
Requires both ``TESSERA_QUANTUM=1`` and ``TESSERA_QUANTUM_CUDA=1`` at
build time::

    TESSERA_QUANTUM=1 TESSERA_QUANTUM_CUDA=1 pip install -e .

A working pytorch install (the CMake config under
``site-packages/torch/share/cmake/Torch``) is required during the C++
build. CUDA is detected from torch at runtime — falls back cleanly to
the CPU torch path on machines without an NVIDIA GPU.

Quickstart
----------

>>> import torch
>>> from tessera.quantum.cuda import MPS, MutualInformation
>>> psi = MPS.bell_chain(1)
>>> rho = psi.one_site_reduced_density(0)
>>> MutualInformation.von_neumann_entropy(rho)
0.6931471805599453
"""
try:
    # The libtorch backend lives in its own pybind11 extension
    # (_tessera_cuda.so) sitting next to this file, kept separate from
    # the main tessera._tessera so libtorch's libstdc++ doesn't cross
    # into the main build.
    from . import _tessera_cuda as _cuda

    MPS                = _cuda.MPS
    MutualInformation  = _cuda.MutualInformation
except (ImportError, AttributeError) as exc:
    raise ImportError(
        "tessera.quantum.cuda is unavailable: this tessera build does "
        "not include the libtorch mirror. Rebuild with "
        "TESSERA_QUANTUM=1 and TESSERA_QUANTUM_CUDA=1 (e.g. "
        "`TESSERA_QUANTUM=1 TESSERA_QUANTUM_CUDA=1 pip install -e .`) "
        "to enable it."
    ) from exc

__all__ = [
    "MPS",
    "MutualInformation",
]
