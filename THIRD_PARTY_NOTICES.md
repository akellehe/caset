# Third-Party Notices

tessera is proprietary software (Copyright (c) 2026 Twin Vector Labs LLC, all
rights reserved). It builds against, links, or redistributes the third-party
components listed below. Each is governed by its own license; nothing here
grants any rights in tessera itself.

This inventory covers components that are **compiled into or dynamically linked
by the distributed `_tessera` extension**. Development-, test-, and docs-only
dependencies (pytest, scikit-build-core, cmake, ninja, sphinx, furo,
myst-parser, breathe, sphinxcontrib-bibtex, matplotlib, networkx, scipy, tqdm,
pybind11-stubgen) are not distributed with the product and are therefore out of
scope; all are under permissive licenses (BSD / MIT / Apache-2.0 / PSF), the
sole exception being tqdm (MPL-2.0, weak/file-level, not imported by the core
package).

## Linked / compiled into `_tessera`

| Component | Version | License | Notes |
|---|---|---|---|
| **ITensor** (v3) | submodule pin | Apache-2.0 | ITensor developers / Simons Foundation. Permissive. Real-time TDVP is provided by tessera's own `TDVPIntegrator` on top of ITensor core — see note below. |
| **Eigen** | 3.4 | MPL-2.0 | We use only MPL-2.0 modules (Core, Dense, SparseCore, KroneckerProduct). `EIGEN_MPL2_ONLY` is defined to make any non-MPL2 module a compile error. MPL-2.0 is file-level copyleft and does not restrict our proprietary use. |
| **BLAS / LAPACK** | system | BSD-3-Clause | Resolves to the platform backend (Debian netlib reference BLAS/LAPACK = BSD-3; may be OpenBLAS = BSD-3, Accelerate = Apple system framework, or MKL = proprietary-redistributable depending on the host). None are copyleft. |
| **zlib** (`libz`) | system | Zlib | Permissive. |
| **OpenMP runtime** (`libgomp`) | system | GPL-3.0 **+ GCC Runtime Library Exception** | The Runtime Library Exception explicitly permits linking into proprietary software. (If built with Clang, `libomp` is Apache-2.0-WITH-LLVM-exception.) Dynamically linked system library. |
| **NVIDIA CUDA runtime** (`libcudart`, `libcublas`) | CUDA 12 | NVIDIA CUDA EULA | Proprietary, **not** copyleft. Redistribution is governed by the CUDA EULA — see the redistribution note below. Built only when `TESSERA_CUDA=ON`. |

## Test-only (not distributed)

| Component | License | Notes |
|---|---|---|
| **Catch2** (`third_party/itensor/unittest/catch.hpp`) | Boost Software License 1.0 | Compiled only into ITensor's unit tests, never into `_tessera`. |
| **pybind11** | BSD-3-Clause | Build-time header-only; the binding glue it generates is BSD-3, compatible with proprietary distribution. |

## Runtime Python dependency

| Component | License | Notes |
|---|---|---|
| **NumPy** | BSD-3-Clause | The only mandatory runtime Python dependency. |

---

## Real-time TDVP is tessera-owned

tessera previously linked the ITensor/TDVP add-on
(<https://github.com/ITensor/TDVP>), which shipped **no license** upstream (no
LICENSE file, no per-file headers, none reported by the GitHub licensing API) —
effectively "all rights reserved". Because the Schwinger-quench pipeline
compiled it into the distributed binary, that was a material obligation.

It has been removed. Real-time evolution is now provided by tessera's own
`TDVPIntegrator` (`include/quantum/TDVPIntegrator.hpp`,
`src/quantum/TDVPIntegrator.cpp`) — an independent two-site TDVP implementation
built solely on Apache-2.0 ITensor *core* primitives. No code of unstated
license is linked.

## GPL-3.0 code present but NOT linked

ITensor vendors GPL-3.0-or-later HDF5-serialization code
(`third_party/itensor/itensor/util/h5/`, Copyright (C) 2011-2014 O. Parcollet).
It is **never compiled or linked** into tessera:

- the h5 `.cc` files are excluded from the ITensor source list in
  `cmake/BuildITensor.cmake` (HDF5 serialization is disabled),
- the headers are gated behind `#ifdef ITENSOR_USE_HDF5`, which tessera never
  defines, and
- `cmake/BuildITensor.cmake` contains a hard guard that fails configuration if
  `ITENSOR_USE_HDF5` is ever set.

It is listed here only for transparency. Do not enable `ITENSOR_USE_HDF5` while
tessera remains proprietary.

## NVIDIA CUDA redistribution

The CUDA runtime libraries (`libcudart`, `libcublas`) are governed by the
NVIDIA CUDA Toolkit EULA. They are not copyleft, but redistribution carries
conditions: ship them as the NVIDIA-provided redistributable libraries
(per the EULA's redistributable-software attachment) rather than statically
embedding CUDA components, and include NVIDIA's required attribution. Builds
with `TESSERA_CUDA=OFF` link none of this and are unaffected.
