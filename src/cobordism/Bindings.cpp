// MIT License
// Copyright (c) 2025 Andrew Kelleher
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Pybind11 bindings for the cobordism subsystem. Lives outside tessera_core
// (which is pybind-free) so the static library can be reused without pulling
// in the Python dependency. This translation unit is always added to
// _tessera's sources (see CMakeLists.txt, TESSERA_PYBIND_SOURCES).

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cobordism/CombinatorialDimension.h"
#include "spacetime/Spacetime.h"  // complete type required by pybind (typeid)

namespace py = pybind11;

using namespace tessera;
using namespace tessera::cobordism;

void register_cobordism(py::module_ m) {
  // Smoke hook: lets tests assert the subsystem loaded before any
  // mathematical capability (issues #63–#70) is implemented. Single leading
  // underscore (not double) to avoid Python name-mangling inside test classes.
  m.def("_cobordism_smoke", [] { return true; },
        "Returns True; confirms the cobordism subsystem is built and importable.");

  // Per-complex scalar measurements are Observables (tessera's convention),
  // not methods on Spacetime or a bespoke wrapper. The characteristic-number
  // capabilities (Euler characteristic, signature, …) follow the same pattern;
  // multi-complex / structural operations (cobordism verification,
  // reconstruction, Pachner search) will be static-only classes taking a
  // Spacetime.
  py::class_<CombinatorialDimension, std::shared_ptr<CombinatorialDimension>>(
      m, "CombinatorialDimension",
      R"doc(Observable: combinatorial dimension of a triangulation.

The largest k with a k-simplex present (max simplex size - 1), or -1 if empty.
A purely combinatorial/topological integer (= n for a PL n-manifold), distinct
from the spectral dimension (a real-valued diffusion quantity) and from the
Spacetime's declared metric dimension.)doc")
      .def(py::init<>())
      .def("compute", &CombinatorialDimension::compute, py::arg("spacetime"),
           "Return the combinatorial dimension of the given Spacetime as a double.");
}
