# MIT License
# Copyright (c) 2025 Andrew Kelleher
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Smoke tests for the cobordism subsystem scaffold (issue #62)."""

import itertools
import unittest

import tessera

# Subsystem submodules are exposed as attributes of the `tessera` package
# (via tessera/__init__.py), matching mesh/spacetime/observables/simulations.
# Dotted `import tessera.cobordism` is not supported for the C++ subsystems
# (only the pure-Python `tessera.quantum` package allows it).
cobordism = tessera.cobordism


class TestCobordismScaffold(unittest.TestCase):

    def test_submodule_imports(self):
        self.assertTrue(hasattr(tessera, "cobordism"))
        self.assertTrue(cobordism._cobordism_smoke())

    def test_combinatorial_dimension_observable_empty(self):
        # Per-complex measurements are Observables (#62). Empty complex -> -1.
        st = tessera.Spacetime()
        self.assertEqual(cobordism.CombinatorialDimension().compute(st), -1.0)

    def test_combinatorial_dimension_is_intrinsic_not_declared(self):
        # Build S^2 = ∂Δ³ (triangles) inside a 4D-declared Spacetime. The
        # combinatorial dimension is 2, independent of the Spacetime's signature.
        st = tessera.Spacetime()
        V = [st.createVertex(i, [0.0]) for i in range(4)]
        for a, b in itertools.combinations(range(4), 2):
            st.createEdge(V[a], V[b], 1.0)
        for combo in itertools.combinations(range(4), 3):
            st.createSimplex([V[i] for i in combo])
        self.assertEqual(cobordism.CombinatorialDimension().compute(st), 2.0)


if __name__ == "__main__":
    unittest.main()
