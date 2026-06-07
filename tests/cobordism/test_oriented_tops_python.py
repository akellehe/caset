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

"""Oriented top-simplices and the fundamental class [W] = ε_t (#107).

The Stage-2 Dijkgraaf–Witten weight ω(·)^{ε_t} needs each top simplex's
orientation sign relative to a coherent orientation of the manifold W — the
fundamental class [W] ∈ H_d. ``ChainComplex.fundamentalClass()`` returns those
signs ε_t = ±1 (the generator of ker ∂_d), and
``ChainComplex.orientedTopSimplices()`` returns the top simplices in the column
order the signs refer to.

We check on closed oriented manifolds (T³, T², S²) that the signed top chain is
a cycle — applying ∂_d gives 0, i.e. every codimension-one face's two incident
top simplices cancel — that the signs are ±1 and deterministically normalized,
and that orientedTopSimplices() lines up with the boundary-matrix columns and
the manifold's actual top simplices.
"""

import unittest

import tessera

cobordism = tessera.cobordism


def _circle():
    return tessera.SimplexBoundarySphere(1)  # S^1


def _two_sphere():
    return tessera.SimplexBoundarySphere(2)  # S^2


def _torus():
    return tessera.SimplicialProduct(_circle(), _circle())  # T^2 = S^1 x S^1


def _three_torus():
    return tessera.SimplicialProduct(_torus(), _circle())  # T^3 = T^2 x S^1


def _build(topology):
    signature = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, signature)
    spacetime = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                                  tessera.PREFERRED, topology)
    spacetime.build()
    return spacetime


def _top_tuples(spacetime):
    """The top simplices of a Spacetime as sorted vertex-id tuples."""
    tuples = [tuple(sorted(v.getId() for v in s.getVertices()))
              for s in spacetime.getSimplices()]
    top_size = max(len(t) for t in tuples)
    return [t for t in tuples if len(t) == top_size]


def _boundary_of_signed_top_chain(chain, d, epsilon):
    """∂_d (Σ_t ε_t·t) as a list of coefficients over the (d-1)-simplices.

    ∂_d is flat row-major with cols = |C_d| (= len(epsilon)); the chain
    coefficient on row r is Σ_c ∂_d[r, c]·ε_c.
    """
    flat = chain.boundaryMatrix(d)
    cols = len(epsilon)
    rows = len(flat) // cols if cols else 0
    return [sum(flat[r * cols + c] * epsilon[c] for c in range(cols))
            for r in range(rows)]


class TestFundamentalClass(unittest.TestCase):

    def _check_closed_oriented(self, topology, expected_dimension):
        spacetime = _build(topology)
        chain = cobordism.ChainComplex.fromSpacetime(spacetime)
        d = chain.dimension()
        self.assertEqual(d, expected_dimension)

        epsilon = list(chain.fundamentalClass())
        tops = [tuple(t) for t in chain.orientedTopSimplices()]

        # orientedTopSimplices length == number of top simplices == |ε|.
        num_top = chain.numSimplices(d)
        self.assertEqual(len(tops), num_top)
        self.assertEqual(len(epsilon), num_top)

        # The returned top simplices are exactly the manifold's top simplices,
        # as sorted vertex-id tuples (the canonical ∂_d column order).
        self.assertEqual(set(tops), set(_top_tuples(spacetime)))
        self.assertEqual(len(set(tops)), len(tops))  # no duplicates
        for t in tops:
            self.assertEqual(list(t), sorted(t))

        # ε_t ∈ {±1}, deterministically sign-normalized (first nonzero is +1).
        self.assertTrue(all(e in (-1, 1) for e in epsilon))
        self.assertEqual(next(e for e in epsilon if e != 0), 1)

        # The signed top chain is a fundamental cycle: ∂_d (Σ ε_t·t) = 0, so
        # every codimension-one face's two incident top simplices cancel.
        self.assertTrue(
            all(coefficient == 0
                for coefficient in _boundary_of_signed_top_chain(chain, d, epsilon)))

    def test_three_torus_is_a_fundamental_cycle(self):
        # T^3 = S^1 x S^1 x S^1: a closed oriented 3-manifold (b_3 = 1).
        self._check_closed_oriented(_three_torus(), 3)

    def test_two_sphere_sanity_check(self):
        # S^2: closed oriented surface (b_2 = 1).
        self._check_closed_oriented(_two_sphere(), 2)

    def test_two_torus_is_a_fundamental_cycle(self):
        # T^2 = S^1 x S^1: another closed oriented surface (b_2 = 1).
        self._check_closed_oriented(_torus(), 2)

    def test_consistent_with_homology_top_betti(self):
        # The fundamental class exists exactly when b_d = 1; cross-check that the
        # closed oriented fixtures have a one-dimensional top homology.
        for topology, dimension in ((_three_torus(), 3),
                                    (_two_sphere(), 2),
                                    (_torus(), 2)):
            with self.subTest(dimension=dimension):
                chain = cobordism.ChainComplex.fromSpacetime(_build(topology))
                self.assertEqual(chain.bettiNumbers()[dimension], 1)


if __name__ == "__main__":
    unittest.main()
