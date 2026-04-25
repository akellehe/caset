"""Phase 6 Python tests — Spacetime → causet-chain extraction (the
``caset.quantum.extract_causet_chain`` adapter) plus the underlying
``caset.Poset.from_spacetime`` static method exposed in the same
module's :class:`Poset`.

Skips cleanly when caset was built without CASET_QUANTUM=1.
"""

from __future__ import annotations

import unittest

import caset

try:
    from caset.quantum import CausetChain, Poset, extract_causet_chain
    HAVE_QUANTUM = True
except ImportError:
    HAVE_QUANTUM = False


def _build_default_cdt(num_simplices: int = 20) -> "caset.Spacetime":
    """Tiny default-CDT Spacetime for testing — toroidal, alpha=1, a=1."""
    metric = caset.Metric(True, caset.Signature(4, caset.Lorentzian))
    st = caset.Spacetime(
        metric, caset.CDT, 1.0, 1.0, caset.PREFERRED, caset.Toroid()
    )
    st.build(num_simplices)
    return st


@unittest.skipUnless(HAVE_QUANTUM, "caset built without CASET_QUANTUM=1")
class TestExtractCausetChain(unittest.TestCase):
    """The extractor produces a self-consistent CausetChain on a built CDT.

    These don't hand-pick exact hop counts (those depend on the toroidal
    Topology's gluing pattern) — instead they assert structural
    invariants that have to hold for any valid extraction:

    * antichain sizes sum to n_sites,
    * vertex_ids has length n_sites and covers exactly the union of
      antichains in time order,
    * every hopping pair (i, j) has i < j and connects sites in
      adjacent antichain layers,
    * partial_order is a valid Poset whose covers are a subset of the
      hopping pairs (covers can be reduced; hops are not).
    """

    def test_default_cdt_self_consistent(self) -> None:
        st = _build_default_cdt(num_simplices=20)
        chain = extract_causet_chain(st)

        # n_sites == sum of antichain sizes == len(vertex_ids).
        self.assertEqual(
            sum(len(a) for a in chain.antichains), chain.n_sites
        )
        self.assertEqual(len(chain.vertex_ids), chain.n_sites)

        # times sorted ascending, no duplicates.
        self.assertEqual(list(chain.times), sorted(set(chain.times)))

        # Flat layout: vertex_ids is the concatenation of antichains
        # in time order (per docstring of CausetChain).
        flat = [vid for ac in chain.antichains for vid in ac]
        self.assertEqual(list(chain.vertex_ids), flat)

        # hopping_pairs invariants.
        layer_of_site: dict[int, int] = {}
        flat_idx = 0
        for layer_idx, ac in enumerate(chain.antichains):
            for _ in ac:
                layer_of_site[flat_idx] = layer_idx
                flat_idx += 1
        for i, j in chain.hopping_pairs:
            self.assertLess(i, j, "hopping pair must be canonicalised i<j")
            self.assertIn(i, layer_of_site)
            self.assertIn(j, layer_of_site)
            self.assertEqual(
                abs(layer_of_site[j] - layer_of_site[i]), 1,
                "hopping pair endpoints must be in adjacent layers"
            )

        # partial_order Poset has n_sites nodes and its covers are a
        # subset of the hopping pairs (covers may be reduced when there
        # are alternative paths; hops include all adjacent-slice
        # timelike edges that survived the cover reduction).
        self.assertEqual(chain.partial_order.n_nodes, chain.n_sites)
        cover_set = set(chain.partial_order.covers)
        hop_set = set(chain.hopping_pairs)
        self.assertTrue(
            cover_set <= hop_set,
            f"covers {cover_set} not subset of hops {hop_set}"
        )


@unittest.skipUnless(HAVE_QUANTUM, "caset built without CASET_QUANTUM=1")
class TestPosetFromSpacetimePython(unittest.TestCase):
    """Direct Python access to caset.Poset.from_spacetime.

    The Phase 6 integration is most useful via extract_causet_chain
    (which packages partial_order alongside lattice metadata) but the
    raw ``Poset.from_spacetime`` is also Python-callable and forms the
    basis of the chain extractor.
    """

    def test_from_spacetime_returns_valid_poset(self) -> None:
        st = _build_default_cdt(num_simplices=20)
        poset = Poset.from_spacetime(st)
        self.assertEqual(poset.n_nodes, st.getVertexList().toVector().__len__())
        # Hasse covers can't have self-loops or duplicates.
        covers = poset.covers
        self.assertEqual(len(set(covers)), len(covers))
        for a, b in covers:
            self.assertNotEqual(a, b)
            self.assertGreaterEqual(a, 0)
            self.assertLess(a, poset.n_nodes)
            self.assertGreaterEqual(b, 0)
            self.assertLess(b, poset.n_nodes)

    def test_to_dot_renders(self) -> None:
        st = _build_default_cdt(num_simplices=20)
        poset = Poset.from_spacetime(st)
        dot = poset.to_dot()
        self.assertIn("digraph poset", dot)
        # Every cover should appear as an edge in the DOT output.
        for a, b in poset.covers:
            self.assertIn(f"{a} -> {b}", dot)


if __name__ == "__main__":
    unittest.main()
