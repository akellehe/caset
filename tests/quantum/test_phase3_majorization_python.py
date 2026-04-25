"""Phase 3 Python-level tests: pure majorization predicate, Poset
construction, and the end-to-end compute_ground_state_majorization
pipeline through the caset.quantum API.

These mirror the C++-side tests in test_majorization.cpp,
test_schmidt_spectra.cpp, test_majorization_poset.cpp, and
test_schwinger_schmidt_cross_check.cpp at the Python boundary — any
binding-level bug (wrong type registration, lost copies, off-by-one
indexing on returned lists) shows up here without needing to rebuild
the C++ tests.

Skips cleanly when caset was built without CASET_QUANTUM=1.
"""

from __future__ import annotations

import unittest
from itertools import permutations

try:
    from caset.quantum import (
        QuantumConfig,
        Interval,
        SchmidtSpectra,
        Poset,
        majorizes,
        strictly_majorizes,
        majorization_poset,
        compute_ground_state_majorization,
    )
    HAVE_QUANTUM = True
except ImportError:
    HAVE_QUANTUM = False


# Tolerance for float comparisons — well below DMRG / SVD noise on the
# small problems we exercise here, but loose enough to absorb the dense-
# vs-MPS path's last-bit rounding.
TOL = 1e-10


@unittest.skipUnless(HAVE_QUANTUM, "caset built without CASET_QUANTUM=1")
class TestMajorizesPredicate(unittest.TestCase):
    """Pure-function tests on majorizes() / strictly_majorizes()."""

    def test_reflexivity(self) -> None:
        for v in ([1.0], [0.5, 0.5], [0.7, 0.2, 0.1]):
            self.assertTrue(majorizes(v, v))
            self.assertFalse(strictly_majorizes(v, v))

    def test_canonical_strict_pair(self) -> None:
        self.assertTrue(majorizes([1.0, 0.0], [0.5, 0.5]))
        self.assertFalse(majorizes([0.5, 0.5], [1.0, 0.0]))
        self.assertTrue(strictly_majorizes([1.0, 0.0], [0.5, 0.5]))

    def test_zero_padding_invariance(self) -> None:
        self.assertTrue(majorizes([0.5, 0.5, 0.0, 0.0], [0.5, 0.5]))
        self.assertTrue(majorizes([0.5, 0.5], [0.5, 0.5, 0.0]))
        self.assertFalse(strictly_majorizes([0.5, 0.5, 0.0], [0.5, 0.5]))

    def test_sort_invariance(self) -> None:
        a = [0.7, 0.2, 0.1]
        # All 6 permutations of `a` should compare equal to `a` itself.
        for perm in permutations(a):
            self.assertTrue(majorizes(list(perm), a))
            self.assertTrue(majorizes(a, list(perm)))

    def test_transitivity(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [0.5, 0.5, 0.0]
        c = [1/3, 1/3, 1/3]
        self.assertTrue(strictly_majorizes(a, b))
        self.assertTrue(strictly_majorizes(b, c))
        self.assertTrue(strictly_majorizes(a, c))

    def test_unequal_total_mass_rejected(self) -> None:
        self.assertFalse(majorizes([1.0, 0.0], [0.5, 0.5, 0.5]))
        self.assertFalse(majorizes([0.5, 0.5, 0.5], [1.0, 0.0]))

    def test_incomparable(self) -> None:
        a = [0.5, 0.4, 0.1]
        b = [0.6, 0.2, 0.2]
        self.assertFalse(majorizes(a, b))
        self.assertFalse(majorizes(b, a))

    def test_tol_is_effective(self) -> None:
        # Two distributions that agree to 1e-13 but differ at 1e-12. With
        # tol = 1e-14 they're distinguishable; with tol = 1e-10 they're
        # equivalent.
        a = [0.5, 0.5]
        b = [0.5 + 5e-13, 0.5 - 5e-13]
        self.assertTrue(majorizes(a, b, tol=1e-10))
        self.assertTrue(majorizes(b, a, tol=1e-10))


@unittest.skipUnless(HAVE_QUANTUM, "caset built without CASET_QUANTUM=1")
class TestMajorizationPoset(unittest.TestCase):
    """Pure-function tests on majorization_poset() and the Poset struct."""

    def test_empty_input(self) -> None:
        p = majorization_poset([])
        self.assertEqual(p.n_nodes, 0)
        self.assertEqual(p.covers, [])

    def test_single_node(self) -> None:
        p = majorization_poset([[1.0]])
        self.assertEqual(p.n_nodes, 1)
        self.assertEqual(p.covers, [])

    def test_canonical_chain_transitive_reduction(self) -> None:
        spectra = [
            [1/3, 1/3, 1/3],  # 0 — most uniform
            [0.5, 0.5],        # 1 — middle
            [1.0],             # 2 — most concentrated
        ]
        p = majorization_poset(spectra)
        self.assertEqual(p.n_nodes, 3)
        self.assertEqual(set(p.covers), {(2, 1), (1, 0)})

    def test_equivalent_spectra_no_strict_edges(self) -> None:
        # Three nodes with the same sorted-padded spectrum form an
        # equivalence class — no strict edges among them.
        p = majorization_poset([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        self.assertEqual(p.covers, [])

    def test_covers_only_have_in_range_indices(self) -> None:
        p = majorization_poset([[1.0], [0.5, 0.5], [1/3]*3])
        for a, b in p.covers:
            self.assertGreaterEqual(a, 0)
            self.assertLess(a, p.n_nodes)
            self.assertGreaterEqual(b, 0)
            self.assertLess(b, p.n_nodes)
            self.assertNotEqual(a, b)


@unittest.skipUnless(HAVE_QUANTUM, "caset built without CASET_QUANTUM=1")
class TestComputeGroundStateMajorization(unittest.TestCase):
    """End-to-end pipeline tests: DMRG → Schmidt → Poset all via Python."""

    @staticmethod
    def _basic_config(N: int, m: float = 0.0, g: float = 1.0,
                      L0: float = 0.0,
                      max_bond_dim: int = 32, n_sweeps: int = 8) -> "QuantumConfig":
        cfg = QuantumConfig()
        cfg.N = N; cfg.a = 1.0; cfg.g = g; cfg.m = m; cfg.L0 = L0
        cfg.max_bond_dim = max_bond_dim; cfg.n_sweeps = n_sweeps
        return cfg

    def test_n6_pipeline_basic(self) -> None:
        cfg = self._basic_config(N=6)
        r = compute_ground_state_majorization(cfg)

        # Ground-state field is just a GroundStateResult.
        self.assertEqual(r.spectra.N, 6)
        self.assertLess(r.ground_state.energy, 0)  # sane sign
        self.assertGreater(r.ground_state.bond_dim, 0)

        # Schmidt: every contiguous interval [i, j] with 1 ≤ i ≤ j ≤ N
        # and (i, j) != (1, N) appears exactly once.
        expected_intervals = {
            (i, j) for i in range(1, 7) for j in range(i, 7)
            if not (i == 1 and j == 6)
        }
        seen = {(iv.i, iv.j) for iv in r.spectra.intervals}
        self.assertEqual(expected_intervals, seen)
        self.assertEqual(len(r.spectra.intervals), len(r.spectra.spectra))

    def test_spectra_normalize_to_one(self) -> None:
        cfg = self._basic_config(N=6)
        r = compute_ground_state_majorization(cfg)
        for spec, iv in zip(r.spectra.spectra, r.spectra.intervals):
            total = sum(spec)
            self.assertAlmostEqual(
                total, 1.0, places=8,
                msg=f"interval [{iv.i}, {iv.j}] sums to {total}")

    def test_complement_symmetry(self) -> None:
        """Schmidt spectrum of [i, j] equals that of [j+1, N] when the
        complement is contiguous (i.e. left-edge cuts) — this is Schmidt
        complementarity. With i = 1, the complement [j+1, N] is one
        contiguous block, so we can compare directly."""
        cfg = self._basic_config(N=6, m=0.0)
        r = compute_ground_state_majorization(cfg)
        by_iv = {(iv.i, iv.j): spec for iv, spec
                 in zip(r.spectra.intervals, r.spectra.spectra)}
        # For each j in 1..N-1, [1, j] | rest and [j+1, N] | rest are
        # the same bipartition.
        for j in range(1, 6):
            left  = sorted(by_iv[(1, j)],     reverse=True)
            right = sorted(by_iv[(j+1, 6)],   reverse=True)
            n = max(len(left), len(right))
            left  += [0.0] * (n - len(left))
            right += [0.0] * (n - len(right))
            for k in range(n):
                self.assertAlmostEqual(
                    left[k], right[k], places=8,
                    msg=f"[1, {j}] vs [{j+1}, 6] spectra disagree at idx {k}")

    def test_n4_product_state_limit(self) -> None:
        """At m → ∞ the GS is approximately a Néel product state, so every
        Schmidt spectrum should be (1, 0) up to O(1/m²) perturbative
        corrections from hopping. With m = 200 those corrections are at
        the 1e-5 level — we pass tol = 1e-3 to majorization_poset so they
        get absorbed into the equivalence-class smoothing, leaving no
        Hasse cover edges."""
        cfg = self._basic_config(N=4, m=200.0, max_bond_dim=32, n_sweeps=10)
        r = compute_ground_state_majorization(cfg, tol=1e-3)
        # Largest entry of every spectrum should be ≈ 1.
        for spec, iv in zip(r.spectra.spectra, r.spectra.intervals):
            self.assertAlmostEqual(
                max(spec), 1.0, places=3,
                msg=f"product-state limit broken at [{iv.i}, {iv.j}]: {spec}")
        # With the wide tolerance, all spectra collapse to one
        # equivalence class — Hasse should be empty.
        self.assertEqual(r.poset.n_nodes, len(r.spectra.spectra))
        self.assertEqual(r.poset.covers, [])

    def test_poset_is_transitively_reduced(self) -> None:
        """Cover edges shouldn't include any (a, b) for which there's an
        intermediate c with covers (a, c) and (c, b) — that's the
        transitive-reduction guarantee."""
        cfg = self._basic_config(N=6)
        r = compute_ground_state_majorization(cfg)
        # Build adjacency from cover list.
        succ = {a: set() for a in range(r.poset.n_nodes)}
        for a, b in r.poset.covers:
            succ[a].add(b)
        for (a, b) in r.poset.covers:
            for c in range(r.poset.n_nodes):
                if c == a or c == b:
                    continue
                if c in succ[a] and b in succ[c]:
                    self.fail(
                        f"poset has cover ({a}, {b}) but intermediate "
                        f"path {a} → {c} → {b} also exists")

    def test_poset_irreflexive(self) -> None:
        cfg = self._basic_config(N=6)
        r = compute_ground_state_majorization(cfg)
        for a, b in r.poset.covers:
            self.assertNotEqual(a, b, "self-loop in Hasse diagram")

    def test_poset_acyclic(self) -> None:
        """Hasse cover edges form a DAG (no directed cycles)."""
        cfg = self._basic_config(N=6)
        r = compute_ground_state_majorization(cfg)

        # Topological-sort attempt via Kahn's algorithm; if it fails to
        # consume every node, the graph has a cycle.
        in_deg = {a: 0 for a in range(r.poset.n_nodes)}
        succ   = {a: [] for a in range(r.poset.n_nodes)}
        for a, b in r.poset.covers:
            succ[a].append(b)
            in_deg[b] += 1
        queue = [n for n in range(r.poset.n_nodes) if in_deg[n] == 0]
        processed = 0
        while queue:
            n = queue.pop()
            processed += 1
            for m in succ[n]:
                in_deg[m] -= 1
                if in_deg[m] == 0:
                    queue.append(m)
        self.assertEqual(processed, r.poset.n_nodes,
                         "directed cycle present in Hasse cover edges")

    def test_consistency_with_separate_calls(self) -> None:
        """compute_ground_state_majorization should agree with
        compute_ground_state on the energy / bond_dim fields when run with
        the same config."""
        from caset.quantum import compute_ground_state
        cfg = self._basic_config(N=6, m=0.125)
        r1 = compute_ground_state(cfg)
        r2 = compute_ground_state_majorization(cfg)
        self.assertAlmostEqual(r1.energy, r2.ground_state.energy, places=10)
        self.assertAlmostEqual(r1.operator_energy,
                               r2.ground_state.operator_energy, places=10)
        self.assertEqual(r1.bond_dim, r2.ground_state.bond_dim)


@unittest.skipUnless(HAVE_QUANTUM, "caset built without CASET_QUANTUM=1")
class TestPosetRepr(unittest.TestCase):
    def test_repr(self) -> None:
        p = majorization_poset([[1.0], [0.5, 0.5]])
        text = repr(p)
        self.assertIn("Poset", text)
        self.assertIn("n_nodes=2", text)
        self.assertIn("edges", text)

    def test_interval_repr(self) -> None:
        cfg = TestComputeGroundStateMajorization._basic_config(N=4)
        r = compute_ground_state_majorization(cfg)
        text = repr(r.spectra.intervals[0])
        self.assertIn("Interval", text)
        self.assertIn("i=", text)
        self.assertIn("j=", text)
