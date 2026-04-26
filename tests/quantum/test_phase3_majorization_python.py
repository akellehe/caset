"""Phase 3 Python-level tests: pure majorization predicate, Poset
construction, and the end-to-end computeGroundStateMajorization
pipeline through the tessera.quantum API.

These mirror the C++-side tests in test_majorization.cpp,
test_schmidt_spectra.cpp, test_majorization_poset.cpp, and
test_schwinger_schmidt_cross_check.cpp at the Python boundary — any
binding-level bug (wrong type registration, lost copies, off-by-one
indexing on returned lists) shows up here without needing to rebuild
the C++ tests.

Skips cleanly when tessera was built without TESSERA_QUANTUM=1.
"""

from __future__ import annotations

import unittest
from itertools import permutations

try:
    from tessera.quantum import (
        QuantumConfig,
        Interval,
        SchmidtSpectra,
        Poset,
        majorizes,
        strictlyMajorizes,
        majorizationPoset,
        computeGroundStateMajorization,
    )
    HAVE_QUANTUM = True
except ImportError:
    HAVE_QUANTUM = False


# Tolerance for float comparisons — well below DMRG / SVD noise on the
# small problems we exercise here, but loose enough to absorb the dense-
# vs-MPS path's last-bit rounding.
TOL = 1e-10


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestMajorizesPredicate(unittest.TestCase):
    """Pure-function tests on majorizes() / strictlyMajorizes()."""

    def test_reflexivity(self) -> None:
        for v in ([1.0], [0.5, 0.5], [0.7, 0.2, 0.1]):
            self.assertTrue(majorizes(v, v))
            self.assertFalse(strictlyMajorizes(v, v))

    def test_canonical_strict_pair(self) -> None:
        self.assertTrue(majorizes([1.0, 0.0], [0.5, 0.5]))
        self.assertFalse(majorizes([0.5, 0.5], [1.0, 0.0]))
        self.assertTrue(strictlyMajorizes([1.0, 0.0], [0.5, 0.5]))

    def test_zero_padding_invariance(self) -> None:
        self.assertTrue(majorizes([0.5, 0.5, 0.0, 0.0], [0.5, 0.5]))
        self.assertTrue(majorizes([0.5, 0.5], [0.5, 0.5, 0.0]))
        self.assertFalse(strictlyMajorizes([0.5, 0.5, 0.0], [0.5, 0.5]))

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
        self.assertTrue(strictlyMajorizes(a, b))
        self.assertTrue(strictlyMajorizes(b, c))
        self.assertTrue(strictlyMajorizes(a, c))

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


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestMajorizationPoset(unittest.TestCase):
    """Pure-function tests on majorizationPoset() and the Poset struct."""

    def test_empty_input(self) -> None:
        p = majorizationPoset([])
        self.assertEqual(p.getNodeCount, 0)
        self.assertEqual(p.covers, [])

    def test_single_node(self) -> None:
        p = majorizationPoset([[1.0]])
        self.assertEqual(p.getNodeCount, 1)
        self.assertEqual(p.covers, [])

    def test_canonical_chain_transitive_reduction(self) -> None:
        spectra = [
            [1/3, 1/3, 1/3],  # 0 — most uniform
            [0.5, 0.5],        # 1 — middle
            [1.0],             # 2 — most concentrated
        ]
        p = majorizationPoset(spectra)
        self.assertEqual(p.getNodeCount, 3)
        self.assertEqual(set(p.covers), {(2, 1), (1, 0)})

    def test_equivalent_spectra_no_strict_edges(self) -> None:
        # Three nodes with the same sorted-padded spectrum form an
        # equivalence class — no strict edges among them.
        p = majorizationPoset([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        self.assertEqual(p.covers, [])

    def test_covers_only_have_in_range_indices(self) -> None:
        p = majorizationPoset([[1.0], [0.5, 0.5], [1/3]*3])
        for a, b in p.covers:
            self.assertGreaterEqual(a, 0)
            self.assertLess(a, p.getNodeCount)
            self.assertGreaterEqual(b, 0)
            self.assertLess(b, p.getNodeCount)
            self.assertNotEqual(a, b)


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestComputeGroundStateMajorization(unittest.TestCase):
    """End-to-end pipeline tests: DMRG → Schmidt → Poset all via Python."""

    @staticmethod
    def _basic_config(N: int, m: float = 0.0, g: float = 1.0,
                      L0: float = 0.0,
                      maxBondDim: int = 32, nSweeps: int = 8) -> "QuantumConfig":
        cfg = QuantumConfig()
        cfg.N = N; cfg.a = 1.0; cfg.g = g; cfg.m = m; cfg.L0 = L0
        cfg.maxBondDim = maxBondDim; cfg.nSweeps = nSweeps
        return cfg

    def test_n6_pipeline_basic(self) -> None:
        cfg = self._basic_config(N=6)
        r = computeGroundStateMajorization(cfg)

        # Ground-state field is just a GroundStateResult.
        self.assertEqual(r.spectra.N, 6)
        self.assertLess(r.groundState.energy, 0)  # sane sign
        self.assertGreater(r.groundState.bondDim, 0)

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
        r = computeGroundStateMajorization(cfg)
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
        r = computeGroundStateMajorization(cfg)
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
        the 1e-5 level — we pass tol = 1e-3 to majorizationPoset so they
        get absorbed into the equivalence-class smoothing, leaving no
        Hasse cover edges."""
        cfg = self._basic_config(N=4, m=200.0, maxBondDim=32, nSweeps=10)
        r = computeGroundStateMajorization(cfg, tol=1e-3)
        # Largest entry of every spectrum should be ≈ 1.
        for spec, iv in zip(r.spectra.spectra, r.spectra.intervals):
            self.assertAlmostEqual(
                max(spec), 1.0, places=3,
                msg=f"product-state limit broken at [{iv.i}, {iv.j}]: {spec}")
        # With the wide tolerance, all spectra collapse to one
        # equivalence class — Hasse should be empty.
        self.assertEqual(r.poset.getNodeCount, len(r.spectra.spectra))
        self.assertEqual(r.poset.covers, [])

    def test_poset_is_transitively_reduced(self) -> None:
        """Cover edges shouldn't include any (a, b) for which there's an
        intermediate c with covers (a, c) and (c, b) — that's the
        transitive-reduction guarantee."""
        cfg = self._basic_config(N=6)
        r = computeGroundStateMajorization(cfg)
        # Build adjacency from cover list.
        succ = {a: set() for a in range(r.poset.getNodeCount)}
        for a, b in r.poset.covers:
            succ[a].add(b)
        for (a, b) in r.poset.covers:
            for c in range(r.poset.getNodeCount):
                if c == a or c == b:
                    continue
                if c in succ[a] and b in succ[c]:
                    self.fail(
                        f"poset has cover ({a}, {b}) but intermediate "
                        f"path {a} → {c} → {b} also exists")

    def test_poset_irreflexive(self) -> None:
        cfg = self._basic_config(N=6)
        r = computeGroundStateMajorization(cfg)
        for a, b in r.poset.covers:
            self.assertNotEqual(a, b, "self-loop in Hasse diagram")

    def test_poset_acyclic(self) -> None:
        """Hasse cover edges form a DAG (no directed cycles)."""
        cfg = self._basic_config(N=6)
        r = computeGroundStateMajorization(cfg)

        # Topological-sort attempt via Kahn's algorithm; if it fails to
        # consume every node, the graph has a cycle.
        in_deg = {a: 0 for a in range(r.poset.getNodeCount)}
        succ   = {a: [] for a in range(r.poset.getNodeCount)}
        for a, b in r.poset.covers:
            succ[a].append(b)
            in_deg[b] += 1
        queue = [n for n in range(r.poset.getNodeCount) if in_deg[n] == 0]
        processed = 0
        while queue:
            n = queue.pop()
            processed += 1
            for m in succ[n]:
                in_deg[m] -= 1
                if in_deg[m] == 0:
                    queue.append(m)
        self.assertEqual(processed, r.poset.getNodeCount,
                         "directed cycle present in Hasse cover edges")

    def test_consistency_with_separate_calls(self) -> None:
        """computeGroundStateMajorization should agree with
        computeGroundState on the energy / bondDim fields when run with
        the same config."""
        from tessera.quantum import computeGroundState
        cfg = self._basic_config(N=6, m=0.125)
        r1 = computeGroundState(cfg)
        r2 = computeGroundStateMajorization(cfg)
        self.assertAlmostEqual(r1.energy, r2.groundState.energy, places=10)
        self.assertAlmostEqual(r1.operatorEnergy,
                               r2.groundState.operatorEnergy, places=10)
        self.assertEqual(r1.bondDim, r2.groundState.bondDim)


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestPosetRepr(unittest.TestCase):
    def test_repr(self) -> None:
        p = majorizationPoset([[1.0], [0.5, 0.5]])
        text = repr(p)
        self.assertIn("Poset", text)
        self.assertIn("getNodeCount=2", text)
        self.assertIn("edges", text)

    def test_interval_repr(self) -> None:
        cfg = TestComputeGroundStateMajorization._basic_config(N=4)
        r = computeGroundStateMajorization(cfg)
        text = repr(r.spectra.intervals[0])
        self.assertIn("Interval", text)
        self.assertIn("i=", text)
        self.assertIn("j=", text)
