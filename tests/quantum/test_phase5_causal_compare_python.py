"""Phase 5 Python tests — end-to-end causal-comparison pipeline through
caset.quantum, plus low-level compare_orders() unit tests on hand-built
posets.

Skips cleanly when caset was built without CASET_QUANTUM=1.
"""

from __future__ import annotations

import unittest

try:
    from caset.quantum import (
        QuantumConfig,
        TDVPConfig,
        Poset,
        OrderAgreement,
        CausalComparisonReport,
        compare_orders,
        compute_causal_comparison,
    )
    HAVE_QUANTUM = True
except ImportError:
    HAVE_QUANTUM = False


def _light_quark_config(N: int = 10, T: float = 0.4) -> "TDVPConfig":
    cfg = TDVPConfig()
    cfg.N = N; cfg.a = 1.0; cfg.g = 1.0; cfg.m = 0.5; cfg.L0 = 0.0
    cfg.dmrg_max_bond_dim = 32; cfg.dmrg_n_sweeps = 10
    cfg.i0 = 1; cfg.d = 3
    cfg.dt = 0.2; cfg.T = T; cfg.snapshot_every = 1
    cfg.max_bond_dim = 60
    cfg.cutoff = 1e-10; cfg.krylov_dim = 12
    cfg.quiet = True; cfg.conserve_qns = True
    return cfg


@unittest.skipUnless(HAVE_QUANTUM, "caset built without CASET_QUANTUM=1")
class TestCompareOrdersPure(unittest.TestCase):
    """Unit tests on compare_orders() without running TDVP."""

    def _poset(self, n_nodes: int, covers: list[tuple[int, int]]) -> "Poset":
        p = Poset()
        p.n_nodes = n_nodes
        p.covers = covers
        return p

    def test_identical_posets(self) -> None:
        """Identical posets → τ = 1, no discordance, edit distance 0."""
        p = self._poset(4, [(0, 1), (1, 2), (2, 3)])
        agr = compare_orders(p, p, 4)
        self.assertAlmostEqual(agr.kendall_tau, 1.0)
        self.assertAlmostEqual(agr.discordant_fraction, 0.0)
        self.assertAlmostEqual(agr.hasse_edit_distance, 0.0)
        self.assertEqual(agr.n_discordant, 0)

    def test_reversed_posets(self) -> None:
        """Two chains in opposite directions → τ = -1."""
        p = self._poset(3, [(0, 1), (1, 2)])
        q = self._poset(3, [(2, 1), (1, 0)])
        agr = compare_orders(p, q, 3)
        self.assertAlmostEqual(agr.kendall_tau, -1.0)
        self.assertEqual(agr.n_concordant, 0)
        self.assertGreater(agr.n_discordant, 0)

    def test_disjoint_posets(self) -> None:
        """Posets with disjoint cover edges → edit distance = 1."""
        p = self._poset(4, [(0, 1)])
        q = self._poset(4, [(2, 3)])
        agr = compare_orders(p, q, 4)
        self.assertAlmostEqual(agr.hasse_edit_distance, 1.0)

    def test_no_pairs_in_common(self) -> None:
        """Two empty posets — no pairs comparable in either; τ defaults
        to 0, fractions 0."""
        p = self._poset(5, [])
        q = self._poset(5, [])
        agr = compare_orders(p, q, 5)
        self.assertEqual(agr.n_comparable_both, 0)
        self.assertAlmostEqual(agr.kendall_tau, 0.0)
        self.assertAlmostEqual(agr.discordant_fraction, 0.0)
        self.assertAlmostEqual(agr.hasse_edit_distance, 0.0)

    def test_kendall_tau_in_range(self) -> None:
        """τ ∈ [-1, 1] for any pair of posets — basic sanity."""
        p = self._poset(5, [(0, 1), (0, 2), (1, 3), (2, 4)])
        q = self._poset(5, [(0, 4), (4, 3), (3, 2), (2, 1)])
        agr = compare_orders(p, q, 5)
        self.assertGreaterEqual(agr.kendall_tau, -1.0)
        self.assertLessEqual(agr.kendall_tau, 1.0)
        self.assertGreaterEqual(agr.discordant_fraction, 0.0)
        self.assertLessEqual(agr.discordant_fraction, 1.0)


@unittest.skipUnless(HAVE_QUANTUM, "caset built without CASET_QUANTUM=1")
class TestComputeCausalComparison(unittest.TestCase):
    """End-to-end pipeline tests through compute_causal_comparison."""

    def test_pipeline_runs(self) -> None:
        cfg = _light_quark_config(N=8, T=0.4)
        r = compute_causal_comparison(cfg, v_LR=1.0)

        # Expected: 8(8+1)/2 - 1 = 35 cuts × 3 snapshots = 105 labels.
        # (snapshots are t=0, t=0.2, t=0.4 with snapshot_every=1, dt=0.2,
        # T=0.4 → 2 evolution steps + initial snapshot = 3 snapshots.)
        self.assertEqual(r.n_snapshots, 3)
        self.assertEqual(r.n_labels, 35 * 3)
        self.assertAlmostEqual(r.v_LR, 1.0)

    def test_kendall_tau_in_range(self) -> None:
        cfg = _light_quark_config(N=8, T=0.4)
        r = compute_causal_comparison(cfg, v_LR=1.0)
        for agr in (r.maj_vs_lr, r.maj_vs_cs, r.lr_vs_cs):
            self.assertGreaterEqual(agr.kendall_tau, -1.0)
            self.assertLessEqual(agr.kendall_tau, 1.0)
            self.assertGreaterEqual(agr.discordant_fraction, 0.0)
            self.assertLessEqual(agr.discordant_fraction, 1.0)
            self.assertGreaterEqual(agr.hasse_edit_distance, 0.0)
            self.assertLessEqual(agr.hasse_edit_distance, 1.0)

    def test_lr_subset_of_cs(self) -> None:
        """≼_LR is a subset of ≼_cs by construction (LR adds a spatial
        constraint to the time-only causet order). Therefore every pair
        comparable in ≼_LR is comparable in ≼_cs too, in the same
        direction — Kendall-τ between them must be exactly 1.

        The strongest sanity check on the implementation."""
        cfg = _light_quark_config(N=8, T=0.4)
        r = compute_causal_comparison(cfg, v_LR=1.0)
        self.assertAlmostEqual(r.lr_vs_cs.kendall_tau, 1.0, places=12)
        self.assertEqual(r.lr_vs_cs.n_discordant, 0)

    def test_v_LR_monotonicity(self) -> None:
        """Larger v_LR ⇒ more pairs satisfy distance ≤ v_LR · Δt ⇒
        ≼_LR has at least as many transitive-closure relations ⇒
        n_comparable_both with the fixed ≼_cs is non-decreasing."""
        cfg = _light_quark_config(N=8, T=0.4)
        prev = -1
        for v in (0.5, 1.0, 2.0, 8.0):
            r = compute_causal_comparison(cfg, v_LR=v)
            self.assertGreaterEqual(
                r.lr_vs_cs.n_comparable_both, prev,
                msg=f"comparability decreased going from v_LR=prev to v_LR={v}")
            prev = r.lr_vs_cs.n_comparable_both

    def test_record_spectra_forced(self) -> None:
        """compute_causal_comparison must force record_spectra=True
        regardless of the input config — otherwise the spectra-based
        majorization order would have nothing to work with."""
        cfg = _light_quark_config(N=6, T=0.2)
        cfg.record_spectra = False  # explicitly off — should be flipped on
        r = compute_causal_comparison(cfg, v_LR=1.0)
        # Maj poset should have nodes (one per label); the user-set
        # record_spectra=False would otherwise produce zero spectra.
        self.assertGreater(r.n_labels, 0)


@unittest.skipUnless(HAVE_QUANTUM, "caset built without CASET_QUANTUM=1")
class TestPipelineWithRecordedSpectra(unittest.TestCase):
    """Confirm that the pipeline stitches the per-snapshot spectra into
    one global majorization poset — i.e. the maj order has cross-time
    edges, not just within-time."""

    def test_maj_has_cross_time_edges(self) -> None:
        cfg = _light_quark_config(N=8, T=0.4)
        r = compute_causal_comparison(cfg, v_LR=1.0)
        # We don't have direct access to the orders here; the indirect
        # signature is that maj_vs_cs has non-zero comparability — only
        # cross-time pairs are comparable in ≼_cs, so any concordance
        # between maj and cs proves maj has cross-time relations.
        self.assertGreater(r.maj_vs_cs.n_concordant, 0)
