"""Phase 4 Python tests: TDVPConfig validation, end-to-end runQqbarQuench
through the Python API, energy conservation, flux-tube formation, snapshot
schedule. Mirrors the C++ test_tdvp_string.cpp at the binding layer.

Skips cleanly when tessera was built without TESSERA_QUANTUM=1.
"""

from __future__ import annotations

import math
import unittest

try:
    from tessera.quantum import (
        TDVPConfig,
        TDVPSnapshot,
        QuenchResult,
        runQqbarQuench,
    )
    HAVE_QUANTUM = True
except ImportError:
    HAVE_QUANTUM = False


def _heavy_quark_config(
    N: int = 14, m: float = 20.0, i0: int = 5, d: int = 5,
    T: float | None = None, dt: float = 0.05,
    snapshotEvery: int = 5, maxBondDim: int = 80,
    recordSpectra: bool = False, recordPoset: bool = False,
) -> "TDVPConfig":
    cfg = TDVPConfig()
    cfg.N = N
    cfg.a = 1.0; cfg.g = 1.0; cfg.m = m; cfg.L0 = 0.0
    cfg.dmrgMaxBondDim = 32; cfg.dmrgNSweeps = 10
    cfg.i0 = i0; cfg.d = d
    cfg.dt = dt
    cfg.T = T if T is not None else d * cfg.a
    cfg.maxBondDim = maxBondDim
    cfg.cutoff = 1e-10
    cfg.krylovDim = 12
    cfg.snapshotEvery = snapshotEvery
    cfg.recordSpectra = recordSpectra
    cfg.recordPoset = recordPoset
    return cfg


def _heavy_quark_vacuum_L(N: int) -> list[float]:
    """L_n profile of the heavy-quark Néel |↑↓↑↓ … ⟩ at L0=0:
    alternating −1 (odd link), 0 (even link). Returns N-1 entries."""
    return [-1.0 if (n % 2 == 1) else 0.0 for n in range(1, N)]


def _expected_flux_tube_L(N: int, i0: int, d: int) -> list[float]:
    v = _heavy_quark_vacuum_L(N)
    for n in range(i0, i0 + d):
        v[n - 1] += 1.0
    return v


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestTDVPConfigDefaults(unittest.TestCase):
    """Default TDVPConfig should fail validation; once set, the basic
    fields round-trip through the binding."""

    def test_defaults_have_invalid_N(self) -> None:
        cfg = TDVPConfig()
        self.assertEqual(cfg.N, 0)
        # Default-construction has N=0; buildSchwingerMpo rejects this.
        with self.assertRaises(Exception):
            runQqbarQuench(cfg)

    def test_field_round_trip(self) -> None:
        cfg = _heavy_quark_config(N=12, m=10.0, i0=3, d=5, T=2.0, dt=0.1)
        self.assertEqual(cfg.N, 12)
        self.assertEqual(cfg.i0, 3)
        self.assertEqual(cfg.d, 5)
        self.assertAlmostEqual(cfg.T, 2.0)
        self.assertAlmostEqual(cfg.dt, 0.1)
        self.assertEqual(cfg.dmrgNSweeps, 10)
        self.assertEqual(cfg.snapshotEvery, 5)


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestParityValidation(unittest.TestCase):
    """The σ⁻ σ⁺ quench requires odd i0 + odd d when enforce_parity is on
    (heavy-quark Néel parity argument from quench.hpp)."""

    def test_even_i0_rejected(self) -> None:
        cfg = _heavy_quark_config(i0=4, d=5)  # even i0 → reject
        cfg.quenchEnforceParity = True
        with self.assertRaises(Exception):
            runQqbarQuench(cfg)

    def test_even_d_rejected(self) -> None:
        cfg = _heavy_quark_config(i0=3, d=4)  # even d → reject
        cfg.quenchEnforceParity = True
        with self.assertRaises(Exception):
            runQqbarQuench(cfg)

    def test_parity_bypass_runs(self) -> None:
        cfg = _heavy_quark_config(N=12, m=20.0, i0=4, d=4, T=0.4, dt=0.1,
                                  snapshotEvery=2)
        cfg.quenchEnforceParity = False
        # Should not raise; resulting state has zero amplitude on heavy-
        # quark Néel components but DMRG/quench/TDVP still execute.
        r = runQqbarQuench(cfg)
        self.assertGreater(len(r.snapshots), 0)


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestFluxTube(unittest.TestCase):
    """Phase 4 PLAN.md acceptance through the Python API: heavy-quark
    flux tube preserved during TDVP, energy conserved."""

    def test_initial_post_quench_profile(self) -> None:
        cfg = _heavy_quark_config(N=14, m=20.0, i0=5, d=5,
                                  T=5.0, dt=0.05,
                                  snapshotEvery=5)
        r = runQqbarQuench(cfg)
        s0 = r.snapshots[0]
        ref = _expected_flux_tube_L(cfg.N, cfg.i0, cfg.d)
        for n, (v, vref) in enumerate(zip(s0.lProfile, ref), start=1):
            self.assertLess(
                abs(v - vref), 0.05,
                msg=f"link {n}: got {v}, expected {vref}",
            )

    def test_mid_run_flux_tube_preserved(self) -> None:
        cfg = _heavy_quark_config(N=14, m=20.0, i0=5, d=5,
                                  T=5.0, dt=0.05, snapshotEvery=5)
        r = runQqbarQuench(cfg)
        mid = r.snapshots[len(r.snapshots) // 2]
        ref = _expected_flux_tube_L(cfg.N, cfg.i0, cfg.d)
        for n, (v, vref) in enumerate(zip(mid.lProfile, ref), start=1):
            self.assertLess(
                abs(v - vref), 0.05,
                msg=f"t={mid.time} link {n}: got {v}, expected {vref}",
            )

    def test_energy_conservation(self) -> None:
        cfg = _heavy_quark_config(N=14, m=20.0, i0=5, d=5,
                                  T=5.0, dt=0.05, snapshotEvery=5)
        r = runQqbarQuench(cfg)
        E0 = r.snapshots[0].energy
        Eend = r.snapshots[-1].energy
        rel = abs((Eend - E0) / E0)
        self.assertLess(rel, 1e-3, msg=f"|ΔE|/|E0| = {rel}")

    def test_total_charge_conserved(self) -> None:
        """Total Sz (= total charge after JW) is preserved by the quench
        (σ⁻ σ⁺ is a +1−1=0 raising/lowering pair) and by H, so it should
        stay 0 across the full evolution."""
        cfg = _heavy_quark_config(N=14, m=20.0, i0=5, d=5,
                                  T=2.0, dt=0.1, snapshotEvery=5)
        r = runQqbarQuench(cfg)
        for snap in r.snapshots:
            total_sz = 0.5 * sum(snap.zProfile)  # σ^z = 2*Sz, so Sz_total = 0.5 Σ σ^z
            self.assertLess(abs(total_sz), 1e-8,
                            msg=f"t={snap.time}: total Sz = {total_sz}")


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestSnapshotSchedule(unittest.TestCase):
    """Snapshot schedule and times line up with config.dt × snapshotEvery."""

    def test_n_snapshots_matches_schedule(self) -> None:
        # T = 1.0, dt = 0.1, snapshotEvery = 2 → take snapshots at
        # t ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0} = 6 snapshots (the last is
        # always recorded explicitly even if not on the cadence).
        cfg = _heavy_quark_config(N=10, m=20.0, i0=3, d=3,
                                  T=1.0, dt=0.1, snapshotEvery=2)
        r = runQqbarQuench(cfg)
        # Initial + every-2 + final → 6 snapshots
        # (t=0, 0.2, 0.4, 0.6, 0.8, 1.0)
        self.assertEqual(len(r.snapshots), 6)
        times = [s.time for s in r.snapshots]
        self.assertAlmostEqual(times[0], 0.0)
        self.assertAlmostEqual(times[-1], 1.0)
        for i in range(1, len(times) - 1):
            self.assertGreater(times[i], times[i - 1])

    def test_initial_snapshot_at_zero(self) -> None:
        cfg = _heavy_quark_config(N=10, T=0.5, dt=0.1, snapshotEvery=10)
        r = runQqbarQuench(cfg)
        self.assertAlmostEqual(r.snapshots[0].time, 0.0)


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestObservableRecording(unittest.TestCase):
    """recordSpectra / recordPoset toggles control optional fields."""

    def test_default_no_spectra(self) -> None:
        cfg = _heavy_quark_config(N=8, i0=1, d=3, T=0.2, dt=0.1,
                                  snapshotEvery=2,
                                  recordSpectra=False, recordPoset=False)
        r = runQqbarQuench(cfg)
        # spectra and poset structs exist but are empty.
        self.assertEqual(len(r.snapshots[0].spectra.intervals), 0)
        self.assertEqual(r.snapshots[0].poset.getNodeCount, 0)

    def test_record_spectra_populates(self) -> None:
        cfg = _heavy_quark_config(N=8, i0=1, d=3, T=0.2, dt=0.1,
                                  snapshotEvery=2,
                                  recordSpectra=True, recordPoset=False)
        r = runQqbarQuench(cfg)
        for snap in r.snapshots:
            # 8(8+1)/2 - 1 = 35 contiguous intervals
            self.assertEqual(len(snap.spectra.intervals), 35)
            # poset still empty when only recordSpectra is on.
            self.assertEqual(snap.poset.getNodeCount, 0)

    def test_record_poset_populates(self) -> None:
        cfg = _heavy_quark_config(N=8, i0=1, d=3, T=0.2, dt=0.1,
                                  snapshotEvery=2,
                                  recordSpectra=True, recordPoset=True)
        r = runQqbarQuench(cfg)
        for snap in r.snapshots:
            self.assertEqual(snap.poset.getNodeCount, 35)

    def test_schmidt_spectra_normalization_invariant(self) -> None:
        """PLAN.md §9 open question: 'sanity-check [Σ λ_α = 1] after TDVP
        truncation'. Every Schmidt spectrum at every recorded timestep
        should sum to 1 (probability normalization), regardless of how
        much real-time evolution / bond-dim truncation has been applied.

        Without this, a small truncation drift could accumulate during
        long evolutions and silently bias the majorization comparisons
        downstream. We pick parameters where TDVP truncation is non-trivial
        (light-quark, longer T) so a regression would be visible."""
        cfg = _heavy_quark_config(N=8, i0=1, d=3, T=1.0, dt=0.1,
                                  snapshotEvery=2,
                                  recordSpectra=True, recordPoset=False)
        cfg.m = 0.5  # light-quark — string spreads, entanglement grows
        cfg.cutoff = 1e-10
        r = runQqbarQuench(cfg)
        for snap in r.snapshots:
            for spec, iv in zip(snap.spectra.spectra,
                                snap.spectra.intervals):
                total = sum(spec)
                self.assertAlmostEqual(
                    total, 1.0, places=8,
                    msg=f"t={snap.time} cut [{iv.i},{iv.j}] sums to {total}")


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestRepr(unittest.TestCase):
    def test_snapshot_repr(self) -> None:
        cfg = _heavy_quark_config(N=8, i0=1, d=3, T=0.1, dt=0.1,
                                  snapshotEvery=1)
        r = runQqbarQuench(cfg)
        text = repr(r.snapshots[0])
        self.assertIn("TDVPSnapshot", text)
        self.assertIn("time=", text)
        self.assertIn("energy=", text)
        self.assertIn("bondDim=", text)
