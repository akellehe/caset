"""Tests for examples/memory_monitor.py."""

import os
import sys
import signal
import threading
import time
import unittest
from unittest import mock

from caset.utils import memory_monitor
from caset.utils.memory_monitor import (
    MemoryMonitor,
    _read_meminfo_available_kb,
    _read_meminfo_total_kb,
    _read_proc_self_rss_kb,
)


# ---------------------------------------------------------------------------
# Tests for the /proc reader helpers
# ---------------------------------------------------------------------------

class TestProcReaders(unittest.TestCase):
    """Test the low-level /proc readers on Linux (skip elsewhere)."""

    @unittest.skipUnless(os.path.exists("/proc/self/status"), "Linux only")
    def test_read_rss_returns_positive_int(self):
        rss = _read_proc_self_rss_kb()
        self.assertIsNotNone(rss)
        self.assertGreater(rss, 0)

    @unittest.skipUnless(os.path.exists("/proc/meminfo"), "Linux only")
    def test_read_available_returns_positive_int(self):
        avail = _read_meminfo_available_kb()
        self.assertIsNotNone(avail)
        self.assertGreater(avail, 0)

    @unittest.skipUnless(os.path.exists("/proc/meminfo"), "Linux only")
    def test_read_total_returns_positive_int(self):
        total = _read_meminfo_total_kb()
        self.assertIsNotNone(total)
        self.assertGreater(total, 0)

    @unittest.skipUnless(os.path.exists("/proc/meminfo"), "Linux only")
    def test_available_less_than_total(self):
        avail = _read_meminfo_available_kb()
        total = _read_meminfo_total_kb()
        self.assertLess(avail, total)

    def test_rss_returns_none_on_missing_file(self):
        with mock.patch("builtins.open", side_effect=OSError("no such file")):
            self.assertIsNone(_read_proc_self_rss_kb())

    def test_available_returns_none_on_missing_file(self):
        with mock.patch("builtins.open", side_effect=OSError("no such file")):
            self.assertIsNone(_read_meminfo_available_kb())

    def test_total_returns_none_on_missing_file(self):
        with mock.patch("builtins.open", side_effect=OSError("no such file")):
            self.assertIsNone(_read_meminfo_total_kb())


# ---------------------------------------------------------------------------
# Helpers for mocking /proc in MemoryMonitor tests
# ---------------------------------------------------------------------------

def _patch_proc(rss_kb, avail_kb, total_kb):
    """Return a context manager that patches all three /proc readers."""
    return mock.patch.multiple(
        "caset.utils.memory_monitor",
        _read_proc_self_rss_kb=mock.DEFAULT,
        _read_meminfo_available_kb=mock.DEFAULT,
        _read_meminfo_total_kb=mock.DEFAULT,
    )


class _FakeProcValues:
    """Thread-safe mutable container for fake /proc values."""

    def __init__(self, rss_kb, avail_kb, total_kb):
        self.lock = threading.Lock()
        self.rss_kb = rss_kb
        self.avail_kb = avail_kb
        self.total_kb = total_kb

    def get_rss(self):
        with self.lock:
            return self.rss_kb

    def get_avail(self):
        with self.lock:
            return self.avail_kb

    def get_total(self):
        with self.lock:
            return self.total_kb


# ---------------------------------------------------------------------------
# Tests for MemoryMonitor construction and lifecycle
# ---------------------------------------------------------------------------

class TestMonitorConstruction(unittest.TestCase):

    @mock.patch("caset.utils.memory_monitor._read_proc_self_rss_kb", return_value=None)
    @mock.patch("caset.utils.memory_monitor._read_meminfo_total_kb", return_value=16_000_000)
    def test_no_proc_disables_monitor(self, _mock_total, _mock_rss):
        """On non-Linux, the monitor should be a no-op (no thread)."""
        mon = MemoryMonitor(check_interval=0.05)
        self.assertIsNone(mon._thread)
        mon.stop()  # should not raise

    @mock.patch("caset.utils.memory_monitor._read_proc_self_rss_kb", return_value=100_000)
    @mock.patch("caset.utils.memory_monitor._read_meminfo_available_kb", return_value=8_000_000)
    @mock.patch("caset.utils.memory_monitor._read_meminfo_total_kb", return_value=16_000_000)
    def test_thread_starts_on_linux(self, *_):
        mon = MemoryMonitor(check_interval=0.05)
        try:
            self.assertIsNotNone(mon._thread)
            self.assertTrue(mon._thread.is_alive())
            self.assertTrue(mon._thread.daemon)
        finally:
            mon.stop()

    @mock.patch("caset.utils.memory_monitor._read_proc_self_rss_kb", return_value=100_000)
    @mock.patch("caset.utils.memory_monitor._read_meminfo_available_kb", return_value=8_000_000)
    @mock.patch("caset.utils.memory_monitor._read_meminfo_total_kb", return_value=16_000_000)
    def test_stop_joins_thread(self, *_):
        mon = MemoryMonitor(check_interval=0.05)
        mon.stop()
        self.assertFalse(mon._thread.is_alive())

    @mock.patch("caset.utils.memory_monitor._read_proc_self_rss_kb", return_value=100_000)
    @mock.patch("caset.utils.memory_monitor._read_meminfo_available_kb", return_value=8_000_000)
    @mock.patch("caset.utils.memory_monitor._read_meminfo_total_kb", return_value=16_000_000)
    def test_explicit_rss_limit(self, *_):
        mon = MemoryMonitor(process_rss_limit_mb=2048, check_interval=0.05)
        try:
            self.assertEqual(mon._rss_limit_kb, 2048 * 1024)
        finally:
            mon.stop()

    @mock.patch("caset.utils.memory_monitor._read_proc_self_rss_kb", return_value=100_000)
    @mock.patch("caset.utils.memory_monitor._read_meminfo_available_kb", return_value=8_000_000)
    @mock.patch("caset.utils.memory_monitor._read_meminfo_total_kb", return_value=16_000_000)
    def test_auto_rss_limit_is_75pct_of_total(self, *_):
        mon = MemoryMonitor(check_interval=0.05)
        try:
            self.assertEqual(mon._rss_limit_kb, int(16_000_000 * 0.75))
        finally:
            mon.stop()

    @mock.patch("caset.utils.memory_monitor._read_proc_self_rss_kb", return_value=100_000)
    @mock.patch("caset.utils.memory_monitor._read_meminfo_available_kb", return_value=8_000_000)
    @mock.patch("caset.utils.memory_monitor._read_meminfo_total_kb", return_value=None)
    def test_auto_rss_limit_fallback_when_total_unknown(self, *_):
        mon = MemoryMonitor(check_interval=0.05)
        try:
            self.assertEqual(mon._rss_limit_kb, 4096 * 1024)
        finally:
            mon.stop()


# ---------------------------------------------------------------------------
# Tests for kill conditions
# ---------------------------------------------------------------------------

class TestKillConditions(unittest.TestCase):
    """Verify the monitor kills the process under the right conditions.

    All tests mock os.kill to capture the signal instead of actually
    killing the test process.
    """

    def _run_monitor_once(self, rss_kb, avail_kb, total_kb, **kwargs):
        """Create a monitor with fake /proc values, let it check once.

        Returns the list of (pid, sig) tuples passed to os.kill.
        """
        fake = _FakeProcValues(rss_kb, avail_kb, total_kb)
        kills = []

        with mock.patch("caset.utils.memory_monitor._read_proc_self_rss_kb", side_effect=lambda: fake.get_rss()), \
             mock.patch("caset.utils.memory_monitor._read_meminfo_available_kb", side_effect=lambda: fake.get_avail()), \
             mock.patch("caset.utils.memory_monitor._read_meminfo_total_kb", side_effect=lambda: fake.get_total()), \
             mock.patch("caset.utils.memory_monitor.os.kill", side_effect=lambda pid, sig: kills.append((pid, sig))):
            kwargs.setdefault("check_interval", 0.02)
            mon = MemoryMonitor(**kwargs)
            # Let the monitor thread run at least one check cycle
            time.sleep(0.15)
            mon.stop()

        return kills

    def test_no_kill_when_everything_fine(self):
        # RSS=100MB, 8GB available, 16GB total → 50% used, well under limits
        kills = self._run_monitor_once(
            rss_kb=100_000, avail_kb=8_000_000, total_kb=16_000_000)
        self.assertEqual(kills, [])

    def test_no_kill_high_rss_but_plenty_available(self):
        # RSS exceeds 75% of 16GB (>12GB), but system has 5GB free (68% used)
        kills = self._run_monitor_once(
            rss_kb=13_000_000, avail_kb=5_000_000, total_kb=16_000_000)
        self.assertEqual(kills, [])

    def test_no_kill_low_available_but_small_rss(self):
        # System only has 300MB free (98% used) but RSS is tiny
        # → still triggers critical pct (95%) so should kill
        kills = self._run_monitor_once(
            rss_kb=50_000, avail_kb=300_000, total_kb=16_000_000)
        self.assertGreater(len(kills), 0)

    def test_kill_on_high_rss_and_low_available(self):
        # RSS=13GB > 75% of 16GB, available=400MB < 512MB floor
        kills = self._run_monitor_once(
            rss_kb=13_000_000, avail_kb=400_000, total_kb=16_000_000)
        self.assertGreater(len(kills), 0)
        self.assertEqual(kills[0], (os.getpid(), signal.SIGKILL))

    def test_kill_on_system_critical_pct(self):
        # 16GB total, 500MB available → 96.9% used > 95% threshold
        kills = self._run_monitor_once(
            rss_kb=100_000, avail_kb=500_000, total_kb=16_000_000)
        self.assertGreater(len(kills), 0)
        self.assertEqual(kills[0], (os.getpid(), signal.SIGKILL))

    def test_no_kill_just_under_critical_pct(self):
        # 16GB total, 1GB available → 93.75% used < 95%
        kills = self._run_monitor_once(
            rss_kb=100_000, avail_kb=1_000_000, total_kb=16_000_000)
        self.assertEqual(kills, [])

    def test_custom_critical_pct(self):
        # 16GB total, 3.2GB available → 80% used
        # Default 95% would not kill, but custom 75% should
        kills = self._run_monitor_once(
            rss_kb=100_000, avail_kb=3_200_000, total_kb=16_000_000,
            system_critical_pct=75.0)
        self.assertGreater(len(kills), 0)

    def test_custom_critical_pct_no_kill_under(self):
        # 16GB total, 5GB available → 68.75% used, custom 75% → no kill
        kills = self._run_monitor_once(
            rss_kb=100_000, avail_kb=5_000_000, total_kb=16_000_000,
            system_critical_pct=75.0)
        self.assertEqual(kills, [])

    def test_custom_rss_limit_triggers_kill(self):
        # RSS=600MB > 500MB limit, available=400MB < 512MB floor
        kills = self._run_monitor_once(
            rss_kb=600_000, avail_kb=400_000, total_kb=16_000_000,
            process_rss_limit_mb=500)
        self.assertGreater(len(kills), 0)

    def test_custom_rss_limit_no_kill_under(self):
        # RSS=400MB < 500MB limit, available=400MB < 512MB floor
        # RSS condition not met even though available is low
        # But system usage = (16M-400k)/16M = 97.5% → critical pct triggers
        # Use a high critical pct to isolate the RSS check
        kills = self._run_monitor_once(
            rss_kb=400_000, avail_kb=400_000, total_kb=16_000_000,
            process_rss_limit_mb=500, system_critical_pct=99.0)
        self.assertEqual(kills, [])

    def test_custom_floor_triggers_kill(self):
        # RSS=13GB > 75% of 16GB, available=900MB < 1024MB custom floor
        kills = self._run_monitor_once(
            rss_kb=13_000_000, avail_kb=900_000, total_kb=16_000_000,
            system_available_floor_mb=1024, system_critical_pct=99.0)
        self.assertGreater(len(kills), 0)

    def test_custom_floor_no_kill_above(self):
        # RSS=13GB > 75% of 16GB, available=1.1GB > 1024MB custom floor
        # System 93% used < 99% critical
        kills = self._run_monitor_once(
            rss_kb=13_000_000, avail_kb=1_100_000, total_kb=16_000_000,
            system_available_floor_mb=1024, system_critical_pct=99.0)
        self.assertEqual(kills, [])

    def test_kill_uses_sigkill(self):
        kills = self._run_monitor_once(
            rss_kb=13_000_000, avail_kb=400_000, total_kb=16_000_000)
        self.assertEqual(kills[0][1], signal.SIGKILL)

    def test_kill_targets_own_pid(self):
        kills = self._run_monitor_once(
            rss_kb=13_000_000, avail_kb=400_000, total_kb=16_000_000)
        self.assertEqual(kills[0][0], os.getpid())


# ---------------------------------------------------------------------------
# Tests for edge cases and robustness
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):

    def test_none_rss_skips_check(self):
        """If /proc/self/status becomes unreadable mid-run, don't crash."""
        call_count = [0]

        def flaky_rss():
            call_count[0] += 1
            return None  # simulate /proc read failure

        with mock.patch("caset.utils.memory_monitor._read_proc_self_rss_kb", return_value=100_000), \
             mock.patch("caset.utils.memory_monitor._read_meminfo_total_kb", return_value=16_000_000):
            mon = MemoryMonitor(check_interval=0.02)

        # Now make RSS unreadable during the loop
        with mock.patch("caset.utils.memory_monitor._read_proc_self_rss_kb", side_effect=flaky_rss), \
             mock.patch("caset.utils.memory_monitor._read_meminfo_available_kb", return_value=400_000), \
             mock.patch("caset.utils.memory_monitor.os.kill") as mock_kill:
            time.sleep(0.1)
            mon.stop()
            # Should not have killed because rss was None
            mock_kill.assert_not_called()

    def test_none_avail_skips_check(self):
        """If /proc/meminfo becomes unreadable mid-run, don't crash."""
        with mock.patch("caset.utils.memory_monitor._read_proc_self_rss_kb", return_value=100_000), \
             mock.patch("caset.utils.memory_monitor._read_meminfo_total_kb", return_value=16_000_000):
            mon = MemoryMonitor(check_interval=0.02)

        with mock.patch("caset.utils.memory_monitor._read_proc_self_rss_kb", return_value=13_000_000), \
             mock.patch("caset.utils.memory_monitor._read_meminfo_available_kb", return_value=None), \
             mock.patch("caset.utils.memory_monitor.os.kill") as mock_kill:
            time.sleep(0.1)
            mon.stop()
            mock_kill.assert_not_called()

    def test_total_kb_none_skips_critical_pct(self):
        """If total RAM is unknown, the critical-pct check is skipped."""
        kills = []
        with mock.patch("caset.utils.memory_monitor._read_proc_self_rss_kb", return_value=100_000), \
             mock.patch("caset.utils.memory_monitor._read_meminfo_available_kb", return_value=400_000), \
             mock.patch("caset.utils.memory_monitor._read_meminfo_total_kb", return_value=None), \
             mock.patch("caset.utils.memory_monitor.os.kill", side_effect=lambda p, s: kills.append((p, s))):
            mon = MemoryMonitor(check_interval=0.02)
            time.sleep(0.15)
            mon.stop()
        # RSS=100MB < 4096MB fallback, and critical pct can't fire with no total
        self.assertEqual(kills, [])

    def test_stop_is_idempotent(self):
        with mock.patch("caset.utils.memory_monitor._read_proc_self_rss_kb", return_value=100_000), \
             mock.patch("caset.utils.memory_monitor._read_meminfo_available_kb", return_value=8_000_000), \
             mock.patch("caset.utils.memory_monitor._read_meminfo_total_kb", return_value=16_000_000):
            mon = MemoryMonitor(check_interval=0.02)
            mon.stop()
            mon.stop()  # should not raise

    def test_stop_on_noop_monitor(self):
        """stop() on a no-op monitor (non-Linux) doesn't raise."""
        with mock.patch("caset.utils.memory_monitor._read_proc_self_rss_kb", return_value=None), \
             mock.patch("caset.utils.memory_monitor._read_meminfo_total_kb", return_value=None):
            mon = MemoryMonitor()
            mon.stop()

    def test_verbose_prints_to_stderr(self, ):
        with mock.patch("caset.utils.memory_monitor._read_proc_self_rss_kb", return_value=100_000), \
             mock.patch("caset.utils.memory_monitor._read_meminfo_available_kb", return_value=8_000_000), \
             mock.patch("caset.utils.memory_monitor._read_meminfo_total_kb", return_value=16_000_000), \
             mock.patch("sys.stderr") as mock_stderr:
            mon = MemoryMonitor(check_interval=0.02, verbose=True)
            time.sleep(0.1)
            mon.stop()
            # Verbose mode should have written something
            self.assertTrue(mock_stderr.write.called)


# ---------------------------------------------------------------------------
# Test that dynamic value changes are picked up
# ---------------------------------------------------------------------------

class TestDynamicValues(unittest.TestCase):

    def test_kill_triggered_after_memory_pressure_increases(self):
        """Monitor should kill when memory pressure rises mid-run."""
        fake = _FakeProcValues(
            rss_kb=100_000, avail_kb=8_000_000, total_kb=16_000_000)
        kills = []

        with mock.patch("caset.utils.memory_monitor._read_proc_self_rss_kb", side_effect=lambda: fake.get_rss()), \
             mock.patch("caset.utils.memory_monitor._read_meminfo_available_kb", side_effect=lambda: fake.get_avail()), \
             mock.patch("caset.utils.memory_monitor._read_meminfo_total_kb", side_effect=lambda: fake.get_total()), \
             mock.patch("caset.utils.memory_monitor.os.kill", side_effect=lambda p, s: kills.append((p, s))):
            mon = MemoryMonitor(check_interval=0.02)
            # Let it run a couple checks with safe values
            time.sleep(0.1)
            self.assertEqual(kills, [])

            # Now simulate memory pressure
            with fake.lock:
                fake.avail_kb = 500_000  # 96.9% used → above 95% critical
            time.sleep(0.1)
            mon.stop()

        self.assertGreater(len(kills), 0)

    def test_no_kill_when_pressure_resolves(self):
        """If memory was briefly high but recovers, no kill."""
        fake = _FakeProcValues(
            rss_kb=13_000_000, avail_kb=8_000_000, total_kb=16_000_000)
        kills = []

        with mock.patch("caset.utils.memory_monitor._read_proc_self_rss_kb", side_effect=lambda: fake.get_rss()), \
             mock.patch("caset.utils.memory_monitor._read_meminfo_available_kb", side_effect=lambda: fake.get_avail()), \
             mock.patch("caset.utils.memory_monitor._read_meminfo_total_kb", side_effect=lambda: fake.get_total()), \
             mock.patch("caset.utils.memory_monitor.os.kill", side_effect=lambda p, s: kills.append((p, s))):
            mon = MemoryMonitor(check_interval=0.02, system_critical_pct=99.0)
            time.sleep(0.1)
            self.assertEqual(kills, [])
            mon.stop()

        self.assertEqual(kills, [])


if __name__ == "__main__":
    unittest.main()
