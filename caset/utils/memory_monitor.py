"""Memory monitor for caset examples.

Spawns a daemon thread that periodically checks:
  1. The RSS (resident set size) of the current process.
  2. The total available memory on the system.

When the process is consuming a large fraction of system memory AND
the system is running low on available memory, the monitor kills the
process to prevent the machine from becoming unresponsive or invoking
the OOM killer on unrelated processes.

Usage::

    from memory_monitor import MemoryMonitor

    monitor = MemoryMonitor()          # starts monitoring immediately
    monitor = MemoryMonitor(
        process_rss_limit_mb=4000,     # kill if RSS exceeds 4 GB
        system_available_floor_mb=500, # ...and system has < 500 MB free
        system_critical_pct=95,        # kill if system memory >= 95% used
        check_interval=2.0,            # check every 2 seconds
    )
    # ... do work ...
    monitor.stop()                     # optional: stop the monitor thread

The monitor is intentionally dependency-free (stdlib only) so it can
be imported from any example.  It reads /proc/self/status and
/proc/meminfo on Linux; on other platforms it is a no-op.
"""
from __future__ import annotations

import os
import sys
import signal
import threading


def _read_proc_self_rss_kb() -> int | None:
    """Read VmRSS from /proc/self/status (Linux only)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])  # value is in kB
    except (OSError, ValueError, IndexError):
        pass
    return None


def _read_meminfo_available_kb() -> int | None:
    """Read MemAvailable from /proc/meminfo (Linux only)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1])  # value is in kB
    except (OSError, ValueError, IndexError):
        pass
    return None


def _read_meminfo_total_kb() -> int | None:
    """Read MemTotal from /proc/meminfo (Linux only)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


class MemoryMonitor:
    """Daemon thread that kills the process on excessive memory use.

    The process is killed when ANY of these conditions is met:
      - Process RSS > process_rss_limit_mb AND system available memory
        < system_available_floor_mb  (this process is the problem)
      - System memory usage >= system_critical_pct (system is about
        to run out regardless — bail out to avoid the OOM killer)

    On non-Linux platforms (no /proc), the monitor logs a warning at
    startup and does nothing.

    Args:
        process_rss_limit_mb:       RSS threshold in MB (default: 75% of
                                    system RAM, or 4096 MB if undetectable).
        system_available_floor_mb:  Available-memory floor in MB (default:
                                    512 MB).  Used with the RSS check.
        system_critical_pct:        Unconditional kill when system memory
                                    usage reaches this percentage
                                    (default: 95).
        check_interval:             Seconds between checks (default: 2.0).
        verbose:                    Print status messages to stderr.
    """

    def __init__(
        self,
        process_rss_limit_mb: int | None = None,
        system_available_floor_mb: int = 512,
        system_critical_pct: float = 95.0,
        check_interval: float = 2.0,
        verbose: bool = False,
    ):
        self._stop_event = threading.Event()
        self._check_interval = check_interval
        self._system_floor_kb = system_available_floor_mb * 1024
        self._system_critical_pct = system_critical_pct
        self._verbose = verbose
        self._total_kb = _read_meminfo_total_kb()
        self._last_rss_kb: int | None = _read_proc_self_rss_kb()

        # Auto-detect RSS limit as 75% of total system RAM.
        if process_rss_limit_mb is not None:
            self._rss_limit_kb = process_rss_limit_mb * 1024
        else:
            total = _read_meminfo_total_kb()
            if total is not None:
                self._rss_limit_kb = int(total * 0.75)
            else:
                self._rss_limit_kb = 4096 * 1024  # 4 GB fallback

        # Check that /proc is available before starting.
        if _read_proc_self_rss_kb() is None:
            if self._verbose:
                print(
                    "memory_monitor: /proc not available, "
                    "monitor disabled (non-Linux?)",
                    file=sys.stderr,
                )
            self._thread = None
            return

        if self._verbose:
            rss_mb = self._rss_limit_kb // 1024
            floor_mb = self._system_floor_kb // 1024
            print(
                f"memory_monitor: started "
                f"(kill when RSS > {rss_mb} MB "
                f"and available < {floor_mb} MB, "
                f"or system memory >= {self._system_critical_pct:.0f}% used, "
                f"checking every {check_interval}s)",
                file=sys.stderr,
            )

        self._thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="memory-monitor"
        )
        self._thread.start()

    @property
    def rss_mb(self) -> int | None:
        """Current process RSS in MB, or *None* if unavailable."""
        rss = self._last_rss_kb
        return rss // 1024 if rss is not None else None

    def stop(self) -> None:
        """Stop the monitor thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._check_interval + 1)

    def _monitor_loop(self) -> None:
        while not self._stop_event.wait(self._check_interval):
            rss_kb = _read_proc_self_rss_kb()
            avail_kb = _read_meminfo_available_kb()
            if rss_kb is not None:
                self._last_rss_kb = rss_kb
            if rss_kb is None or avail_kb is None:
                continue

            if self._verbose:
                print(
                    f"memory_monitor: RSS={rss_kb // 1024} MB, "
                    f"available={avail_kb // 1024} MB",
                    file=sys.stderr,
                )

            rss_mb = rss_kb // 1024
            avail_mb = avail_kb // 1024

            # Critical: system is about to run out of memory entirely.
            if self._total_kb and self._total_kb > 0:
                used_pct = (1.0 - avail_kb / self._total_kb) * 100.0
                if used_pct >= self._system_critical_pct:
                    total_mb = self._total_kb // 1024
                    print(
                        f"\nmemory_monitor: KILLING PROCESS — "
                        f"system memory {used_pct:.1f}% used "
                        f"({avail_mb} MB available / {total_mb} MB total, "
                        f"threshold {self._system_critical_pct:.0f}%) "
                        f"(process RSS: {rss_mb} MB)\n",
                        file=sys.stderr,
                        flush=True,
                    )
                    os.kill(os.getpid(), signal.SIGKILL)

            # This process is too large and the system is getting low.
            if rss_kb > self._rss_limit_kb and avail_kb < self._system_floor_kb:
                limit_mb = self._rss_limit_kb // 1024
                floor_mb = self._system_floor_kb // 1024
                print(
                    f"\nmemory_monitor: KILLING PROCESS — "
                    f"RSS {rss_mb} MB > {limit_mb} MB limit "
                    f"and only {avail_mb} MB available "
                    f"(floor {floor_mb} MB)\n",
                    file=sys.stderr,
                    flush=True,
                )
                os.kill(os.getpid(), signal.SIGKILL)
