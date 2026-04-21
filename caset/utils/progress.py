"""Shared terminal progress display for caset examples.

Provides two display classes:

  ProgressDisplay   – thread-safe multi-line display for parallel workers
                      (configs, grid points, chains, etc.)
  SingleTaskProgress – single-line spinner for sequential scripts
                      (build → tune → thermalize → solve → render)

Both use ANSI colors and emoji when connected to a TTY, and degrade
gracefully to plain text otherwise.  Both show a nonlinear ETA estimate
that adapts to sub-linear and super-linear progress curves, and a
sliding-window sparkline showing per-iteration timing trends.
"""
import atexit
import sys
import threading
import time

from caset.utils.eta import ETAEstimator, _fmt_duration

try:
    import termios
    _HAS_TERMIOS = True
except ImportError:
    _HAS_TERMIOS = False

_SPARK_CHARS = " ▁▂▃▄▅▆▇█"

# ─── ANSI helpers ────────────────────────────────────────────────────
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RST = "\033[0m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_MAGENTA = "\033[35m"
_BLUE = "\033[34m"
_RED = "\033[31m"
_WHITE = "\033[37m"
_ERASE_LINE = "\033[2K"

PHASE_STYLE = {
    "building":      ("🔨", _YELLOW),
    "tuning":        ("🎛️ ", _MAGENTA),
    "thermalizing":  ("🔥", _CYAN),
    "sweeping":      ("🔄", _CYAN),
    "decorrelating": ("🔀", _BLUE),
    "measuring":     ("📐", _BLUE),
    "diffusing":     ("🌊", _BLUE),
    "solving":       ("⚙️ ", _MAGENTA),
    "rendering":     ("🎨", _YELLOW),
    "embedding":     ("📐", _BLUE),
    "computing":     ("🧮", _BLUE),
    "layouting":     ("📐", _BLUE),
    "saving":        ("💾", _GREEN),
    "done":          ("✅", _GREEN),
}
_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


# ─── Stdin echo suppression ──────────────────────────────────────────
#
# The live display moves the cursor up and rewrites lines in place.  If
# the user hits Enter (or types anything) while a display is active, the
# terminal echoes that input, which inserts a newline into the output
# stream and throws off the cursor-up line count — the next redraw then
# overwrites part of its own prior output.  Suppressing input echo on
# stdin while a display is active eliminates the corruption.
#
# ISIG is left on so Ctrl-C still raises KeyboardInterrupt; ICANON is
# left on so input remains line-buffered (nothing reads stdin, but we
# want to stay as close to normal terminal semantics as possible).

class _StdinEchoSuppressor:
    """Disable stdin echo while a progress display is active.

    No-op when stdin is not a TTY or when termios is unavailable
    (Windows).  Registers an atexit hook the first time it activates so
    the terminal is restored even if the process exits abnormally
    without reaching ``finish()``.
    """

    def __init__(self):
        self._old = None
        self._fd = None

    def enable(self):
        if self._old is not None:
            return
        if not _HAS_TERMIOS or not sys.stdin.isatty():
            return
        try:
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            new_attrs = list(self._old)
            # c_lflag is index 3. Clearing ECHO silences typed chars;
            # clearing ECHONL silences newline echo when ICANON is on.
            new_attrs[3] &= ~(termios.ECHO | termios.ECHONL)
            termios.tcsetattr(self._fd, termios.TCSANOW, new_attrs)
            atexit.register(self.disable)
        except (termios.error, OSError, ValueError):
            self._old = None
            self._fd = None

    def disable(self):
        if self._old is None:
            return
        try:
            termios.tcsetattr(self._fd, termios.TCSANOW, self._old)
        except (termios.error, OSError, ValueError):
            pass
        finally:
            self._old = None
            self._fd = None


def _mini_bar(done, total, width=10, color=True):
    """Compact progress bar: ▓▓▓▓░░░░░░ 40%"""
    if total <= 0:
        return ""
    frac = min(done / total, 1.0)
    filled = int(width * frac)
    pct = frac * 100
    bar = "▓" * filled + "░" * (width - filled)
    if color:
        return f"{bar} {done}/{total} ({pct:.0f}%)"
    return f"{bar} {done}/{total} ({pct:.0f}%)"


def make_tune_cb(phase_cb, item_id=None):
    """Create a progress callback for cdt.tune() from a phase callback.

    For multi-worker examples (ProgressDisplay), pass the phase_cb and
    item_id.  For single-task examples (SingleTaskProgress), pass None
    for item_id and use prog.on_tick directly instead.

    Returns a callable(i, n) suitable for cdt.tune(progress=...).
    """
    if phase_cb is None:
        return None
    if item_id is not None:
        return lambda i, n: phase_cb(item_id, "tuning", i, n)
    return lambda i, n: phase_cb("tuning", i, n)


# ─── Sparkline ───────────────────────────────────────────────────────

class _Sparkline:
    """Sliding-window sparkline of per-tick duration.

    Records monotonic timestamps on each ``record()`` call and renders
    the last *width* inter-tick durations as a Unicode block-element
    mini line chart.  Taller bars = slower iterations.

    Thread-safe: callers must hold the owning display's lock.
    """

    def __init__(self, window=30):
        self._times: list[float] = []
        self._window = window

    def record(self):
        """Record the current timestamp (call once per tick/sweep)."""
        self._times.append(time.monotonic())
        # Keep one extra so we can compute `window` durations
        if len(self._times) > self._window + 1:
            self._times = self._times[-(self._window + 1):]

    def reset(self):
        """Clear history (e.g. on phase change)."""
        self._times.clear()

    def render(self, width=20, color=True):
        """Return a sparkline string, or '' if not enough data."""
        if len(self._times) < 3:
            return ""
        durations = [self._times[i + 1] - self._times[i]
                     for i in range(len(self._times) - 1)]
        # Show the most recent `width` durations
        durations = durations[-width:]

        lo = min(durations)
        hi = max(durations)
        span = hi - lo

        chars = []
        n_levels = len(_SPARK_CHARS) - 1  # 8
        for d in durations:
            if span < 1e-9:
                level = n_levels // 2
            else:
                level = int((d - lo) / span * n_levels + 0.5)
                level = max(0, min(n_levels, level))
            chars.append(_SPARK_CHARS[level])
        spark = "".join(chars)

        latest = durations[-1]
        if latest >= 10:
            rate = f"{latest:.0f}s/it"
        elif latest >= 1:
            rate = f"{latest:.1f}s/it"
        elif latest >= 0.01:
            rate = f"{latest*1000:.0f}ms/it"
        else:
            rate = f"{latest*1000:.1f}ms/it"

        if color:
            return f"{_DIM}📈{spark} {rate}{_RST}"
        return f"[{spark}] {rate}"


# ─── Multi-worker display ────────────────────────────────────────────

class ProgressDisplay:
    """Thread-safe multi-line progress display with emoji phases and a spinner.

    Designed for parallel workloads where multiple configs/points/chains
    are running concurrently.  Shows a nonlinear ETA based on item
    completion rate.

    Args:
        n_items:      Total number of work items (configs, grid points, …).
        total_sweeps: Total expected sweeps across all items.
        item_label:   Noun for the progress bar (e.g. "Configs", "Points").
        use_color:    Whether to emit ANSI codes (auto-detected from TTY).
    """

    # Emit a plain-text progress line to stdout at this interval (seconds).
    # Useful for long-running jobs where the live terminal display may not
    # be visible (nohup, detached screen/tmux, CI logs, etc.).
    _LOG_INTERVAL = 120  # 2 minutes

    def __init__(self, n_items, total_sweeps, *,
                 item_label="Configs", use_color=True,
                 memory_monitor=None):
        self._lock = threading.Lock()
        self.n_items = n_items
        self.total_sweeps = total_sweeps
        self.item_label = item_label
        self.completed = 0
        self.sweeps_done = 0
        self.use_color = use_color and sys.stderr.isatty()
        self._memory_monitor = memory_monitor

        self._phases = {}       # item_id → phase string
        self._info = {}         # item_id → extra info string
        self._progress = {}     # item_id → (done, total)
        self._lines_drawn = 0
        self._spin_idx = 0
        self._start = time.time()
        self._active = True
        self._eta = ETAEstimator()
        self._sparkline = _Sparkline()
        self._last_log = time.monotonic()

        self._echo_suppressor = _StdinEchoSuppressor()
        self._echo_suppressor.enable()

        self._spinner_thread = threading.Thread(
            target=self._spin_loop, daemon=True)
        self._spinner_thread.start()

    # ── callbacks (called from worker threads) ──

    def on_phase(self, item_id, phase, done=0, total=0):
        with self._lock:
            self._phases[item_id] = phase
            if total > 0:
                self._progress[item_id] = (done, total)
            elif item_id in self._progress:
                del self._progress[item_id]

    def on_sweep(self, _i, _n):
        with self._lock:
            self.sweeps_done += 1
            self._eta.update(self.sweeps_done, self.total_sweeps)
            self._sparkline.record()

    def on_item_done(self, item_id, info=""):
        with self._lock:
            self._phases[item_id] = "done"
            self._info[item_id] = info
            self._progress.pop(item_id, None)
            self.completed += 1
            self._eta.update(self.completed, self.n_items)

    def finish(self):
        self._active = False
        self._spinner_thread.join(timeout=1)
        with self._lock:
            # Only draw if the spinner thread never got a chance to
            # (e.g. workers completed in < 80 ms).  Otherwise the
            # spinner's last iteration already rendered the final state
            # and re-drawing causes a visible flash of duplication.
            if self._lines_drawn == 0:
                self._draw()
            sys.stderr.write("\n")
            sys.stderr.flush()
        self._echo_suppressor.disable()

    # ── rendering ──

    def _spin_loop(self):
        while self._active:
            time.sleep(0.08)
            with self._lock:
                self._spin_idx = (self._spin_idx + 1) % len(_SPINNER)
                self._draw()
                self._maybe_log()

    def _maybe_log(self):
        """Emit a plain-text progress summary to stdout periodically.

        Runs under self._lock.  The line is written to stdout (not stderr)
        so it persists in scrollback and is captured by nohup/CI.
        """
        now = time.monotonic()
        if now - self._last_log < self._LOG_INTERVAL:
            return
        self._last_log = now

        elapsed = time.time() - self._start
        sw_pct = (self.sweeps_done / self.total_sweeps * 100
                  ) if self.total_sweeps else 0

        # Per-item status summary
        item_parts = []
        for item_id in sorted(self._phases.keys()):
            phase = self._phases[item_id]
            prog_pair = self._progress.get(item_id)
            if prog_pair:
                d, t = prog_pair
                item_parts.append(f"#{item_id+1} {phase} {d}/{t}")
            else:
                item_parts.append(f"#{item_id+1} {phase}")
        items_str = ", ".join(item_parts) if item_parts else ""

        eta_str = self._eta.format_eta()  # "ETA 2m 15s" or "ETA --:--"

        mem_str = ""
        if self._memory_monitor is not None:
            rss = self._memory_monitor.rss_mb
            if rss is not None:
                mem_str = f", RSS {rss} MB"

        line = (
            f"[{_fmt_duration(elapsed)} elapsed] "
            f"{self.item_label}: {self.completed}/{self.n_items} done, "
            f"sweeps: {self.sweeps_done}/{self.total_sweeps} ({sw_pct:.0f}%), "
            f"{eta_str}{mem_str}"
        )
        if items_str:
            line += f"\n  {items_str}"
        # Write to stdout so it persists in scrollback / nohup.out
        print(line, flush=True)

    def _draw(self):
        c = self.use_color
        out = []

        if self._lines_drawn > 0:
            out.append(f"\033[{self._lines_drawn}A")

        elapsed = time.time() - self._start
        spin = _SPINNER[self._spin_idx] if c else "|"

        pct = (self.completed / self.n_items * 100) if self.n_items else 0
        bar_w = 24
        filled = int(bar_w * self.completed / self.n_items) if self.n_items else 0
        bar = "█" * filled + "░" * (bar_w - filled)
        sw_pct = (self.sweeps_done / self.total_sweeps * 100) if self.total_sweeps else 0

        eta_str = self._eta.format_compact()
        if eta_str:
            eta_str = f"  {eta_str}"

        mem_str = ""
        if self._memory_monitor is not None:
            rss = self._memory_monitor.rss_mb
            if rss is not None:
                mem_str = f"  💾 {rss} MB" if c else f"  RSS {rss} MB"

        if c:
            out.append(
                f"{_ERASE_LINE}{_BOLD}📊 {self.item_label}{_RST} "
                f"{_GREEN}{bar}{_RST} "
                f"{_BOLD}{self.completed}{_RST}/{self.n_items} "
                f"{_DIM}({pct:.0f}%){_RST}  "
                f"{_BOLD}🔄 Sweeps{_RST} {self.sweeps_done}/{self.total_sweeps} "
                f"{_DIM}({sw_pct:.0f}%){_RST}  "
                f"{_DIM}⏱ {elapsed:.1f}s{_RST}"
                f"{_DIM}{eta_str}{_RST}"
                f"{_DIM}{mem_str}{_RST}"
            )
        else:
            out.append(
                f"{_ERASE_LINE}{self.item_label} [{bar}] "
                f"{self.completed}/{self.n_items} ({pct:.0f}%)  "
                f"Sweeps {self.sweeps_done}/{self.total_sweeps} "
                f"({sw_pct:.0f}%)  {elapsed:.1f}s"
                f"{eta_str}"
                f"{mem_str}"
            )

        spark_str = self._sparkline.render(width=30, color=c)
        if spark_str:
            out.append(f"{_ERASE_LINE}  {spark_str}")

        for item_id in sorted(self._phases.keys()):
            phase = self._phases[item_id]
            emoji, color = PHASE_STYLE.get(phase, ("⏳", _WHITE))
            extra = self._info.get(item_id, "")
            if extra:
                extra = f"  {extra}"

            prog_str = ""
            prog_pair = self._progress.get(item_id)
            if prog_pair:
                d, t = prog_pair
                prog_str = f" {_mini_bar(d, t, width=10, color=c)}" if t else ""

            if c:
                indicator = emoji if phase == "done" else f"{color}{spin}{_RST} {emoji}"
                out.append(
                    f"{_ERASE_LINE}  {indicator} "
                    f"{_DIM}#{item_id+1:>2}{_RST} "
                    f"{color}{phase}{_RST}"
                    f"{_DIM}{prog_str}{extra}{_RST}"
                )
            else:
                out.append(
                    f"{_ERASE_LINE}  {emoji} #{item_id+1:>2} {phase}"
                    f"{prog_str}{extra}"
                )

        # Erase leftover lines if the display shrank (e.g. items finished)
        while len(out) < self._lines_drawn:
            out.append(_ERASE_LINE)
        self._lines_drawn = len(out)
        sys.stderr.write("\n".join(out) + "\n")
        sys.stderr.flush()


# ─── Single-task sequential display ──────────────────────────────────

class SingleTaskProgress:
    """Single-line spinner for sequential scripts.

    Shows the current phase with an animated spinner and optional
    counter (e.g. sweep count, solver iteration).  When a total is
    provided, displays a nonlinear ETA estimate.

    Usage::

        prog = SingleTaskProgress()
        prog.phase("building")
        ...
        prog.phase("tuning")
        ...
        prog.phase("thermalizing", total=200)
        # use prog.on_tick as a callback
        cdt.sweep(200, progress=prog.on_tick)
        prog.phase("rendering")
        ...
        prog.finish("done")
    """

    def __init__(self, use_color=True, memory_monitor=None):
        self._lock = threading.Lock()
        self.use_color = use_color and sys.stderr.isatty()
        self._memory_monitor = memory_monitor
        self._phase = ""
        self._count = 0
        self._total = 0
        self._extra = ""
        self._spin_idx = 0
        self._start = time.time()
        self._lines_drawn = 0
        self._active = True
        self._eta = ETAEstimator()
        self._sparkline = _Sparkline()

        self._echo_suppressor = _StdinEchoSuppressor()
        self._echo_suppressor.enable()

        self._spinner_thread = threading.Thread(
            target=self._spin_loop, daemon=True)
        self._spinner_thread.start()

    def phase(self, name, *, total=0, extra=""):
        with self._lock:
            self._phase = name
            self._count = 0
            self._total = total
            self._extra = extra
            self._eta.reset()
            self._sparkline.reset()

    def on_tick(self, _i=None, _n=None):
        with self._lock:
            self._count += 1
            self._sparkline.record()
            if self._total > 0:
                self._eta.update(self._count, self._total)

    def finish(self, message="done"):
        self._active = False
        self._spinner_thread.join(timeout=1)
        with self._lock:
            self._phase = message
            self._draw(final=True)
            sys.stderr.write("\n")
            sys.stderr.flush()
        self._echo_suppressor.disable()

    def _spin_loop(self):
        while self._active:
            time.sleep(0.08)
            with self._lock:
                self._spin_idx = (self._spin_idx + 1) % len(_SPINNER)
                self._draw()

    def _draw(self, final=False):
        c = self.use_color
        elapsed = time.time() - self._start
        phase = self._phase
        emoji, color = PHASE_STYLE.get(phase, ("⏳", _WHITE))
        spin = _SPINNER[self._spin_idx] if c else "|"

        counter = ""
        if self._total > 0:
            counter = f" {_mini_bar(self._count, self._total, width=10, color=c)}"
        elif self._count > 0:
            counter = f" {self._count}"

        extra = f"  {self._extra}" if self._extra else ""

        eta_str = self._eta.format_compact()
        if eta_str and not final:
            eta_str = f"  {eta_str}"
        else:
            eta_str = ""

        mem_str = ""
        if self._memory_monitor is not None:
            rss = self._memory_monitor.rss_mb
            if rss is not None:
                mem_str = f"  💾 {rss} MB" if c else f"  RSS {rss} MB"

        if final:
            emoji, color = PHASE_STYLE.get("done", ("✅", _GREEN))

        if c:
            if final:
                indicator = emoji
            else:
                indicator = f"{color}{spin}{_RST} {emoji}"
            line = (
                f"{_ERASE_LINE}{indicator} "
                f"{color}{phase}{_RST}"
                f"{_DIM}{counter}{extra}  ⏱ {elapsed:.1f}s{_RST}"
                f"{_DIM}{eta_str}{_RST}"
                f"{_DIM}{mem_str}{_RST}"
            )
        else:
            line = (
                f"{_ERASE_LINE}{emoji} {phase}"
                f"{counter}{extra}  {elapsed:.1f}s"
                f"{eta_str}"
                f"{mem_str}"
            )

        lines = [line]
        spark_str = self._sparkline.render(width=30, color=c)
        if spark_str and not final:
            lines.append(f"{_ERASE_LINE}  {spark_str}")

        # Pad with blank lines if we previously drew more (e.g. sparkline
        # disappeared on phase change) so the old content is erased.
        while len(lines) < self._lines_drawn:
            lines.append(_ERASE_LINE)

        if self._lines_drawn > 0:
            prefix = f"\033[{self._lines_drawn}A"
        else:
            prefix = ""
        self._lines_drawn = len(lines)
        sys.stderr.write(f"{prefix}" + "\n".join(lines) + "\n")
        sys.stderr.flush()
