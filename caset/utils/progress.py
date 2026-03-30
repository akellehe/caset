"""Shared terminal progress display for caset examples.

Provides two display classes:

  ProgressDisplay   – thread-safe multi-line display for parallel workers
                      (configs, grid points, chains, etc.)
  SingleTaskProgress – single-line spinner for sequential scripts
                      (build → tune → thermalize → solve → render)

Both use ANSI colors and emoji when connected to a TTY, and degrade
gracefully to plain text otherwise.  Both show a nonlinear ETA estimate
that adapts to sub-linear and super-linear progress curves.
"""
import sys
import threading
import time

from caset.utils.eta import ETAEstimator

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

    # ── rendering ──

    def _spin_loop(self):
        while self._active:
            time.sleep(0.08)
            with self._lock:
                self._spin_idx = (self._spin_idx + 1) % len(_SPINNER)
                self._draw()

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
                prog_str = f" {d}/{t} ({d/t*100:.0f}%)" if t else ""

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
        self._drawn = False
        self._active = True
        self._eta = ETAEstimator()

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

    def on_tick(self, _i=None, _n=None):
        with self._lock:
            self._count += 1
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
            pct = self._count / self._total * 100
            counter = f" {self._count}/{self._total} ({pct:.0f}%)"
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

        prefix = "\033[1A" if self._drawn else ""
        sys.stderr.write(f"{prefix}{line}\n")
        sys.stderr.flush()
        self._drawn = True
