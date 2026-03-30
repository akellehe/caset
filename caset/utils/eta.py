"""Nonlinear ETA estimator for long-running processes.

Records (elapsed_time, fraction_done) samples and fits a power-law model
    t(f) = a * f^b + c
to predict the total time.  Falls back to linear extrapolation when the
fit is unreliable (too few samples, poor conditioning, or the power-law
predicts an unreasonable result).

The estimator is deliberately kept dependency-free (no numpy/scipy at
runtime) so it can be imported from any example without pulling in heavy
libraries.  The fitting uses a Gauss-Newton solver on the log-linearized
power-law.

Usage::

    from eta import ETAEstimator

    est = ETAEstimator()
    for i in range(n):
        do_work()
        est.update(i + 1, n)          # records (elapsed, fraction)
        print(est.format_eta())        # "ETA 1m 23s" or "ETA --:--"

Thread-safe: all public methods acquire an internal lock.
"""
from __future__ import annotations

import math
import threading
import time


def _fmt_duration(seconds: float) -> str:
    """Human-readable duration: '1h 23m', '4m 56s', '12s', '<1s'."""
    if not math.isfinite(seconds) or seconds < 0:
        return "--:--"
    if seconds < 1:
        return "<1s"
    s = int(seconds)
    if s >= 3600:
        h, m = divmod(s, 3600)
        m //= 60
        return f"{h}h {m:02d}m"
    if s >= 60:
        m, sec = divmod(s, 60)
        return f"{m}m {sec:02d}s"
    return f"{s}s"


class ETAEstimator:
    """Nonlinear ETA estimator using power-law regression.

    Model:  elapsed(f) = a * f^b + c

    where f is the fraction complete [0, 1].  A purely linear process
    has b=1; sub-linear (slowing down) has b>1; super-linear (speeding
    up) has b<1.

    Parameters
    ----------
    min_samples : int
        Minimum number of samples before attempting the nonlinear fit.
        Below this threshold, linear extrapolation is used.
    sample_interval : float
        Minimum seconds between recorded samples (prevents flooding
        from high-frequency callbacks).
    """

    def __init__(self, *, min_samples: int = 5, sample_interval: float = 0.1):
        self._lock = threading.Lock()
        self._start = time.monotonic()
        self._samples: list[tuple[float, float]] = []  # (elapsed, fraction)
        self._min_samples = min_samples
        self._sample_interval = sample_interval
        self._last_sample_time = -math.inf
        # Cached prediction
        self._eta_seconds: float = math.nan
        self._total_predicted: float = math.nan
        self._fit_exponent: float = 1.0

    def reset(self) -> None:
        """Clear all samples and restart the clock."""
        with self._lock:
            self._start = time.monotonic()
            self._samples.clear()
            self._last_sample_time = -math.inf
            self._eta_seconds = math.nan
            self._total_predicted = math.nan
            self._fit_exponent = 1.0

    def update(self, done: int | float, total: int | float) -> None:
        """Record a progress sample.

        Parameters
        ----------
        done  : number of completed units (e.g. sweep index)
        total : total number of units (e.g. total sweeps)
        """
        if total <= 0:
            return
        now = time.monotonic()
        elapsed = now - self._start
        frac = min(float(done) / float(total), 1.0)

        with self._lock:
            if elapsed - self._last_sample_time < self._sample_interval:
                return
            self._last_sample_time = elapsed
            self._samples.append((elapsed, frac))
            self._refit()

    @property
    def eta_seconds(self) -> float:
        """Estimated seconds remaining (NaN if unknown)."""
        with self._lock:
            return self._eta_seconds

    @property
    def elapsed(self) -> float:
        """Seconds since construction or last reset."""
        return time.monotonic() - self._start

    @property
    def exponent(self) -> float:
        """Current fit exponent b (1.0 = linear)."""
        with self._lock:
            return self._fit_exponent

    def format_eta(self) -> str:
        """Human-readable ETA string, e.g. 'ETA 2m 15s' or 'ETA --:--'."""
        eta = self.eta_seconds
        if math.isnan(eta):
            return "ETA --:--"
        return f"ETA {_fmt_duration(eta)}"

    def format_compact(self) -> str:
        """Compact string for embedding in progress lines: '⏳ 2m 15s' or ''."""
        eta = self.eta_seconds
        if math.isnan(eta):
            return ""
        return f"⏳{_fmt_duration(eta)}"

    # ── internals ──

    def _refit(self) -> None:
        """Recompute ETA from accumulated samples. Must hold self._lock."""
        samples = self._samples
        n = len(samples)
        elapsed_now = samples[-1][0]
        frac_now = samples[-1][1]

        if frac_now <= 0:
            return
        if frac_now >= 1.0:
            self._eta_seconds = 0.0
            return

        # Always compute linear estimate as fallback
        linear_total = elapsed_now / frac_now
        linear_eta = linear_total - elapsed_now

        if n < self._min_samples:
            self._eta_seconds = max(linear_eta, 0.0)
            self._total_predicted = linear_total
            self._fit_exponent = 1.0
            return

        # Attempt power-law fit: t(f) = a * f^b
        # Log-linearize: log(t) = log(a) + b * log(f)
        # Use least-squares on samples where both t > 0 and f > 0.
        log_t = []
        log_f = []
        for t, f in samples:
            if t > 0 and f > 0:
                log_t.append(math.log(t))
                log_f.append(math.log(f))

        if len(log_t) < self._min_samples:
            self._eta_seconds = max(linear_eta, 0.0)
            self._total_predicted = linear_total
            self._fit_exponent = 1.0
            return

        # Least-squares: log(t) = b * log(f) + log(a)
        #   [sum(x^2)  sum(x)] [b   ]   [sum(x*y)]
        #   [sum(x)    n     ] [ln_a] = [sum(y)  ]
        n_pts = len(log_t)
        sx = sum(log_f)
        sy = sum(log_t)
        sxx = sum(x * x for x in log_f)
        sxy = sum(x * y for x, y in zip(log_f, log_t))

        det = sxx * n_pts - sx * sx
        if abs(det) < 1e-15:
            self._eta_seconds = max(linear_eta, 0.0)
            self._total_predicted = linear_total
            self._fit_exponent = 1.0
            return

        b = (sxy * n_pts - sx * sy) / det
        ln_a = (sxx * sy - sx * sxy) / det
        a = math.exp(ln_a)

        # Sanity checks: reject degenerate fits
        #   - b should be positive (time increases with progress)
        #   - b shouldn't be extreme (< 0.1 or > 10 means poor fit)
        #   - predicted total shouldn't be absurd (< 0.5x or > 20x linear)
        if b <= 0.1 or b > 10:
            self._eta_seconds = max(linear_eta, 0.0)
            self._total_predicted = linear_total
            self._fit_exponent = 1.0
            return

        predicted_total = a  # t(1.0) = a * 1^b = a
        if predicted_total < 0.5 * linear_total or predicted_total > 20 * linear_total:
            self._eta_seconds = max(linear_eta, 0.0)
            self._total_predicted = linear_total
            self._fit_exponent = 1.0
            return

        self._fit_exponent = b
        self._total_predicted = predicted_total
        self._eta_seconds = max(predicted_total - elapsed_now, 0.0)
