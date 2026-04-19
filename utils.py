"""
utils.py — Shared utilities: smoothing filters, FPS counter, calibration store, and drawing helpers.
"""

import time
import collections
import numpy as np


class ExponentialMovingAverage:
    """EMA filter for 1-D signals. Lower alpha = more smoothing."""

    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha
        self._value: float | None = None

    def update(self, raw: float) -> float:
        if self._value is None:
            self._value = raw
        else:
            self._value = self.alpha * raw + (1.0 - self.alpha) * self._value
        return self._value

    def reset(self):
        self._value = None

    @property
    def value(self) -> float:
        return self._value if self._value is not None else 0.0


class OneEuroFilter:
    """
    1€ filter — adaptive low-pass that reduces jitter at low speed
    while preserving responsiveness at high speed.
    Ref: Casiez et al., CHI 2012.
    """

    def __init__(self, freq: float = 30.0, min_cutoff: float = 1.0,
                 beta: float = 0.007, d_cutoff: float = 1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    @staticmethod
    def _smoothing_factor(te: float, cutoff: float) -> float:
        r = 2.0 * np.pi * cutoff * te
        return r / (r + 1.0)

    def update(self, x: float, t: float | None = None) -> float:
        if t is None:
            t = time.perf_counter()

        if self._t_prev is None:
            self._x_prev = x
            self._dx_prev = 0.0
            self._t_prev = t
            return x

        te = t - self._t_prev
        if te <= 0:
            te = 1.0 / self.freq

        # Derivative estimation
        a_d = self._smoothing_factor(te, self.d_cutoff)
        dx = (x - self._x_prev) / te
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._smoothing_factor(te, cutoff)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat

    def reset(self):
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None


class FPSCounter:
    """Rolling-window FPS counter."""

    def __init__(self, window: int = 30):
        self._times: collections.deque[float] = collections.deque(maxlen=window)

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


class CalibrationStore:
    """
    In-memory parameter store populated during an optional calibration phase.
    Stores sensitivity, dead-zone, and scaling factors per user session.
    """

    def __init__(self):
        self.sensitivity: float = 1.0          # global multiplier
        self.dead_zone: float = 0.005          # normalised Δy below which motion is ignored
        self.slow_threshold: float = 0.02      # boundary between fine and coarse scroll
        self.fast_multiplier: float = 3.0      # acceleration factor for fast movements
        self.scroll_gain: float = 5.0          # base scroll-lines per normalised Δy
        self.calibrated: bool = False
        # Raw samples collected during calibration
        self._samples: list[float] = []

    def record_sample(self, dy: float):
        """Collect absolute Δy samples during calibration."""
        self._samples.append(abs(dy))

    def finalise(self):
        """Derive parameters from collected samples."""
        if len(self._samples) < 10:
            return  # not enough data
        arr = np.array(self._samples)
        # Dead zone = 25th percentile of movement
        self.dead_zone = float(np.percentile(arr, 25))
        # Slow threshold = median
        self.slow_threshold = float(np.percentile(arr, 50))
        # Sensitivity from range
        p90 = float(np.percentile(arr, 90))
        if p90 > 0:
            self.sensitivity = 1.0 / p90
        self.calibrated = True
        self._samples.clear()

    def to_dict(self) -> dict:
        return {
            "sensitivity": self.sensitivity,
            "dead_zone": self.dead_zone,
            "slow_threshold": self.slow_threshold,
            "fast_multiplier": self.fast_multiplier,
            "scroll_gain": self.scroll_gain,
            "calibrated": self.calibrated,
        }


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation clamped to [a, b]."""
    return a + (b - a) * max(0.0, min(1.0, t))
