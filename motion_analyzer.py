"""
motion_analyzer.py — Computes vertical displacement of the index finger tip,
applies filtering and dead-zone gating, and outputs a scroll velocity signal.
"""

import time
from utils import OneEuroFilter, ExponentialMovingAverage, CalibrationStore
from hand_tracker import HandResult, HandTracker


class MotionAnalyzer:
    """
    Pipeline:
        1. Extract index-finger-tip y coordinate (normalised 0–1).
        2. Compute Δy between consecutive frames.
        3. Pass through 1€ filter to remove jitter but preserve responsiveness.
        4. Apply dead-zone gating.
        5. Map to scroll velocity with dynamic (non-linear) scaling.
    """

    def __init__(self, calibration: CalibrationStore | None = None):
        self.calibration = calibration or CalibrationStore()

        # 1€ filter for raw y-position (reduces jitter in landmark itself)
        self._pos_filter = OneEuroFilter(
            freq=30.0, min_cutoff=1.5, beta=0.01, d_cutoff=1.0
        )
        # EMA for the final scroll velocity (smooths output)
        self._vel_ema = ExponentialMovingAverage(alpha=0.4)

        self._prev_y: float | None = None
        self._prev_t: float | None = None

    def reset(self):
        """Reset state when hand is lost or gesture deactivates."""
        self._prev_y = None
        self._prev_t = None
        self._pos_filter.reset()
        self._vel_ema.reset()

    def update(self, hand: HandResult) -> float:
        """
        Accepts a HandResult, returns scroll velocity in abstract units.
        Positive = scroll down, negative = scroll up.
        Returns 0.0 when inside the dead zone.
        """
        raw_y = hand.landmarks[HandTracker.INDEX_TIP][1]  # normalised [0,1]

        now = time.perf_counter()
        filtered_y = self._pos_filter.update(raw_y, now)

        if self._prev_y is None:
            self._prev_y = filtered_y
            self._prev_t = now
            return 0.0

        dt = now - self._prev_t
        if dt <= 0:
            return 0.0

        dy = filtered_y - self._prev_y
        self._prev_y = filtered_y
        self._prev_t = now

        # --- Dead zone ---
        abs_dy = abs(dy)
        if abs_dy < self.calibration.dead_zone:
            # Drain velocity smoothly toward zero
            return self._vel_ema.update(0.0)

        # --- Dynamic (non-linear) scaling ---
        # Below slow_threshold → linear (fine control)
        # Above slow_threshold → quadratic ramp (acceleration)
        cal = self.calibration
        if abs_dy < cal.slow_threshold:
            magnitude = abs_dy * cal.scroll_gain * cal.sensitivity
        else:
            # Quadratic acceleration above threshold
            excess = abs_dy - cal.slow_threshold
            base = cal.slow_threshold * cal.scroll_gain * cal.sensitivity
            magnitude = base + (excess ** 1.6) * cal.fast_multiplier * cal.scroll_gain * cal.sensitivity

        velocity = magnitude if dy > 0 else -magnitude
        smoothed = self._vel_ema.update(velocity)
        return smoothed

    @property
    def finger_y(self) -> float:
        """Last filtered y position (for debug overlay)."""
        return self._prev_y if self._prev_y is not None else 0.0
