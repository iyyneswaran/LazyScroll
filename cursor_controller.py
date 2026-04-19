"""
cursor_controller.py — Maps index-finger-tip position from camera space
to screen coordinates and drives the OS mouse pointer.

Coordinate mapping strategy
---------------------------
The camera frame (normalised 0–1) is divided into an "active zone" — a
sub-rectangle centred in the frame.  Only hand positions inside this zone
map to the full screen.  Positions outside are clamped to the nearest
screen edge.

    Camera frame:
    ┌──────────────────────────────┐
    │                              │
    │   ┌────────────────────┐     │
    │   │   Active Zone      │     │
    │   │  x: 0.15 → 0.85   │     │
    │   │  y: 0.10 → 0.80   │     │
    │   └────────────────────┘     │
    │                              │
    └──────────────────────────────┘

This approach lets the user control the entire screen with comfortable
wrist-range movements instead of requiring large arm sweeps.

Smoothing pipeline
------------------
    raw landmark → 1€ filter (jitter removal) → screen mapping →
    EMA (trajectory smoothing) → dead-zone gate → pynput position set
"""

import time
import math
from utils import OneEuroFilter, ExponentialMovingAverage, get_screen_resolution

try:
    from pynput.mouse import Controller as MouseController
    _HAS_PYNPUT = True
except ImportError:
    _HAS_PYNPUT = False


class CursorController:
    """
    Translates normalised hand-landmark coordinates into screen-space
    mouse positions with multi-stage smoothing.

    Parameters
    ----------
    active_zone : tuple[float,float,float,float]
        (x_min, x_max, y_min, y_max) in normalised camera coords [0-1].
        Hand positions inside this rectangle map to the full screen.
    smoothing_alpha : float
        EMA alpha for final screen-coordinate smoothing (0 = max smooth,
        1 = no smoothing).  Default 0.35 gives a good balance.
    dead_zone_px : float
        Minimum cursor displacement (pixels) to trigger a move.  Prevents
        micro-tremor from causing tiny jittery cursor jumps.
    velocity_damp_threshold : float
        Normalised landmark velocity below which cursor speed is halved
        for fine-positioning mode.
    """

    def __init__(
        self,
        active_zone: tuple[float, float, float, float] = (0.15, 0.85, 0.10, 0.80),
        smoothing_alpha: float = 0.35,
        dead_zone_px: float = 3.0,
        velocity_damp_threshold: float = 0.005,
    ):
        # Screen dimensions
        self.screen_w, self.screen_h = get_screen_resolution()

        # Active zone boundaries (normalised camera coords)
        self.az_x_min, self.az_x_max, self.az_y_min, self.az_y_max = active_zone
        self.az_w = self.az_x_max - self.az_x_min
        self.az_h = self.az_y_max - self.az_y_min

        # Smoothing filters for raw landmark x, y
        self._filter_x = OneEuroFilter(freq=30.0, min_cutoff=1.5, beta=0.05, d_cutoff=1.0)
        self._filter_y = OneEuroFilter(freq=30.0, min_cutoff=1.5, beta=0.05, d_cutoff=1.0)

        # EMA for final screen coordinates
        self._ema_x = ExponentialMovingAverage(alpha=smoothing_alpha)
        self._ema_y = ExponentialMovingAverage(alpha=smoothing_alpha)

        # Dead zone and velocity damping
        self._dead_zone_px = dead_zone_px
        self._velocity_damp_threshold = velocity_damp_threshold

        # State
        self._last_screen_x: float | None = None
        self._last_screen_y: float | None = None
        self._prev_raw_x: float | None = None
        self._prev_raw_y: float | None = None
        self._prev_t: float | None = None

        # Exposed for debug overlay
        self.screen_x: float = 0.0
        self.screen_y: float = 0.0
        self.raw_norm_x: float = 0.0
        self.raw_norm_y: float = 0.0

        # Mouse controller
        if _HAS_PYNPUT:
            self._mouse = MouseController()
        else:
            self._mouse = None

    def _clamp(self, val: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, val))

    def _map_to_screen(self, norm_x: float, norm_y: float) -> tuple[float, float]:
        """
        Map normalised camera coordinate to screen pixel coordinate.

        The camera frame is mirrored (cv2.flip), so norm_x=0 is the right
        side of the real world.  We do NOT re-flip here because the main
        loop already mirrors the frame, making norm_x=0 correspond to the
        user's right (screen right when looking at the preview).

        Steps:
            1. Clamp to active zone
            2. Normalise within active zone → [0, 1]
            3. Scale to screen resolution
        """
        # Clamp into active zone
        cx = self._clamp(norm_x, self.az_x_min, self.az_x_max)
        cy = self._clamp(norm_y, self.az_y_min, self.az_y_max)

        # Normalise within the active zone → [0, 1]
        fx = (cx - self.az_x_min) / self.az_w
        fy = (cy - self.az_y_min) / self.az_h

        # Scale to screen
        sx = fx * self.screen_w
        sy = fy * self.screen_h

        return sx, sy

    def update(self, raw_x: float, raw_y: float) -> tuple[float, float]:
        """
        Process a new raw landmark position and move the cursor.

        Parameters
        ----------
        raw_x, raw_y : float
            Normalised [0, 1] index-finger-tip position from MediaPipe
            (already in mirrored camera space).

        Returns
        -------
        (screen_x, screen_y) : tuple[float, float]
            The resulting screen-space cursor position.
        """
        now = time.perf_counter()

        # Store raw values for debug
        self.raw_norm_x = raw_x
        self.raw_norm_y = raw_y

        # ── Stage 1: 1€ filter on raw landmarks ──
        filt_x = self._filter_x.update(raw_x, now)
        filt_y = self._filter_y.update(raw_y, now)

        # ── Stage 2: Map to screen coordinates ──
        sx, sy = self._map_to_screen(filt_x, filt_y)

        # ── Stage 3: Velocity damping for fine control ──
        if self._prev_raw_x is not None and self._prev_t is not None:
            dt = now - self._prev_t
            if dt > 0:
                vx = abs(filt_x - self._prev_raw_x) / dt
                vy = abs(filt_y - self._prev_raw_y) / dt
                speed = math.sqrt(vx ** 2 + vy ** 2)

                if speed < self._velocity_damp_threshold:
                    # Slow movement → increase smoothing for precision
                    self._ema_x.alpha = 0.20
                    self._ema_y.alpha = 0.20
                else:
                    # Normal / fast movement → default smoothing
                    self._ema_x.alpha = 0.35
                    self._ema_y.alpha = 0.35

        self._prev_raw_x = filt_x
        self._prev_raw_y = filt_y
        self._prev_t = now

        # ── Stage 4: EMA on screen coordinates ──
        smooth_x = self._ema_x.update(sx)
        smooth_y = self._ema_y.update(sy)

        # ── Stage 5: Dead-zone gating ──
        if self._last_screen_x is not None:
            dx = abs(smooth_x - self._last_screen_x)
            dy = abs(smooth_y - self._last_screen_y)
            if dx < self._dead_zone_px and dy < self._dead_zone_px:
                # Movement too small — hold position
                self.screen_x = self._last_screen_x
                self.screen_y = self._last_screen_y
                return self.screen_x, self.screen_y

        # ── Stage 6: Clamp to screen and move ──
        final_x = self._clamp(smooth_x, 0, self.screen_w - 1)
        final_y = self._clamp(smooth_y, 0, self.screen_h - 1)

        self._last_screen_x = final_x
        self._last_screen_y = final_y
        self.screen_x = final_x
        self.screen_y = final_y

        # Actually move the OS cursor
        if self._mouse is not None:
            self._mouse.position = (int(final_x), int(final_y))

        return final_x, final_y

    def reset(self):
        """Reset filters and state when hand is lost or mode switches."""
        self._filter_x.reset()
        self._filter_y.reset()
        self._ema_x.reset()
        self._ema_y.reset()
        self._last_screen_x = None
        self._last_screen_y = None
        self._prev_raw_x = None
        self._prev_raw_y = None
        self._prev_t = None

    def set_active_zone(self, x_min: float, x_max: float,
                        y_min: float, y_max: float):
        """Update the active zone (e.g. after calibration)."""
        self.az_x_min = x_min
        self.az_x_max = x_max
        self.az_y_min = y_min
        self.az_y_max = y_max
        self.az_w = x_max - x_min
        self.az_h = y_max - y_min
