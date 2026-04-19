"""
click_detector.py — Detects pinch gestures and triggers system-level
mouse click events.

Click gesture detection logic
------------------------------
Uses the Euclidean distance between thumb tip (landmark 4) and index
finger tip (landmark 8) in normalised 3D coordinates.

State machine (with hysteresis):

    IDLE ──(dist < pinch_threshold)──► PINCHING
    PINCHING ──(internal)──► CLICKED  (fires click event)
    CLICKED ──(cooldown elapsed)──► COOLDOWN_DONE
    COOLDOWN_DONE ──(dist > release_threshold)──► IDLE

Hysteresis band:
    pinch_threshold  = 0.045  (engage)
    release_threshold = 0.065  (disengage)

This 0.02 gap prevents oscillation when the finger distance hovers
near the boundary.

Double-click detection:
    If two CLICKED events occur within 400ms, a double-click is emitted
    instead of two single clicks.
"""

import math
import time

try:
    from pynput.mouse import Controller as MouseController, Button
    _HAS_PYNPUT = True
except ImportError:
    _HAS_PYNPUT = False


# Click states
STATE_IDLE = "IDLE"
STATE_PINCHING = "PINCHING"
STATE_CLICKED = "CLICKED"
STATE_COOLDOWN = "COOLDOWN"


class ClickDetector:
    """
    Pinch-based click detector with hysteresis, cooldown, and
    double-click support.

    Parameters
    ----------
    pinch_threshold : float
        Normalised landmark distance below which a pinch is detected.
    release_threshold : float
        Distance above which the pinch is considered released.
        Must be > pinch_threshold for hysteresis.
    cooldown_sec : float
        Minimum seconds between consecutive clicks.
    double_click_window : float
        Maximum seconds between two clicks to register as double-click.
    """

    def __init__(
        self,
        pinch_threshold: float = 0.045,
        release_threshold: float = 0.065,
        cooldown_sec: float = 0.25,
        double_click_window: float = 0.40,
    ):
        self.pinch_threshold = pinch_threshold
        self.release_threshold = release_threshold
        self.cooldown_sec = cooldown_sec
        self.double_click_window = double_click_window

        # State machine
        self._state: str = STATE_IDLE
        self._last_click_time: float = 0.0
        self._pending_single_click: bool = False
        self._pending_single_time: float = 0.0

        # Exposed for debug / overlay
        self.pinch_distance: float = 1.0
        self.state: str = STATE_IDLE
        self.last_event: str = ""          # "CLICK", "DOUBLE_CLICK", ""
        self._last_event_time: float = 0.0

        # Mouse controller
        if _HAS_PYNPUT:
            self._mouse = MouseController()
            self._button = Button.left
        else:
            self._mouse = None
            self._button = None

    @staticmethod
    def _dist_3d(a: tuple[float, float, float],
                 b: tuple[float, float, float]) -> float:
        """Euclidean distance in normalised 3D landmark space."""
        return math.sqrt(
            (a[0] - b[0]) ** 2 +
            (a[1] - b[1]) ** 2 +
            (a[2] - b[2]) ** 2
        )

    def update(self, thumb_tip: tuple[float, float, float],
               index_tip: tuple[float, float, float],
               is_cursor_mode: bool = True) -> str:
        """
        Process one frame of landmark data.

        Parameters
        ----------
        thumb_tip : landmark 4 (x, y, z) normalised
        index_tip : landmark 8 (x, y, z) normalised
        is_cursor_mode : bool
            Only allow clicks when in cursor mode to prevent
            accidental clicks during scroll.

        Returns
        -------
        event : str
            "CLICK", "DOUBLE_CLICK", or "" (no event).
        """
        now = time.perf_counter()
        dist = self._dist_3d(thumb_tip, index_tip)
        self.pinch_distance = dist
        event = ""

        # Clear stale event display after 300ms
        if now - self._last_event_time > 0.3:
            self.last_event = ""

        if not is_cursor_mode:
            # Not in cursor mode — stay idle, don't detect clicks
            self._state = STATE_IDLE
            self.state = self._state
            return ""

        # ── State machine ──
        if self._state == STATE_IDLE:
            if dist < self.pinch_threshold:
                self._state = STATE_PINCHING

        elif self._state == STATE_PINCHING:
            if dist < self.pinch_threshold:
                # Still pinching — fire the click
                event = self._fire_click(now)
                self._state = STATE_CLICKED
            else:
                # Released before fully pinching — back to idle
                self._state = STATE_IDLE

        elif self._state == STATE_CLICKED:
            # Wait for cooldown
            if now - self._last_click_time >= self.cooldown_sec:
                self._state = STATE_COOLDOWN

        elif self._state == STATE_COOLDOWN:
            # Wait for release (hysteresis)
            if dist > self.release_threshold:
                self._state = STATE_IDLE

        self.state = self._state
        return event

    def _fire_click(self, now: float) -> str:
        """Execute the click and handle double-click detection."""
        time_since_last = now - self._last_click_time
        self._last_click_time = now

        if time_since_last < self.double_click_window and self._pending_single_click:
            # Double click
            self._pending_single_click = False
            if self._mouse is not None:
                self._mouse.click(self._button, 2)
            self.last_event = "DOUBLE_CLICK"
            self._last_event_time = now
            return "DOUBLE_CLICK"
        else:
            # Single click
            self._pending_single_click = True
            self._pending_single_time = now
            if self._mouse is not None:
                self._mouse.click(self._button, 1)
            self.last_event = "CLICK"
            self._last_event_time = now
            return "CLICK"

    def reset(self):
        """Reset state when hand is lost or mode switches."""
        self._state = STATE_IDLE
        self.state = STATE_IDLE
        self.pinch_distance = 1.0
        self._pending_single_click = False
