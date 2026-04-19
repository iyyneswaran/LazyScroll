"""
scroll_controller.py — Dispatches OS-level scroll events via pynput.
"""

import platform
import threading
import time

try:
    from pynput.mouse import Controller as MouseController
    _BACKEND = "pynput"
except ImportError:
    import pyautogui
    _BACKEND = "pyautogui"


class ScrollController:
    """
    Converts a continuous scroll-velocity signal into discrete OS scroll events.

    Key design decisions
    --------------------
    * Uses pynput.mouse.Controller.scroll() which generates real WM_MOUSEWHEEL
      (Windows), XTest Scroll (Linux), or CGEvent (macOS) events.
    * Accumulates fractional scroll amounts so sub-1-line velocities still
      produce scroll events over time.
    * Rate-limits to avoid flooding the event queue (max ~60 events/sec).
    * Thread-safe: can be called from any thread.

    Platform caveats
    ----------------
    * Windows: works out of the box.  UAC-elevated apps may ignore injected
      input unless this script is also elevated.
    * macOS: requires Accessibility permission (System Preferences → Security
      & Privacy → Privacy → Accessibility).
    * Linux/Wayland: pynput uses uinput; may require `sudo` or the user must
      be in the `input` group.  Works seamlessly on X11.
    """

    def __init__(self, invert: bool = False):
        """
        invert: if True, inverts the scroll direction (natural scrolling).
        """
        self._invert = invert
        self._accumulator: float = 0.0
        self._lock = threading.Lock()
        self._last_event_time: float = 0.0
        self._min_interval: float = 1.0 / 60.0  # max 60 events/sec

        if _BACKEND == "pynput":
            self._mouse = MouseController()
        else:
            self._mouse = None

        self._platform = platform.system()  # "Windows", "Darwin", "Linux"

    def scroll(self, velocity: float):
        """
        Accept a continuous velocity value and convert to discrete scroll ticks.
        velocity > 0 → scroll down,  velocity < 0 → scroll up.
        """
        if abs(velocity) < 0.01:
            # Below minimum threshold — drain accumulator
            with self._lock:
                self._accumulator *= 0.5
            return

        direction = -1.0 if self._invert else 1.0

        with self._lock:
            self._accumulator += velocity * direction

            # Only emit when we've accumulated at least 1 full tick
            ticks = int(self._accumulator)
            if ticks == 0:
                return

            # Rate-limit
            now = time.perf_counter()
            if (now - self._last_event_time) < self._min_interval:
                return

            self._accumulator -= ticks
            self._last_event_time = now

        # pynput scroll: (dx, dy) — positive dy = scroll UP on most platforms
        # We negate ticks so positive velocity → scroll DOWN (page moves up)
        scroll_dy = -ticks

        if _BACKEND == "pynput":
            self._mouse.scroll(0, scroll_dy)
        else:
            # pyautogui fallback — scroll() takes positive = up
            import pyautogui
            pyautogui.scroll(scroll_dy)

    def reset(self):
        with self._lock:
            self._accumulator = 0.0

    @property
    def backend(self) -> str:
        return _BACKEND
