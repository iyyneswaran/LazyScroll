"""
gesture_detector.py — Classifies hand gestures from landmark geometry.

Gesture modes:
    "SCROLL"   — index + middle fingers extended, ring + pinky folded  → scroll control
    "CURSOR"   — index finger only extended, others folded             → cursor control
    "CLICK"    — pinch detected while in cursor mode (thumb↔index)     → mouse click
    "FIST"     — all fingers folded                                    → idle
    "OPEN_PALM" — all fingers extended                                 → idle

The CLICK gesture is a *sub-state* of CURSOR: it is only detected when
the base gesture is CURSOR and the pinch distance drops below threshold.
This prevents accidental clicks during scrolling.
"""

import math
from hand_tracker import HandResult, HandTracker


# ---------- Gesture labels ----------
GESTURE_NONE = "NONE"
GESTURE_SCROLL = "SCROLL"        # index + middle up, others down
GESTURE_CURSOR = "CURSOR"        # index up only, others down
GESTURE_CLICK = "CLICK"          # pinch (thumb + index close) while in cursor mode
GESTURE_FIST = "FIST"            # all fingers folded
GESTURE_OPEN = "OPEN_PALM"       # all fingers extended


class GestureDetector:
    """
    Geometric gesture classifier using inter-landmark distances and angles.
    No ML needed — pure landmark math for sub-millisecond classification.

    Finger extension heuristic:
        A finger is considered *extended* if its tip is farther from the wrist
        than its PIP (proximal interphalangeal) joint, measured in Euclidean
        distance.  For the thumb, we compare tip-to-wrist vs. IP-to-wrist.

    This is more robust than simple y-coordinate comparison because it
    handles hand rotations and non-upright orientations.
    """

    def __init__(self, extension_ratio: float = 1.15,
                 pinch_threshold: float = 0.045):
        """
        extension_ratio: tip_dist / pip_dist must exceed this for
        a finger to be classified as extended.  Tune up for stricter
        detection, down for more lenient.

        pinch_threshold: normalised 3D distance between thumb tip and
        index tip below which a pinch is detected.
        """
        self.extension_ratio = extension_ratio
        self.pinch_threshold = pinch_threshold

        # Debounce: require N consecutive identical classifications
        self._history: list[str] = []
        self._debounce_frames: int = 3
        self._current_gesture: str = GESTURE_NONE

        # Exposed landmark data for downstream modules
        self._pinch_distance: float = 1.0
        self._finger_states: list[bool] = [False] * 5  # [thumb, index, middle, ring, pinky]

    @staticmethod
    def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    def _is_finger_extended(self, landmarks: list[tuple[float, float, float]],
                            tip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
        """Check if a finger is extended using distance-from-wrist ratio."""
        wrist = landmarks[HandTracker.WRIST]
        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]

        tip_dist = self._dist(tip, wrist)
        pip_dist = self._dist(pip, wrist)

        if pip_dist < 1e-6:
            return False
        return (tip_dist / pip_dist) > self.extension_ratio

    def _is_thumb_extended(self, landmarks: list[tuple[float, float, float]]) -> bool:
        """
        Thumb uses a lateral spread check: angle between thumb-tip → wrist
        and index-mcp → wrist vectors.
        """
        wrist = landmarks[HandTracker.WRIST]
        thumb_tip = landmarks[HandTracker.THUMB_TIP]
        thumb_ip = landmarks[HandTracker.THUMB_IP]
        index_mcp = landmarks[HandTracker.INDEX_MCP]

        # Distance-based: thumb tip should be farther from index MCP than thumb IP is
        tip_to_idx = self._dist(thumb_tip, index_mcp)
        ip_to_idx = self._dist(thumb_ip, index_mcp)

        if ip_to_idx < 1e-6:
            return False
        return (tip_to_idx / ip_to_idx) > self.extension_ratio

    def classify(self, hand: HandResult) -> str:
        """
        Classify gesture from a single hand result.
        Returns one of GESTURE_NONE, GESTURE_SCROLL, GESTURE_CURSOR,
        GESTURE_FIST, GESTURE_OPEN.

        Note: GESTURE_CLICK is NOT returned here — pinch detection is
        handled separately by ClickDetector to allow hysteresis and
        cooldown logic.  Use `pinch_distance` property to feed
        the ClickDetector.
        """
        lms = hand.landmarks

        thumb_ext = self._is_thumb_extended(lms)
        index_ext = self._is_finger_extended(lms, HandTracker.INDEX_TIP,
                                             HandTracker.INDEX_PIP, HandTracker.INDEX_MCP)
        middle_ext = self._is_finger_extended(lms, HandTracker.MIDDLE_TIP,
                                              HandTracker.MIDDLE_PIP, HandTracker.MIDDLE_MCP)
        ring_ext = self._is_finger_extended(lms, HandTracker.RING_TIP,
                                            HandTracker.RING_PIP, HandTracker.RING_MCP)
        pinky_ext = self._is_finger_extended(lms, HandTracker.PINKY_TIP,
                                             HandTracker.PINKY_PIP, HandTracker.PINKY_MCP)

        self._finger_states = [thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext]
        num_extended = sum(self._finger_states)

        # Compute pinch distance (always, for ClickDetector)
        self._pinch_distance = self._dist(
            lms[HandTracker.THUMB_TIP], lms[HandTracker.INDEX_TIP]
        )

        # --- Scroll gesture: index + middle extended, ring + pinky folded ---
        if index_ext and middle_ext and not ring_ext and not pinky_ext:
            raw = GESTURE_SCROLL
        # --- Cursor gesture: index only extended, others folded ---
        elif index_ext and not middle_ext and not ring_ext and not pinky_ext:
            raw = GESTURE_CURSOR
        elif num_extended == 0:
            raw = GESTURE_FIST
        elif num_extended >= 4:
            raw = GESTURE_OPEN
        else:
            raw = GESTURE_NONE

        # --- Temporal debounce ---
        self._history.append(raw)
        if len(self._history) > self._debounce_frames:
            self._history.pop(0)

        # Only change gesture if all recent frames agree
        if len(self._history) == self._debounce_frames and len(set(self._history)) == 1:
            self._current_gesture = raw

        return self._current_gesture

    @property
    def is_scroll_active(self) -> bool:
        return self._current_gesture == GESTURE_SCROLL

    @property
    def is_cursor_active(self) -> bool:
        return self._current_gesture == GESTURE_CURSOR

    @property
    def pinch_distance(self) -> float:
        """Normalised 3D distance between thumb tip and index tip."""
        return self._pinch_distance

    @property
    def finger_states(self) -> list[bool]:
        """[thumb, index, middle, ring, pinky] extension states."""
        return self._finger_states
