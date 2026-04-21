"""
fusion_engine.py — Multimodal fusion of voice intent and gesture state.

Combines real-time gesture state (cursor position, scroll velocity, pinch
detection) with parsed voice intents to produce actionable FusedActions.

The engine enforces:
    * Temporal alignment: voice intents expire after a configurable window
    * Mode consistency: voice commands must match the current gesture mode
    * Conflict resolution: gesture-only actions always work; voice enhances
    * Drag state machine: voice "drag" holds, "drop"/"stop" releases

Threading: this module is called from the main loop (single-threaded).
It reads from the voice pipeline's output queues but does not start
its own threads.

Usage:
    fusion = FusionEngine(intent_parser, stt, llm_intent=None)
    ...
    # Each frame in main loop:
    action = fusion.process(gesture_state)
    if action:
        execute(action)
"""

import time
import logging
from dataclasses import dataclass, field
from enum import Enum

from intent_parser import IntentParser, VoiceIntent

logger = logging.getLogger(__name__)


# ─────────────────────── Data structures ───────────────────────


class ActionType(Enum):
    """Types of fused actions the engine can produce."""
    NONE = "none"
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    OPEN = "open"               # same as click, semantically
    SCROLL_ADJUST = "scroll_adjust"   # change scroll sensitivity
    SCROLL_DIRECTION = "scroll_direction"  # force scroll direction
    STOP_SCROLL = "stop_scroll"
    DRAG_START = "drag_start"
    DRAG_END = "drag_end"


@dataclass(slots=True)
class GestureState:
    """Snapshot of the current gesture pipeline state (fed each frame)."""
    gesture: str = "NONE"              # CURSOR, SCROLL, NONE, FIST, OPEN_PALM
    cursor_x: float = 0.0             # screen X
    cursor_y: float = 0.0             # screen Y
    cursor_stable: bool = False        # True if cursor barely moved for 0.3 s
    pinch_detected: bool = False       # True if pinch distance < threshold
    pinch_distance: float = 1.0        # raw pinch distance
    scroll_velocity: float = 0.0       # current scroll velocity
    hand_detected: bool = False        # True if any hand is visible
    timestamp: float = field(default_factory=time.perf_counter)


@dataclass(slots=True)
class FusedAction:
    """An action produced by the fusion engine for the main loop to execute."""
    action: ActionType = ActionType.NONE
    scroll_multiplier: float = 1.0     # for SCROLL_ADJUST
    scroll_direction: float = 0.0      # for SCROLL_DIRECTION (+1=down, -1=up)
    confidence: float = 0.0
    source: str = "none"               # "voice", "gesture", "fused"
    voice_text: str = ""               # original voice command
    voice_intent: str = ""             # parsed intent name
    timestamp: float = field(default_factory=time.perf_counter)

    @property
    def is_valid(self) -> bool:
        return self.action != ActionType.NONE


# ─────────────────────── Drag state machine ───────────────────────


class DragState(Enum):
    IDLE = "idle"
    DRAGGING = "dragging"


# ─────────────────────── Fusion Engine ───────────────────────


class FusionEngine:
    """
    Combines gesture state with voice intent to produce fused actions.

    Parameters
    ----------
    intent_parser : IntentParser
        The Tier 1 regex-based parser for fast intent extraction.
    stt : SpeechToText
        The speech-to-text engine to poll for recognised text.
    llm_intent : LLMIntent | None
        Optional Tier 2 LLM classifier.
    voice_window_sec : float
        Seconds a voice intent remains valid after recognition.
    cursor_stable_threshold_px : float
        Maximum cursor movement (pixels) over stability_window to
        consider the cursor "stable" (pointing at something).
    stability_window_sec : float
        Time window for cursor stability check.
    """

    def __init__(
        self,
        intent_parser: IntentParser,
        stt=None,          # SpeechToText (optional import)
        llm_intent=None,   # LLMIntent (optional import)
        voice_window_sec: float = 1.5,
        cursor_stable_threshold_px: float = 15.0,
        stability_window_sec: float = 0.3,
    ):
        self._parser = intent_parser
        self._stt = stt
        self._llm = llm_intent
        self._voice_window = voice_window_sec
        self._stable_threshold = cursor_stable_threshold_px
        self._stability_window = stability_window_sec

        # Internal state
        self._last_voice_intent: VoiceIntent | None = None
        self._drag_state = DragState.IDLE
        self._scroll_multiplier: float = 1.0

        # Cursor stability tracking
        self._cursor_history: list[tuple[float, float, float]] = []  # (x, y, time)

        # Feedback data (read by overlay)
        self.last_command_text: str = ""
        self.last_intent_name: str = ""
        self.last_confidence: float = 0.0
        self.last_action: str = ""
        self.voice_active: bool = False
        self.mode_label: str = "GESTURE ONLY"
        self.voice_latency_ms: float = 0.0

        # Scroll multiplier (adjusted by voice commands)
        self._base_scroll_multiplier: float = 1.0

    @property
    def scroll_multiplier(self) -> float:
        """Current scroll sensitivity multiplier."""
        return self._scroll_multiplier

    @property
    def drag_active(self) -> bool:
        """True if in drag mode."""
        return self._drag_state == DragState.DRAGGING

    def _is_cursor_stable(self, state: GestureState) -> bool:
        """Check if cursor has been stable for the stability window."""
        now = state.timestamp

        # Record current position
        self._cursor_history.append((state.cursor_x, state.cursor_y, now))

        # Trim old entries
        cutoff = now - self._stability_window
        self._cursor_history = [
            (x, y, t) for x, y, t in self._cursor_history if t >= cutoff
        ]

        if len(self._cursor_history) < 2:
            return False

        # Check max displacement from current position
        cx, cy = state.cursor_x, state.cursor_y
        for hx, hy, _ in self._cursor_history:
            dx = abs(cx - hx)
            dy = abs(cy - hy)
            if dx > self._stable_threshold or dy > self._stable_threshold:
                return False

        return True

    def _poll_voice(self) -> VoiceIntent | None:
        """
        Poll the STT engine for new text and parse it into an intent.
        Non-blocking.
        """
        if self._stt is None or not self._stt.is_active:
            return None

        text = self._stt.get_text(timeout=0.0)
        if text is None:
            return None

        # Tier 1: fast regex parse
        intent = self._parser.parse(text)

        # Tier 2: submit to LLM for async refinement
        if self._llm is not None and self._llm.is_active:
            self._llm.submit(text)

        if intent.is_valid():
            self.voice_latency_ms = (
                (time.perf_counter() - intent.timestamp) * 1000
            )
            return intent

        return None

    def _check_llm_override(self) -> VoiceIntent | None:
        """Check for LLM results that can refine the current intent."""
        if self._llm is None or not self._llm.is_active:
            return None

        result = self._llm.get_result(timeout=0.0)
        if result is not None and result.is_valid():
            return result
        return None

    def process(self, state: GestureState) -> FusedAction | None:
        """
        Main fusion logic — called once per frame from the main loop.

        Parameters
        ----------
        state : GestureState
            Current gesture pipeline snapshot.

        Returns
        -------
        FusedAction if an action should be executed, None otherwise.
        """
        now = state.timestamp
        cursor_stable = self._is_cursor_stable(state)

        # Update mode label
        self.voice_active = (self._stt is not None and self._stt.is_active)
        self.mode_label = "VOICE + GESTURE" if self.voice_active else "GESTURE ONLY"

        # ── Poll voice pipeline ──
        new_intent = self._poll_voice()
        llm_override = self._check_llm_override()

        # Prefer LLM result if it has higher confidence and is recent
        if llm_override is not None:
            if (self._last_voice_intent is not None and
                    llm_override.confidence > self._last_voice_intent.confidence):
                new_intent = llm_override

        # Update active intent
        if new_intent is not None and new_intent.is_valid():
            self._last_voice_intent = new_intent
            self.last_command_text = new_intent.raw_text
            self.last_intent_name = new_intent.intent
            self.last_confidence = new_intent.confidence

        # ── Check if voice intent is still valid (temporal window) ──
        active_intent: VoiceIntent | None = None
        if self._last_voice_intent is not None:
            age = now - self._last_voice_intent.timestamp
            if age <= self._voice_window:
                active_intent = self._last_voice_intent
            else:
                # Expired
                self._last_voice_intent = None

        # ── No voice intent → no fused action ──
        if active_intent is None:
            return None

        # ── Decision engine: match voice intent with gesture state ──
        action = self._decide(active_intent, state, cursor_stable)

        if action is not None and action.is_valid:
            self.last_action = action.action.value
            # Consume the intent (one-shot)
            self._last_voice_intent = None
            return action

        return None

    def _decide(self, intent: VoiceIntent, state: GestureState,
                cursor_stable: bool) -> FusedAction | None:
        """
        Core decision logic: maps (voice_intent, gesture_state) → FusedAction.
        """
        now = state.timestamp
        i = intent.intent
        m = intent.modifier

        # ── Click ──
        if i == "click":
            if state.gesture in ("CURSOR", "NONE") and state.hand_detected:
                return FusedAction(
                    action=ActionType.CLICK,
                    confidence=intent.confidence,
                    source="fused" if state.pinch_detected else "voice",
                    voice_text=intent.raw_text,
                    voice_intent=i,
                    timestamp=now,
                )

        # ── Double click ──
        elif i == "double_click":
            if state.gesture in ("CURSOR", "NONE") and state.hand_detected:
                return FusedAction(
                    action=ActionType.DOUBLE_CLICK,
                    confidence=intent.confidence,
                    source="voice",
                    voice_text=intent.raw_text,
                    voice_intent=i,
                    timestamp=now,
                )

        # ── Open (= click at pointer) ──
        elif i == "open":
            if state.hand_detected and cursor_stable:
                return FusedAction(
                    action=ActionType.OPEN,
                    confidence=intent.confidence,
                    source="fused",
                    voice_text=intent.raw_text,
                    voice_intent=i,
                    timestamp=now,
                )

        # ── Scroll adjustment ──
        elif i == "scroll":
            if m == "faster":
                self._scroll_multiplier = min(self._scroll_multiplier * 2.0, 8.0)
                return FusedAction(
                    action=ActionType.SCROLL_ADJUST,
                    scroll_multiplier=self._scroll_multiplier,
                    confidence=intent.confidence,
                    source="voice",
                    voice_text=intent.raw_text,
                    voice_intent=f"scroll_{m}",
                    timestamp=now,
                )
            elif m == "slower":
                self._scroll_multiplier = max(self._scroll_multiplier / 2.0, 0.25)
                return FusedAction(
                    action=ActionType.SCROLL_ADJUST,
                    scroll_multiplier=self._scroll_multiplier,
                    confidence=intent.confidence,
                    source="voice",
                    voice_text=intent.raw_text,
                    voice_intent=f"scroll_{m}",
                    timestamp=now,
                )
            elif m in ("up", "down"):
                direction = -1.0 if m == "up" else 1.0
                return FusedAction(
                    action=ActionType.SCROLL_DIRECTION,
                    scroll_direction=direction,
                    confidence=intent.confidence,
                    source="voice",
                    voice_text=intent.raw_text,
                    voice_intent=f"scroll_{m}",
                    timestamp=now,
                )

        # ── Stop scrolling ──
        elif i == "stop":
            self._scroll_multiplier = 1.0  # reset multiplier
            if self._drag_state == DragState.DRAGGING:
                # Stop can also end a drag
                self._drag_state = DragState.IDLE
                return FusedAction(
                    action=ActionType.DRAG_END,
                    confidence=intent.confidence,
                    source="voice",
                    voice_text=intent.raw_text,
                    voice_intent=i,
                    timestamp=now,
                )
            return FusedAction(
                action=ActionType.STOP_SCROLL,
                confidence=intent.confidence,
                source="voice",
                voice_text=intent.raw_text,
                voice_intent=i,
                timestamp=now,
            )

        # ── Drag ──
        elif i == "drag":
            if (state.gesture in ("CURSOR", "NONE") and
                    state.hand_detected and cursor_stable):
                self._drag_state = DragState.DRAGGING
                return FusedAction(
                    action=ActionType.DRAG_START,
                    confidence=intent.confidence,
                    source="voice",
                    voice_text=intent.raw_text,
                    voice_intent=i,
                    timestamp=now,
                )

        # ── Drop / release ──
        elif i == "drop":
            if self._drag_state == DragState.DRAGGING:
                self._drag_state = DragState.IDLE
                return FusedAction(
                    action=ActionType.DRAG_END,
                    confidence=intent.confidence,
                    source="voice",
                    voice_text=intent.raw_text,
                    voice_intent=i,
                    timestamp=now,
                )

        return None

    def reset(self):
        """Reset all fusion state."""
        self._last_voice_intent = None
        self._drag_state = DragState.IDLE
        self._scroll_multiplier = 1.0
        self._cursor_history.clear()
        self.last_command_text = ""
        self.last_intent_name = ""
        self.last_confidence = 0.0
        self.last_action = ""
