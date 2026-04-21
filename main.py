"""
main.py — Entry point for gesture-controlled scrolling + cursor + click + voice.

Usage:
    python main.py                  # Run with defaults (gesture only)
    python main.py --voice          # Enable voice commands
    python main.py --use-llm        # Enable Gemma LLM intent (requires GGUF model)
    python main.py --calibrate      # Run calibration phase first
    python main.py --invert         # Use natural (inverted) scrolling
    python main.py --camera 1       # Use camera index 1
    python main.py --no-overlay     # Disable debug overlay
    python main.py --no-cursor      # Disable cursor control (scroll only)

Controls (keyboard):
    q / ESC     Quit
    c           Toggle calibration mode
    d           Toggle debug overlay
    v           Toggle voice mode on/off
    +/-         Adjust sensitivity in real-time

Gesture Modes:
    ☝️  Index finger only    → Cursor control (move mouse)
    ✌️  Index + middle       → Scroll mode (existing behavior)
    🤏  Pinch (thumb+index)  → Click (single / double)
    ✋  Open palm / fist     → Idle

Voice Commands (when --voice enabled):
    "click here"      → Click at cursor
    "double click"    → Double click at cursor
    "open this"       → Click at cursor (= open)
    "scroll faster"   → Increase scroll speed
    "scroll slower"   → Decrease scroll speed
    "stop scrolling"  → Stop scrolling
    "drag this"       → Initiate drag
    "drop" / "release"→ End drag
"""

import argparse
import sys
import time
import logging
import cv2
import numpy as np

from video_capture import VideoCapture
from hand_tracker import HandTracker, HandResult
from gesture_detector import (
    GestureDetector, GESTURE_SCROLL, GESTURE_CURSOR,
    GESTURE_CLICK, GESTURE_NONE, GESTURE_FIST, GESTURE_OPEN
)
from motion_analyzer import MotionAnalyzer
from scroll_controller import ScrollController
from cursor_controller import CursorController
from click_detector import ClickDetector
from utils import FPSCounter, CalibrationStore

# Voice / multimodal imports (lazy-loaded in main())
# from voice_input import VoiceInput
# from speech_to_text import SpeechToText
# from intent_parser import IntentParser
# from llm_intent import LLMIntent
# from fusion_engine import FusionEngine, GestureState, FusedAction, ActionType

logger = logging.getLogger(__name__)


# ────────────────────────── Debug overlay ──────────────────────────

class DebugOverlay:
    """Renders real-time diagnostics onto the video frame."""

    # Color palette (BGR)
    BG = (20, 20, 20)
    ACTIVE = (0, 255, 120)
    INACTIVE = (80, 80, 200)
    TEXT = (220, 220, 220)
    ACCENT = (255, 180, 0)
    VELOCITY_POS = (0, 120, 255)    # orange for scroll-down
    VELOCITY_NEG = (255, 120, 0)    # blue for scroll-up
    CURSOR_COLOR = (255, 200, 50)   # cyan-yellow for cursor mode
    CLICK_COLOR = (0, 0, 255)       # red for click event
    PINCH_READY = (0, 255, 255)     # yellow for pinch-ready
    VOICE_COLOR = (255, 100, 255)   # magenta for voice
    VOICE_ACTIVE_COLOR = (0, 255, 180)  # teal for listening
    FUSED_COLOR = (100, 255, 100)   # bright green for fused action

    # Gesture → display color
    MODE_COLORS = {
        GESTURE_CURSOR: (255, 200, 50),   # gold
        GESTURE_SCROLL: (0, 255, 120),    # green
        GESTURE_CLICK:  (0, 0, 255),      # red
        GESTURE_NONE:   (80, 80, 200),    # grey-blue
        GESTURE_FIST:   (80, 80, 200),
        GESTURE_OPEN:   (80, 80, 200),
    }

    def __init__(self, frame_w: int, frame_h: int):
        self.fw = frame_w
        self.fh = frame_h
        self._action_flash_time: float = 0.0
        self._action_flash_text: str = ""

    def trigger_action_flash(self, text: str):
        """Trigger a visual flash for a fused action."""
        self._action_flash_time = time.perf_counter()
        self._action_flash_text = text

    def draw(self, frame: np.ndarray, fps: float, gesture: str,
             finger_y: float, velocity: float, calibrating: bool,
             hand: HandResult | None, calibration: CalibrationStore,
             cursor_ctrl: CursorController | None = None,
             click_det: ClickDetector | None = None,
             click_event: str = "",
             fusion_engine=None):
        """Overlay all debug info onto frame (mutates in place)."""
        h, w = frame.shape[:2]
        mode_color = self.MODE_COLORS.get(gesture, self.INACTIVE)

        # --- Semi-transparent panel on the left ---
        panel_h = 310 if cursor_ctrl else 220
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (280, panel_h), self.BG, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        y0 = 28
        dy = 26

        # FPS
        fps_color = self.ACTIVE if fps >= 25 else self.INACTIVE
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)

        # Gesture / Mode state
        y0 += dy
        cv2.putText(frame, f"Mode: {gesture}", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)

        # Mode indicator dot
        cv2.circle(frame, (265, y0 - 8), 8, mode_color, -1)

        if gesture == GESTURE_SCROLL:
            # ── Scroll-specific info ──
            y0 += dy
            cv2.putText(frame, f"Finger Y: {finger_y:.4f}", (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, self.TEXT, 1)

            y0 += dy
            vel_color = self.VELOCITY_POS if velocity > 0 else self.VELOCITY_NEG
            cv2.putText(frame, f"Velocity: {velocity:+.2f}", (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, vel_color, 2)

            # Velocity bar
            y0 += 8
            bar_cx = 140
            bar_len = int(np.clip(velocity * 15, -100, 100))
            cv2.rectangle(frame, (bar_cx, y0), (bar_cx + bar_len, y0 + 10),
                          vel_color, -1)
            cv2.line(frame, (bar_cx, y0 - 2), (bar_cx, y0 + 12), self.TEXT, 1)
            y0 += 18

            # Scroll multiplier (from voice)
            if fusion_engine and fusion_engine.scroll_multiplier != 1.0:
                y0 += dy
                cv2.putText(frame,
                            f"Scroll x{fusion_engine.scroll_multiplier:.1f}",
                            (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                            self.VOICE_COLOR, 2)

        elif gesture == GESTURE_CURSOR and cursor_ctrl:
            # ── Cursor-specific info ──
            y0 += dy
            cv2.putText(frame,
                        f"Cursor: ({cursor_ctrl.screen_x:.0f}, {cursor_ctrl.screen_y:.0f})",
                        (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                        self.CURSOR_COLOR, 1)

            y0 += dy
            cv2.putText(frame,
                        f"Screen: {cursor_ctrl.screen_w}x{cursor_ctrl.screen_h}",
                        (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.TEXT, 1)

            if click_det:
                y0 += dy
                pinch_color = self.PINCH_READY if click_det.pinch_distance < 0.08 else self.TEXT
                cv2.putText(frame,
                            f"Pinch: {click_det.pinch_distance:.3f}  [{click_det.state}]",
                            (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            pinch_color, 1)

                # Pinch distance bar
                y0 += 8
                bar_w = int(np.clip((1.0 - click_det.pinch_distance) * 200, 0, 200))
                bar_color = self.CLICK_COLOR if click_det.pinch_distance < 0.045 else self.PINCH_READY
                cv2.rectangle(frame, (10, y0), (10 + bar_w, y0 + 8), bar_color, -1)
                cv2.rectangle(frame, (10, y0), (210, y0 + 8), self.TEXT, 1)
                y0 += 16

                if click_event:
                    y0 += dy
                    cv2.putText(frame, f">> {click_event} <<", (10, y0),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.CLICK_COLOR, 2)

            # Drag indicator
            if fusion_engine and fusion_engine.drag_active:
                y0 += dy
                cv2.putText(frame, "** DRAGGING **", (10, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.VOICE_COLOR, 2)

        # Sensitivity / calibration
        y0 += dy
        sens_text = f"Sens: {calibration.sensitivity:.2f}  DZ: {calibration.dead_zone:.4f}"
        cv2.putText(frame, sens_text, (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.ACCENT, 1)

        if calibrating:
            y0 += dy
            cv2.putText(frame, "** CALIBRATING **", (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # --- Draw finger marker ---
        if hand is not None:
            idx_tip = hand.landmarks[HandTracker.INDEX_TIP]
            cx, cy = int(idx_tip[0] * w), int(idx_tip[1] * h)

            if gesture == GESTURE_CURSOR:
                # Cursor mode: large crosshair + circle
                color = self.CURSOR_COLOR

                # Check if click-ready (pinch close)
                if click_det and click_det.pinch_distance < 0.08:
                    color = self.PINCH_READY
                if click_event:
                    color = self.CLICK_COLOR

                # Drag mode: change crosshair to magenta
                if fusion_engine and fusion_engine.drag_active:
                    color = self.VOICE_COLOR

                # Crosshair
                cv2.line(frame, (cx - 20, cy), (cx + 20, cy), color, 2)
                cv2.line(frame, (cx, cy - 20), (cx, cy + 20), color, 2)
                cv2.circle(frame, (cx, cy), 18, color, 2)
                cv2.circle(frame, (cx, cy), 4, color, -1)

                # Draw thumb tip indicator when pinching
                if click_det and click_det.pinch_distance < 0.08:
                    thumb_tip = hand.landmarks[HandTracker.THUMB_TIP]
                    tx, ty = int(thumb_tip[0] * w), int(thumb_tip[1] * h)
                    cv2.line(frame, (cx, cy), (tx, ty), self.PINCH_READY, 2)
                    cv2.circle(frame, (tx, ty), 8, self.PINCH_READY, 2)

            elif gesture == GESTURE_SCROLL:
                # Scroll mode: existing visualization
                color = self.ACTIVE
                cv2.circle(frame, (cx, cy), 14, color, 2)
                cv2.circle(frame, (cx, cy), 4, color, -1)

                mid_tip = hand.landmarks[HandTracker.MIDDLE_TIP]
                mx, my = int(mid_tip[0] * w), int(mid_tip[1] * h)
                cv2.line(frame, (cx, cy), (mx, my), self.ACTIVE, 2)
                cv2.circle(frame, (mx, my), 10, self.ACTIVE, 2)

            else:
                # Other gestures: dim marker
                cv2.circle(frame, (cx, cy), 10, self.INACTIVE, 1)
                cv2.circle(frame, (cx, cy), 3, self.INACTIVE, -1)

        # ── Voice panel (right side) ──
        if fusion_engine is not None:
            self._draw_voice_panel(frame, fusion_engine, w, h)

        # ── Action flash effect ──
        flash_age = time.perf_counter() - self._action_flash_time
        if flash_age < 0.8 and self._action_flash_text:
            alpha = max(0.0, 1.0 - flash_age / 0.8)
            flash_color = (
                int(self.FUSED_COLOR[0] * alpha),
                int(self.FUSED_COLOR[1] * alpha),
                int(self.FUSED_COLOR[2] * alpha),
            )
            text_size = cv2.getTextSize(
                self._action_flash_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )[0]
            tx = (w - text_size[0]) // 2
            ty = h // 2
            cv2.putText(frame, self._action_flash_text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, flash_color, 2)

        # --- Status bar at bottom ---
        cv2.rectangle(frame, (0, h - 34), (w, h), self.BG, -1)

        if gesture == GESTURE_CURSOR:
            status = "CURSOR ACTIVE"
            if click_event:
                status = f"CURSOR — {click_event}!"
            if fusion_engine and fusion_engine.drag_active:
                status = "CURSOR — DRAGGING"
        elif gesture == GESTURE_SCROLL:
            status = "SCROLL ACTIVE"
            if fusion_engine and fusion_engine.scroll_multiplier != 1.0:
                status += f" (x{fusion_engine.scroll_multiplier:.1f})"
        else:
            status = "Waiting for gesture (index=cursor, index+middle=scroll)"

        st_color = mode_color
        cv2.putText(frame, status, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, st_color, 1)

        mode_text = fusion_engine.mode_label if fusion_engine else "GESTURE ONLY"
        keys_text = f"[q]Quit [c]Cal [d]Overlay [v]Voice  |  {mode_text}"
        cv2.putText(frame, keys_text, (w - 480, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)

    def _draw_voice_panel(self, frame: np.ndarray, fusion, w: int, h: int):
        """Draw the voice / multimodal status panel on the right side."""
        panel_w = 260
        panel_h = 160
        px = w - panel_w - 5
        py = 5

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), self.BG, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Header
        voice_on = fusion.voice_active
        header_color = self.VOICE_ACTIVE_COLOR if voice_on else self.INACTIVE
        header_text = "🎤 LISTENING" if voice_on else "🎤 OFF"
        cv2.putText(frame, header_text, (px + 8, py + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, header_color, 2)

        # Listening indicator dot (pulsing)
        if voice_on:
            pulse = abs((time.perf_counter() * 3) % 2.0 - 1.0)
            radius = int(6 + pulse * 4)
            cv2.circle(frame, (px + panel_w - 20, py + 18), radius,
                       self.VOICE_ACTIVE_COLOR, -1)

        # Last command
        y = py + 48
        cmd = fusion.last_command_text or "—"
        if len(cmd) > 28:
            cmd = cmd[:25] + "..."
        cv2.putText(frame, f"Cmd: {cmd}", (px + 8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.TEXT, 1)

        # Intent
        y += 24
        intent_text = fusion.last_intent_name or "—"
        conf = fusion.last_confidence
        conf_color = self.ACTIVE if conf > 0.7 else self.ACCENT if conf > 0.4 else self.INACTIVE
        cv2.putText(frame, f"Intent: {intent_text}", (px + 8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, conf_color, 1)

        # Confidence bar
        y += 4
        bar_w = int(conf * (panel_w - 20))
        cv2.rectangle(frame, (px + 8, y), (px + 8 + bar_w, y + 5),
                      conf_color, -1)
        cv2.rectangle(frame, (px + 8, y), (px + panel_w - 12, y + 5),
                      self.TEXT, 1)

        # Last action
        y += 22
        action_text = fusion.last_action or "—"
        cv2.putText(frame, f"Action: {action_text}", (px + 8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.FUSED_COLOR, 1)

        # Voice latency
        y += 24
        lat = fusion.voice_latency_ms
        lat_color = self.ACTIVE if lat < 150 else self.ACCENT if lat < 300 else self.CLICK_COLOR
        cv2.putText(frame, f"Latency: {lat:.0f} ms", (px + 8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, lat_color, 1)

        # Border
        border_color = self.VOICE_ACTIVE_COLOR if voice_on else self.INACTIVE
        cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h),
                      border_color, 1)


# ────────────────────────── Calibration phase ──────────────────────────

def run_calibration(cap: VideoCapture, tracker: HandTracker,
                    gesture_det: GestureDetector, calibration: CalibrationStore,
                    duration: float = 5.0):
    """
    Interactive calibration: user moves index finger up and down slowly
    for `duration` seconds while making the scroll gesture.
    Samples are collected and used to derive dead-zone and sensitivity.
    """
    print(f"\n{'='*50}")
    print("  CALIBRATION MODE")
    print("  Make the scroll gesture (index+middle fingers up)")
    print("  and move your hand slowly up and down.")
    print(f"  Recording for {duration:.0f} seconds...")
    print(f"{'='*50}\n")

    prev_y = None
    start = time.monotonic()
    sample_count = 0

    while time.monotonic() - start < duration:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame = cv2.flip(frame, 1)
        hands = tracker.process(frame)

        if hands:
            gesture = gesture_det.classify(hands[0])
            y = hands[0].landmarks[HandTracker.INDEX_TIP][1]
            if prev_y is not None:
                dy = abs(y - prev_y)
                if dy > 1e-6:
                    calibration.record_sample(dy)
                    sample_count += 1
            prev_y = y
        else:
            prev_y = None

        # Show progress
        elapsed = time.monotonic() - start
        progress = min(elapsed / duration, 1.0)
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 40), (int(w * progress), h), (0, 255, 120), -1)
        cv2.putText(frame, f"Calibrating... {sample_count} samples ({elapsed:.1f}s / {duration:.0f}s)",
                    (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("AutoScroll", frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    calibration.finalise()
    print(f"\nCalibration complete! Parameters: {calibration.to_dict()}\n")


# ────────────────────────── Voice action executor ──────────────────────────

class VoiceActionExecutor:
    """
    Executes fused actions from the voice+gesture fusion engine.
    Translates FusedAction objects into system-level input events.
    """

    def __init__(self, scroller: ScrollController, calibration: CalibrationStore):
        self._scroller = scroller
        self._calibration = calibration
        self._dragging = False

        try:
            from pynput.mouse import Controller as MouseController, Button
            self._mouse = MouseController()
            self._button = Button.left
        except ImportError:
            self._mouse = None
            self._button = None

    def execute(self, action) -> str:
        """
        Execute a FusedAction and return a description string.
        """
        from fusion_engine import ActionType

        if action.action == ActionType.CLICK:
            return self._click()
        elif action.action == ActionType.DOUBLE_CLICK:
            return self._double_click()
        elif action.action == ActionType.OPEN:
            return self._click()  # open = click
        elif action.action == ActionType.SCROLL_ADJUST:
            return self._scroll_adjust(action.scroll_multiplier)
        elif action.action == ActionType.SCROLL_DIRECTION:
            return self._scroll_direction(action.scroll_direction)
        elif action.action == ActionType.STOP_SCROLL:
            return self._stop_scroll()
        elif action.action == ActionType.DRAG_START:
            return self._drag_start()
        elif action.action == ActionType.DRAG_END:
            return self._drag_end()
        return ""

    def _click(self) -> str:
        if self._mouse:
            self._mouse.click(self._button, 1)
        return "VOICE CLICK"

    def _double_click(self) -> str:
        if self._mouse:
            self._mouse.click(self._button, 2)
        return "VOICE DOUBLE CLICK"

    def _scroll_adjust(self, multiplier: float) -> str:
        self._calibration.sensitivity = self._calibration.sensitivity  # maintain base
        return f"SCROLL x{multiplier:.1f}"

    def _scroll_direction(self, direction: float) -> str:
        # Inject a scroll event in the requested direction
        velocity = direction * 3.0  # moderate speed
        self._scroller.scroll(velocity)
        return f"SCROLL {'DOWN' if direction > 0 else 'UP'}"

    def _stop_scroll(self) -> str:
        self._scroller.reset()
        return "SCROLL STOPPED"

    def _drag_start(self) -> str:
        if self._mouse and not self._dragging:
            self._mouse.press(self._button)
            self._dragging = True
        return "DRAG START"

    def _drag_end(self) -> str:
        if self._mouse and self._dragging:
            self._mouse.release(self._button)
            self._dragging = False
        return "DRAG END"

    def cleanup(self):
        """Release any held buttons."""
        if self._dragging and self._mouse:
            self._mouse.release(self._button)
            self._dragging = False


# ────────────────────────── Main loop ──────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Gesture-controlled OS-level scrolling + cursor + click + voice via webcam"
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration first")
    parser.add_argument("--invert", action="store_true", help="Invert scroll direction")
    parser.add_argument("--no-overlay", action="store_true", help="Disable debug overlay")
    parser.add_argument("--no-cursor", action="store_true", help="Disable cursor control")
    parser.add_argument("--sensitivity", type=float, default=1.0, help="Scroll sensitivity multiplier")

    # Voice / multimodal arguments
    parser.add_argument("--voice", action="store_true", help="Enable voice commands")
    parser.add_argument("--use-llm", action="store_true",
                        help="Enable Gemma LLM for intent parsing (requires GGUF model)")
    parser.add_argument("--llm-model-path", type=str, default=None,
                        help="Path to Gemma GGUF model file")
    parser.add_argument("--voice-window", type=float, default=1.5,
                        help="Voice intent validity window in seconds (default: 1.5)")
    parser.add_argument("--voice-cooldown", type=float, default=1.0,
                        help="Minimum seconds between same voice commands (default: 1.0)")
    args = parser.parse_args()

    # ── Configure logging ──
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Initialise core components ──
    print("Starting AutoScroll...")
    print(f"  Camera: {args.camera}  Resolution: {args.width}×{args.height}")

    cap = VideoCapture(source=args.camera, width=args.width, height=args.height)
    cap.start()
    print(f"  Actual resolution: {cap.width}×{cap.height}")

    tracker = HandTracker(max_hands=1)
    gesture_det = GestureDetector(extension_ratio=1.15)
    calibration = CalibrationStore()
    calibration.sensitivity = args.sensitivity
    motion = MotionAnalyzer(calibration=calibration)
    scroller = ScrollController(invert=args.invert)
    fps_counter = FPSCounter(window=30)
    overlay = DebugOverlay(cap.width, cap.height)

    # Cursor and click modules
    cursor_ctrl: CursorController | None = None
    click_det: ClickDetector | None = None

    if not args.no_cursor:
        cursor_ctrl = CursorController(
            active_zone=(0.15, 0.85, 0.10, 0.80),
            smoothing_alpha=0.35,
            dead_zone_px=3.0,
        )
        click_det = ClickDetector(
            pinch_threshold=0.045,
            release_threshold=0.065,
            cooldown_sec=0.25,
            double_click_window=0.40,
        )
        print(f"  Screen: {cursor_ctrl.screen_w}×{cursor_ctrl.screen_h}")
        print(f"  Cursor control: ENABLED")
    else:
        print(f"  Cursor control: DISABLED")

    # ── Initialise voice / multimodal components ──
    voice_input = None
    stt = None
    intent_parser = None
    llm_intent = None
    fusion_engine = None
    voice_executor = None
    voice_enabled = args.voice

    if voice_enabled:
        print("\n  Initialising voice pipeline...")
        try:
            from voice_input import VoiceInput
            from speech_to_text import SpeechToText
            from intent_parser import IntentParser
            from fusion_engine import FusionEngine

            # Audio capture
            voice_input = VoiceInput(sample_rate=16000, block_duration_ms=100)
            mic_ok = voice_input.start()
            if mic_ok:
                print(f"  Microphone: ACTIVE")
            else:
                print(f"  Microphone: FAILED (voice disabled)")
                voice_enabled = False

            if voice_enabled:
                # Speech-to-text
                stt = SpeechToText(voice_input=voice_input, sample_rate=16000)
                stt_ok = stt.start()
                if stt_ok:
                    print(f"  Speech-to-text: ACTIVE (Vosk)")
                else:
                    print(f"  Speech-to-text: FAILED (voice disabled)")
                    voice_enabled = False

            if voice_enabled:
                # Intent parser (always available)
                intent_parser = IntentParser(
                    min_confidence=0.5,
                    cooldown_sec=args.voice_cooldown,
                )

                # Optional LLM
                if args.use_llm:
                    try:
                        from llm_intent import LLMIntent
                        llm_intent = LLMIntent(model_path=args.llm_model_path)
                        llm_ok = llm_intent.start()
                        if llm_ok:
                            print(f"  LLM intent: ACTIVE (Gemma)")
                        else:
                            print(f"  LLM intent: UNAVAILABLE (using regex only)")
                            llm_intent = None
                    except Exception as e:
                        print(f"  LLM intent: ERROR ({e})")
                        llm_intent = None

                # Fusion engine
                fusion_engine = FusionEngine(
                    intent_parser=intent_parser,
                    stt=stt,
                    llm_intent=llm_intent,
                    voice_window_sec=args.voice_window,
                )

                # Voice action executor
                voice_executor = VoiceActionExecutor(scroller, calibration)

                print(f"  Voice pipeline: READY")
                print(f"  Voice window: {args.voice_window}s")

        except ImportError as e:
            print(f"  Voice dependencies missing: {e}")
            print(f"  Install with: pip install vosk sounddevice")
            voice_enabled = False
        except Exception as e:
            print(f"  Voice init error: {e}")
            voice_enabled = False

    show_overlay = not args.no_overlay
    calibrating = False

    print(f"\n  Scroll backend: {scroller.backend}")
    print(f"  Invert: {args.invert}")
    print(f"  Overlay: {show_overlay}")
    print(f"  Voice: {'ENABLED' if voice_enabled else 'DISABLED'}")
    print("\nReady! Gestures:")
    print("  ☝️  Index finger only  → Cursor mode (move & click)")
    print("  ✌️  Index + middle     → Scroll mode")
    print("  🤏 Pinch thumb+index  → Click")
    if voice_enabled:
        print("\nVoice commands:")
        print("  'click here'       → Click at cursor")
        print("  'double click'     → Double click")
        print("  'open this'        → Open (click) at cursor")
        print("  'scroll faster'    → Increase scroll speed")
        print("  'scroll slower'    → Decrease scroll speed")
        print("  'stop scrolling'   → Stop scroll")
        print("  'drag this'        → Start drag")
        print("  'drop' / 'release' → End drag")
    print("\nPress 'q' or ESC to quit, 'v' to toggle voice.\n")

    # ── Optional initial calibration ──
    if args.calibrate:
        run_calibration(cap, tracker, gesture_det, calibration, duration=5.0)

    # ── Main processing loop ──
    prev_gesture = GESTURE_NONE

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.001)
                continue

            # Mirror for intuitive interaction
            frame = cv2.flip(frame, 1)

            # ── Hand tracking ──
            hands = tracker.process(frame)
            current_fps = fps_counter.tick()

            gesture = GESTURE_NONE
            velocity = 0.0
            finger_y = 0.0
            click_event = ""
            hand_result: HandResult | None = None

            if hands:
                hand_result = hands[0]
                gesture = gesture_det.classify(hand_result)

                if calibrating:
                    # Collect calibration samples
                    y = hand_result.landmarks[HandTracker.INDEX_TIP][1]
                    if motion._prev_y is not None:
                        calibration.record_sample(abs(y - motion._prev_y))
                    motion.update(hand_result)

                elif gesture == GESTURE_CURSOR and cursor_ctrl:
                    # ── Cursor mode ──
                    # Reset scroll state if we just switched from scroll
                    if prev_gesture == GESTURE_SCROLL:
                        motion.reset()
                        scroller.reset()

                    # Move cursor
                    idx_tip = hand_result.landmarks[HandTracker.INDEX_TIP]
                    cursor_ctrl.update(idx_tip[0], idx_tip[1])

                    # Check for click
                    if click_det:
                        thumb_tip = hand_result.landmarks[HandTracker.THUMB_TIP]
                        index_tip = hand_result.landmarks[HandTracker.INDEX_TIP]
                        click_event = click_det.update(
                            thumb_tip, index_tip, is_cursor_mode=True
                        )

                elif gesture == GESTURE_SCROLL:
                    # ── Scroll mode (existing behavior) ──
                    # Reset cursor state if we just switched from cursor
                    if prev_gesture == GESTURE_CURSOR:
                        if cursor_ctrl:
                            cursor_ctrl.reset()
                        if click_det:
                            click_det.reset()

                    velocity = motion.update(hand_result)
                    finger_y = motion.finger_y

                    # Apply voice scroll multiplier
                    effective_velocity = velocity
                    if fusion_engine:
                        effective_velocity = velocity * fusion_engine.scroll_multiplier

                    scroller.scroll(effective_velocity)

                else:
                    # ── No active gesture — reset everything ──
                    motion.reset()
                    scroller.reset()
                    if cursor_ctrl:
                        cursor_ctrl.reset()
                    if click_det:
                        click_det.reset()

                prev_gesture = gesture

            else:
                # No hand — reset everything
                motion.reset()
                scroller.reset()
                if cursor_ctrl:
                    cursor_ctrl.reset()
                if click_det:
                    click_det.reset()
                prev_gesture = GESTURE_NONE

            # ── Voice + gesture fusion ──
            if fusion_engine and voice_enabled:
                from fusion_engine import GestureState, ActionType

                # Build gesture state snapshot
                g_state = GestureState(
                    gesture=gesture,
                    cursor_x=cursor_ctrl.screen_x if cursor_ctrl else 0.0,
                    cursor_y=cursor_ctrl.screen_y if cursor_ctrl else 0.0,
                    cursor_stable=False,  # fusion engine tracks its own stability
                    pinch_detected=(click_det.pinch_distance < 0.06) if click_det else False,
                    pinch_distance=click_det.pinch_distance if click_det else 1.0,
                    scroll_velocity=velocity,
                    hand_detected=hand_result is not None,
                    timestamp=time.perf_counter(),
                )

                # Process fusion
                fused_action = fusion_engine.process(g_state)

                # Execute fused action
                if fused_action and fused_action.is_valid and voice_executor:
                    action_desc = voice_executor.execute(fused_action)
                    if action_desc:
                        print(f"  🎤 Voice action: {action_desc} "
                              f"(intent={fused_action.voice_intent}, "
                              f"conf={fused_action.confidence:.2f})")
                        if show_overlay:
                            overlay.trigger_action_flash(action_desc)

            # ── Debug overlay ──
            if show_overlay:
                overlay.draw(
                    frame, current_fps, gesture, finger_y,
                    velocity, calibrating, hand_result, calibration,
                    cursor_ctrl=cursor_ctrl,
                    click_det=click_det,
                    click_event=click_event,
                    fusion_engine=fusion_engine if voice_enabled else None,
                )

            cv2.imshow("AutoScroll", frame)

            # ── Keyboard input ──
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # q or ESC
                break
            elif key == ord('d'):
                show_overlay = not show_overlay
            elif key == ord('v'):
                # Toggle voice mode
                if voice_input is not None and stt is not None:
                    if voice_enabled:
                        voice_enabled = False
                        print("  Voice: DISABLED")
                    else:
                        voice_enabled = True
                        print("  Voice: ENABLED")
                else:
                    print("  Voice: not available (start with --voice)")
            elif key == ord('c'):
                if not calibrating:
                    calibrating = True
                    calibration._samples.clear()
                    print("Calibration started — move your hand up/down with scroll gesture.")
                    print("Press 'c' again to finish.")
                else:
                    calibrating = False
                    calibration.finalise()
                    print(f"Calibration finished: {calibration.to_dict()}")
            elif key == ord('+') or key == ord('='):
                calibration.sensitivity = min(calibration.sensitivity + 0.1, 5.0)
                print(f"Sensitivity: {calibration.sensitivity:.2f}")
            elif key == ord('-'):
                calibration.sensitivity = max(calibration.sensitivity - 0.1, 0.1)
                print(f"Sensitivity: {calibration.sensitivity:.2f}")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        # Cleanup voice pipeline
        if voice_executor:
            voice_executor.cleanup()
        if llm_intent:
            llm_intent.stop()
        if stt:
            stt.stop()
        if voice_input:
            voice_input.stop()

        cap.stop()
        tracker.close()
        cv2.destroyAllWindows()
        print("AutoScroll stopped.")


if __name__ == "__main__":
    main()
