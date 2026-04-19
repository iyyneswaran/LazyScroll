"""
main.py — Entry point for gesture-controlled scrolling + cursor + click.

Usage:
    python main.py                  # Run with defaults
    python main.py --calibrate      # Run calibration phase first
    python main.py --invert         # Use natural (inverted) scrolling
    python main.py --camera 1       # Use camera index 1
    python main.py --no-overlay     # Disable debug overlay
    python main.py --no-cursor      # Disable cursor control (scroll only)

Controls (keyboard):
    q / ESC     Quit
    c           Toggle calibration mode
    d           Toggle debug overlay
    +/-         Adjust sensitivity in real-time

Gesture Modes:
    ☝️  Index finger only    → Cursor control (move mouse)
    ✌️  Index + middle       → Scroll mode (existing behavior)
    🤏  Pinch (thumb+index)  → Click (single / double)
    ✋  Open palm / fist     → Idle
"""

import argparse
import sys
import time
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

    def draw(self, frame: np.ndarray, fps: float, gesture: str,
             finger_y: float, velocity: float, calibrating: bool,
             hand: HandResult | None, calibration: CalibrationStore,
             cursor_ctrl: CursorController | None = None,
             click_det: ClickDetector | None = None,
             click_event: str = ""):
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

        # --- Status bar at bottom ---
        cv2.rectangle(frame, (0, h - 34), (w, h), self.BG, -1)

        if gesture == GESTURE_CURSOR:
            status = "CURSOR ACTIVE"
            if click_event:
                status = f"CURSOR — {click_event}!"
        elif gesture == GESTURE_SCROLL:
            status = "SCROLL ACTIVE"
        else:
            status = "Waiting for gesture (index=cursor, index+middle=scroll)"

        st_color = mode_color
        cv2.putText(frame, status, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, st_color, 1)

        keys_text = "[q] Quit  [c] Calibrate  [d] Overlay  [+/-] Sensitivity"
        cv2.putText(frame, keys_text, (w - 430, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)


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


# ────────────────────────── Main loop ──────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Gesture-controlled OS-level scrolling + cursor + click via webcam"
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration first")
    parser.add_argument("--invert", action="store_true", help="Invert scroll direction")
    parser.add_argument("--no-overlay", action="store_true", help="Disable debug overlay")
    parser.add_argument("--no-cursor", action="store_true", help="Disable cursor control")
    parser.add_argument("--sensitivity", type=float, default=1.0, help="Scroll sensitivity multiplier")
    args = parser.parse_args()

    # ── Initialise components ──
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

    show_overlay = not args.no_overlay
    calibrating = False

    print(f"  Scroll backend: {scroller.backend}")
    print(f"  Invert: {args.invert}")
    print(f"  Overlay: {show_overlay}")
    print("\nReady! Gestures:")
    print("  ☝️  Index finger only  → Cursor mode (move & click)")
    print("  ✌️  Index + middle     → Scroll mode")
    print("  🤏 Pinch thumb+index  → Click")
    print("Press 'q' or ESC to quit.\n")

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
                    scroller.scroll(velocity)

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

            # ── Debug overlay ──
            if show_overlay:
                overlay.draw(
                    frame, current_fps, gesture, finger_y,
                    velocity, calibrating, hand_result, calibration,
                    cursor_ctrl=cursor_ctrl,
                    click_det=click_det,
                    click_event=click_event,
                )

            cv2.imshow("AutoScroll", frame)

            # ── Keyboard input ──
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # q or ESC
                break
            elif key == ord('d'):
                show_overlay = not show_overlay
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
        cap.stop()
        tracker.close()
        cv2.destroyAllWindows()
        print("AutoScroll stopped.")


if __name__ == "__main__":
    main()
