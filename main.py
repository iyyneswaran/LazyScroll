"""
main.py — Entry point for gesture-controlled scrolling.

Usage:
    python main.py                  # Run with defaults
    python main.py --calibrate      # Run calibration phase first
    python main.py --invert         # Use natural (inverted) scrolling
    python main.py --camera 1       # Use camera index 1
    python main.py --no-overlay     # Disable debug overlay

Controls (keyboard):
    q / ESC     Quit
    c           Toggle calibration mode
    d           Toggle debug overlay
    +/-         Adjust sensitivity in real-time
"""

import argparse
import sys
import time
import cv2
import numpy as np

from video_capture import VideoCapture
from hand_tracker import HandTracker, HandResult
from gesture_detector import GestureDetector, GESTURE_SCROLL
from motion_analyzer import MotionAnalyzer
from scroll_controller import ScrollController
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

    def __init__(self, frame_w: int, frame_h: int):
        self.fw = frame_w
        self.fh = frame_h

    def draw(self, frame: np.ndarray, fps: float, gesture: str,
             finger_y: float, velocity: float, calibrating: bool,
             hand: HandResult | None, calibration: CalibrationStore):
        """Overlay all debug info onto frame (mutates in place)."""
        h, w = frame.shape[:2]

        # --- Semi-transparent panel on the left ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (260, 220), self.BG, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        y0 = 28
        dy = 28

        # FPS
        fps_color = self.ACTIVE if fps >= 25 else self.INACTIVE
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)

        # Gesture state
        y0 += dy
        gest_color = self.ACTIVE if gesture == GESTURE_SCROLL else self.INACTIVE
        cv2.putText(frame, f"Gesture: {gesture}", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, gest_color, 2)

        # Finger Y
        y0 += dy
        cv2.putText(frame, f"Finger Y: {finger_y:.4f}", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.TEXT, 1)

        # Scroll velocity bar
        y0 += dy
        vel_color = self.VELOCITY_POS if velocity > 0 else self.VELOCITY_NEG
        vel_text = f"Velocity: {velocity:+.2f}"
        cv2.putText(frame, vel_text, (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, vel_color, 2)

        # Velocity bar visualisation
        y0 += 8
        bar_center_x = 130
        bar_len = int(np.clip(velocity * 15, -100, 100))
        cv2.rectangle(frame, (bar_center_x, y0), (bar_center_x + bar_len, y0 + 10),
                      vel_color, -1)
        cv2.line(frame, (bar_center_x, y0 - 2), (bar_center_x, y0 + 12),
                 self.TEXT, 1)

        # Sensitivity / calibration
        y0 += 30
        sens_text = f"Sens: {calibration.sensitivity:.2f}  DZ: {calibration.dead_zone:.4f}"
        cv2.putText(frame, sens_text, (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.ACCENT, 1)

        if calibrating:
            y0 += dy
            cv2.putText(frame, "** CALIBRATING **", (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # --- Draw finger marker ---
        if hand is not None:
            idx_tip = hand.landmarks[HandTracker.INDEX_TIP]
            cx, cy = int(idx_tip[0] * w), int(idx_tip[1] * h)
            color = self.ACTIVE if gesture == GESTURE_SCROLL else self.INACTIVE
            cv2.circle(frame, (cx, cy), 14, color, 2)
            cv2.circle(frame, (cx, cy), 4, color, -1)

            # Draw line between index and middle tip when scroll gesture active
            if gesture == GESTURE_SCROLL:
                mid_tip = hand.landmarks[HandTracker.MIDDLE_TIP]
                mx, my = int(mid_tip[0] * w), int(mid_tip[1] * h)
                cv2.line(frame, (cx, cy), (mx, my), self.ACTIVE, 2)
                cv2.circle(frame, (mx, my), 10, self.ACTIVE, 2)

        # --- Status bar at bottom ---
        cv2.rectangle(frame, (0, h - 30), (w, h), self.BG, -1)
        status = "SCROLL ACTIVE" if gesture == GESTURE_SCROLL else "Waiting for gesture (index + middle up)"
        st_color = self.ACTIVE if gesture == GESTURE_SCROLL else self.TEXT
        cv2.putText(frame, status, (10, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, st_color, 1)

        backend_text = f"[q] Quit  [c] Calibrate  [d] Overlay  [+/-] Sensitivity"
        cv2.putText(frame, backend_text, (w - 420, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)


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
        description="Gesture-controlled OS-level scrolling via webcam"
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration first")
    parser.add_argument("--invert", action="store_true", help="Invert scroll direction")
    parser.add_argument("--no-overlay", action="store_true", help="Disable debug overlay")
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

    show_overlay = not args.no_overlay
    calibrating = False

    print(f"  Scroll backend: {scroller.backend}")
    print(f"  Invert: {args.invert}")
    print(f"  Overlay: {show_overlay}")
    print("\nReady! Show scroll gesture (index + middle fingers) to begin scrolling.")
    print("Press 'q' or ESC to quit.\n")

    # ── Optional initial calibration ──
    if args.calibrate:
        run_calibration(cap, tracker, gesture_det, calibration, duration=5.0)

    # ── Main processing loop ──
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

            gesture = "NONE"
            velocity = 0.0
            finger_y = 0.0
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

                elif gesture == GESTURE_SCROLL:
                    # ── Active scrolling ──
                    velocity = motion.update(hand_result)
                    finger_y = motion.finger_y
                    scroller.scroll(velocity)
                else:
                    # Gesture lost — reset motion state to prevent jump on re-entry
                    motion.reset()
                    scroller.reset()
            else:
                # No hand — reset everything
                motion.reset()
                scroller.reset()

            # ── Debug overlay ──
            if show_overlay:
                overlay.draw(frame, current_fps, gesture, finger_y,
                             velocity, calibrating, hand_result, calibration)

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
