"""
hand_tracker.py — MediaPipe HandLandmarker (Tasks API) wrapper.

Uses the new mp.tasks.vision.HandLandmarker which replaced the deprecated
mp.solutions.hands in mediapipe ≥ 0.10.14.

Requires the model file:  hand_landmarker.task
Download: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


@dataclass(slots=True)
class HandResult:
    """Compact representation of a single hand detection."""
    landmarks: list[tuple[float, float, float]]   # 21 × (x, y, z) normalised [0-1]
    handedness: str                                 # "Left" | "Right"
    score: float                                    # detection confidence


class HandTracker:
    """
    Wrapper around mp.tasks.vision.HandLandmarker.
    Runs in VIDEO mode for sequential frame processing with timestamp tracking.
    Produces landmark lists at ≥30 FPS on 640×480 input.
    """

    # Landmark indices (MediaPipe convention)
    WRIST = 0
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
    RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

    # Default model path (same directory as this script)
    _DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "hand_landmarker.task")

    def __init__(self, max_hands: int = 1, min_detection_conf: float = 0.7,
                 min_tracking_conf: float = 0.6, model_path: str | None = None):
        """
        Args:
            max_hands: Maximum number of hands to detect.
            min_detection_conf: Minimum confidence for hand detection.
            min_tracking_conf: Minimum confidence for landmark tracking.
            model_path: Path to hand_landmarker.task file.
        """
        model = model_path or self._DEFAULT_MODEL
        if not os.path.isfile(model):
            raise FileNotFoundError(
                f"Hand landmarker model not found at: {model}\n"
                "Download it with:\n"
                "  curl -L -o hand_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/latest/hand_landmarker.task"
            )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model),
            running_mode=RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_conf,
            min_hand_presence_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._frame_ts_ms: int = 0  # monotonic timestamp for VIDEO mode

    def process(self, frame_bgr: np.ndarray) -> list[HandResult]:
        """
        Process a BGR frame and return detected hands.
        Converts to MediaPipe Image internally.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # VIDEO mode requires monotonically increasing timestamps
        self._frame_ts_ms += 33  # ~30 FPS interval
        results = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)

        if not results.hand_landmarks:
            return []

        hands: list[HandResult] = []
        for i, hand_lms in enumerate(results.hand_landmarks):
            lms = [(lm.x, lm.y, lm.z) for lm in hand_lms]

            # Extract handedness
            if results.handedness and i < len(results.handedness):
                h_cat = results.handedness[i]
                label = h_cat[0].category_name if h_cat else "Unknown"
                score = h_cat[0].score if h_cat else 0.0
            else:
                label = "Unknown"
                score = 0.0

            hands.append(HandResult(landmarks=lms, handedness=label, score=score))

        return hands

    def draw_landmarks_on_frame(self, frame_bgr: np.ndarray,
                                 hand: HandResult) -> np.ndarray:
        """Draw landmarks and connections on frame for debug visualisation."""
        h, w = frame_bgr.shape[:2]

        # MediaPipe hand connections (pairs of landmark indices)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),     # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),     # Index
            (0, 9), (9, 10), (10, 11), (11, 12), # Middle
            (0, 13), (13, 14), (14, 15), (15, 16), # Ring
            (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
            (5, 9), (9, 13), (13, 17),            # Palm
        ]

        # Draw connections
        for start_idx, end_idx in connections:
            sx, sy = hand.landmarks[start_idx][:2]
            ex, ey = hand.landmarks[end_idx][:2]
            pt1 = (int(sx * w), int(sy * h))
            pt2 = (int(ex * w), int(ey * h))
            cv2.line(frame_bgr, pt1, pt2, (144, 238, 144), 2)

        # Draw landmarks
        for lm in hand.landmarks:
            cx, cy = int(lm[0] * w), int(lm[1] * h)
            cv2.circle(frame_bgr, (cx, cy), 4, (0, 255, 0), -1)
            cv2.circle(frame_bgr, (cx, cy), 6, (0, 180, 0), 1)

        return frame_bgr

    def close(self):
        self._landmarker.close()
