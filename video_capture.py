"""
video_capture.py — Efficient webcam capture with configurable resolution and threaded read-ahead.
"""

import cv2
import threading
import time


class VideoCapture:
    """
    Wraps cv2.VideoCapture with a dedicated reader thread so the main
    loop never blocks on I/O.  Always exposes the *latest* frame
    (drops stale frames automatically).
    """

    def __init__(self, source: int = 0, width: int = 640, height: int = 480,
                 fps_target: int = 30):
        self._cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)  # DirectShow on Windows
        if not self._cap.isOpened():
            # Fallback without DirectShow (Linux / macOS)
            self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera source {source}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps_target)
        # Reduce internal buffer to 1 so we always get latest frame
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._frame = None
        self._ret = False
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> "VideoCapture":
        """Begin background frame acquisition."""
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        # Wait until first frame is available
        deadline = time.monotonic() + 3.0
        while self._frame is None and time.monotonic() < deadline:
            time.sleep(0.01)
        return self

    def _reader(self):
        while self._running:
            ret, frame = self._cap.read()
            with self._lock:
                self._ret = ret
                self._frame = frame

    def read(self):
        """Return (success: bool, frame: np.ndarray | None)."""
        with self._lock:
            return self._ret, self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._cap.release()

    def __del__(self):
        self.stop()
