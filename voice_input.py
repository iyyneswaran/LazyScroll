"""
voice_input.py — Real-time microphone audio capture for the voice pipeline.

Captures 16 kHz mono PCM audio from the default microphone using sounddevice.
Runs in a dedicated daemon thread with callback-based streaming to avoid
blocking the main CV loop.

Audio chunks are pushed into a thread-safe queue for consumption by
speech_to_text.py.

Usage:
    mic = VoiceInput(sample_rate=16000, block_duration_ms=100)
    mic.start()
    while True:
        chunk = mic.get_audio_chunk(timeout=0.1)
        if chunk is not None:
            process(chunk)  # numpy int16 array
    mic.stop()
"""

import queue
import threading
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Lazy-load sounddevice to degrade gracefully when not installed
_sd = None


def _ensure_sounddevice():
    """Import sounddevice on first use so the module loads even without it."""
    global _sd
    if _sd is None:
        try:
            import sounddevice as sd
            _sd = sd
        except ImportError:
            raise ImportError(
                "sounddevice is required for voice input.\n"
                "Install it with:  pip install sounddevice\n"
                "On Linux you may also need:  sudo apt install libportaudio2"
            )
    return _sd


class VoiceInput:
    """
    Callback-based microphone capture that pushes raw PCM chunks into a queue.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate in Hz.  16000 is optimal for Vosk small models.
    block_duration_ms : int
        Duration of each audio block in milliseconds.  Smaller = lower
        latency but higher CPU overhead.  100 ms is a good balance.
    max_queue_size : int
        Maximum number of audio chunks buffered in the queue.  If the
        consumer falls behind, oldest chunks are silently dropped.
    device : int | str | None
        Sounddevice device index or name.  None = system default mic.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        block_duration_ms: int = 100,
        max_queue_size: int = 50,
        device: int | str | None = None,
    ):
        self.sample_rate = sample_rate
        self.block_duration_ms = block_duration_ms
        self.block_size = int(sample_rate * block_duration_ms / 1000)
        self.device = device

        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queue_size)
        self._stream = None
        self._running = False
        self._lock = threading.Lock()

        # Diagnostics
        self.chunks_captured: int = 0
        self.chunks_dropped: int = 0
        self.is_active: bool = False

    def _audio_callback(self, indata: np.ndarray, frames: int,
                        time_info, status):
        """Called by sounddevice for each audio block (runs in audio thread)."""
        if status:
            logger.debug(f"Audio status: {status}")

        # Convert to int16 (Vosk expects int16 PCM)
        audio_chunk = (indata[:, 0] * 32767).astype(np.int16)

        try:
            self._queue.put_nowait(audio_chunk)
            self.chunks_captured += 1
        except queue.Full:
            # Drop oldest chunk and insert new one
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(audio_chunk)
                self.chunks_dropped += 1
            except queue.Empty:
                pass

    def start(self) -> bool:
        """
        Start capturing audio from the microphone.

        Returns True if started successfully, False if microphone is
        unavailable (system continues without voice).
        """
        with self._lock:
            if self._running:
                return True

            try:
                sd = _ensure_sounddevice()

                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    blocksize=self.block_size,
                    channels=1,
                    dtype="float32",
                    device=self.device,
                    callback=self._audio_callback,
                )
                self._stream.start()
                self._running = True
                self.is_active = True
                logger.info(
                    f"Microphone started: {self.sample_rate} Hz, "
                    f"{self.block_duration_ms} ms blocks, "
                    f"device={self.device or 'default'}"
                )
                return True

            except Exception as e:
                logger.warning(f"Failed to start microphone: {e}")
                self.is_active = False
                return False

    def stop(self):
        """Stop the audio stream and clean up."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            self.is_active = False

            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception as e:
                    logger.debug(f"Error closing audio stream: {e}")
                self._stream = None

            # Drain the queue
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

            logger.info(
                f"Microphone stopped. "
                f"Captured: {self.chunks_captured}, "
                f"Dropped: {self.chunks_dropped}"
            )

    def get_audio_chunk(self, timeout: float = 0.05) -> np.ndarray | None:
        """
        Retrieve the next audio chunk from the queue.

        Parameters
        ----------
        timeout : float
            Maximum seconds to wait for a chunk.  Use 0 for non-blocking.

        Returns
        -------
        numpy.ndarray (int16) or None if no chunk is available.
        """
        try:
            if timeout <= 0:
                return self._queue.get_nowait()
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def queue_size(self) -> int:
        """Number of audio chunks currently buffered."""
        return self._queue.qsize()

    @property
    def running(self) -> bool:
        return self._running

    def list_devices(self) -> str:
        """List available audio input devices (for debug / setup)."""
        sd = _ensure_sounddevice()
        return str(sd.query_devices())
