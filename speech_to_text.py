"""
speech_to_text.py — Streaming offline speech-to-text using Vosk.

Consumes int16 PCM audio chunks from VoiceInput's queue and produces
recognised text.  Runs the Vosk recogniser in a dedicated daemon thread
so the main CV loop is never blocked.

Model management:
    On first run, automatically downloads the small English Vosk model
    (~40 MB) to a local cache directory.  Set VOSK_MODEL_PATH env var
    to use a pre-downloaded or larger model.

Usage:
    from voice_input import VoiceInput
    from speech_to_text import SpeechToText

    mic = VoiceInput()
    stt = SpeechToText(voice_input=mic)
    mic.start()
    stt.start()

    text = stt.get_text()  # non-blocking, returns str or None
"""

import json
import os
import queue
import threading
import time
import logging
import zipfile
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

# Default model info
_MODEL_NAME = "vosk-model-small-en-us-0.15"
_MODEL_URL = f"https://alphacephei.com/vosk/models/{_MODEL_NAME}.zip"
_CACHE_DIR = Path.home() / ".cache" / "autoscroll" / "vosk"


class SpeechToText:
    """
    Streaming speech-to-text engine backed by Vosk.

    Parameters
    ----------
    voice_input : VoiceInput
        The audio source to consume chunks from.
    model_path : str | None
        Path to a Vosk model directory.  If None, auto-downloads the
        small English model.
    sample_rate : int
        Must match VoiceInput's sample rate.
    partial_results : bool
        If True, emit partial (in-progress) results for lower latency.
        Partial results are prefixed with "..." in the output.
    """

    def __init__(
        self,
        voice_input,  # VoiceInput instance
        model_path: str | None = None,
        sample_rate: int = 16000,
        partial_results: bool = True,
    ):
        self._voice_input = voice_input
        self._sample_rate = sample_rate
        self._partial_results = partial_results

        # Resolve model path
        self._model_path = model_path or os.environ.get("VOSK_MODEL_PATH")
        if self._model_path is None:
            self._model_path = str(_CACHE_DIR / _MODEL_NAME)

        # Output queue for recognised text
        self._text_queue: queue.Queue[str] = queue.Queue(maxsize=20)

        # State
        self._running = False
        self._thread: threading.Thread | None = None
        self._recogniser = None

        # Diagnostics
        self.last_text: str = ""
        self.last_partial: str = ""
        self.texts_produced: int = 0
        self.is_active: bool = False
        self._last_text_time: float = 0.0

    def _ensure_model(self):
        """Download the Vosk model if not present."""
        if os.path.isdir(self._model_path):
            logger.info(f"Vosk model found: {self._model_path}")
            return

        logger.info(f"Vosk model not found at {self._model_path}")
        logger.info(f"Downloading {_MODEL_NAME} (~40 MB)...")

        os.makedirs(_CACHE_DIR, exist_ok=True)
        zip_path = _CACHE_DIR / f"{_MODEL_NAME}.zip"

        try:
            # Download with progress
            def _report(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    pct = min(100.0, downloaded / total_size * 100)
                    print(f"\r  Downloading Vosk model: {pct:.0f}%", end="", flush=True)

            urllib.request.urlretrieve(_MODEL_URL, str(zip_path), _report)
            print()  # newline after progress

            # Extract
            logger.info(f"Extracting model to {_CACHE_DIR}...")
            with zipfile.ZipFile(str(zip_path), "r") as zf:
                zf.extractall(str(_CACHE_DIR))

            # Clean up zip
            zip_path.unlink(missing_ok=True)
            logger.info("Vosk model ready.")

        except Exception as e:
            logger.error(f"Failed to download Vosk model: {e}")
            raise RuntimeError(
                f"Could not download Vosk model from {_MODEL_URL}\n"
                f"Please download manually and set VOSK_MODEL_PATH.\n"
                f"Error: {e}"
            )

    def start(self) -> bool:
        """
        Initialise Vosk and start the recognition thread.
        Returns True on success, False if Vosk is unavailable.
        """
        if self._running:
            return True

        try:
            import vosk
            vosk.SetLogLevel(-1)  # Suppress Vosk's noisy logs
        except ImportError:
            logger.warning(
                "vosk is not installed. Voice recognition disabled.\n"
                "Install with:  pip install vosk"
            )
            return False

        try:
            self._ensure_model()
            model = vosk.Model(self._model_path)
            self._recogniser = vosk.KaldiRecognizer(model, self._sample_rate)
            self._recogniser.SetWords(True)  # Include word timings
        except Exception as e:
            logger.error(f"Failed to initialise Vosk: {e}")
            return False

        self._running = True
        self.is_active = True
        self._thread = threading.Thread(target=self._recognition_loop, daemon=True)
        self._thread.start()
        logger.info("Speech-to-text started (Vosk).")
        return True

    def _recognition_loop(self):
        """Main loop: consume audio chunks and feed to Vosk."""
        while self._running:
            chunk = self._voice_input.get_audio_chunk(timeout=0.1)
            if chunk is None:
                continue

            # Feed raw bytes to Vosk
            data = chunk.tobytes()

            if self._recogniser.AcceptWaveform(data):
                # Final result for this utterance
                result = json.loads(self._recogniser.Result())
                text = result.get("text", "").strip()
                if text:
                    self._emit_text(text, is_partial=False)
            elif self._partial_results:
                # Partial (in-progress) result
                partial = json.loads(self._recogniser.PartialResult())
                partial_text = partial.get("partial", "").strip()
                if partial_text:
                    self.last_partial = partial_text

    def _emit_text(self, text: str, is_partial: bool):
        """Push recognised text to the output queue."""
        now = time.perf_counter()

        # Deduplicate: ignore if same text within 0.5 s
        if text == self.last_text and (now - self._last_text_time) < 0.5:
            return

        self.last_text = text
        self._last_text_time = now
        self.texts_produced += 1

        try:
            self._text_queue.put_nowait(text)
        except queue.Full:
            # Drop oldest
            try:
                self._text_queue.get_nowait()
                self._text_queue.put_nowait(text)
            except queue.Empty:
                pass

    def get_text(self, timeout: float = 0.0) -> str | None:
        """
        Retrieve the next recognised text string.

        Parameters
        ----------
        timeout : float
            Max seconds to wait.  0 = non-blocking.

        Returns
        -------
        Recognised text string, or None if nothing available.
        """
        try:
            if timeout <= 0:
                return self._text_queue.get_nowait()
            return self._text_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        """Stop the recognition thread."""
        self._running = False
        self.is_active = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        # Get any final result
        if self._recogniser is not None:
            try:
                final = json.loads(self._recogniser.FinalResult())
                text = final.get("text", "").strip()
                if text:
                    self._emit_text(text, is_partial=False)
            except Exception:
                pass

        logger.info(
            f"Speech-to-text stopped. "
            f"Texts produced: {self.texts_produced}"
        )

    @property
    def running(self) -> bool:
        return self._running
