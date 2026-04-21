"""
llm_intent.py — Gemma-based intent classification (Tier 2, opt-in).

Uses llama-cpp-python to run a quantised Gemma-2B GGUF model locally
for more nuanced voice command understanding.  Runs inference in a
background thread and posts results to a queue.

This module is optional — the system works without it using the fast
regex parser in intent_parser.py.  Enable with --use-llm flag.

Requirements:
    pip install llama-cpp-python
    Download a Gemma GGUF model (e.g., gemma-2b-it-q4_k_m.gguf)

Usage:
    llm = LLMIntent(model_path="path/to/gemma-2b-it-q4_k_m.gguf")
    llm.start()
    llm.submit("scroll faster")
    result = llm.get_result(timeout=0.5)  # VoiceIntent or None
"""

import json
import queue
import re
import threading
import time
import logging
import os
from pathlib import Path

from intent_parser import VoiceIntent

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = Path.home() / ".cache" / "autoscroll" / "models"

# Prompt template for Gemma to produce structured JSON
GEMMA_PROMPT_TEMPLATE = """<start_of_turn>user
You are a voice command classifier for a gesture-controlled computer interface.
The user controls their computer with hand gestures and voice commands.

Given the user's spoken command, output ONLY a valid JSON object with these exact fields:
- "intent": one of ["click", "double_click", "scroll", "open", "drag", "drop", "stop", "none"]
- "modifier": an optional qualifier such as "faster", "slower", "up", "down", "here", or null
- "target": "pointed_element" if the user refers to something on screen, otherwise null

Rules:
- If the command is unclear or not a valid command, set intent to "none"
- "open this/that" means the user wants to click on what they're pointing at
- "drag" means initiate a drag operation, "drop" or "release" means end it
- "stop" means stop the current action (usually scrolling)
- Only output the JSON object, nothing else

User command: "{text}"
<end_of_turn>
<start_of_turn>model
"""


class LLMIntent:
    """
    Gemma-based intent classifier running locally via llama-cpp-python.

    Parameters
    ----------
    model_path : str | None
        Path to the GGUF model file.  If None, looks in the default
        cache directory.
    n_ctx : int
        Context window size.  256 is enough for our short prompts.
    n_gpu_layers : int
        Number of layers to offload to GPU.  -1 = all layers (requires
        CUDA build of llama-cpp-python).  0 = CPU only.
    max_tokens : int
        Maximum tokens to generate.  Our JSON responses are ~50 tokens.
    timeout_sec : float
        Maximum seconds to wait for LLM inference before falling back
        to the regex parser result.
    """

    def __init__(
        self,
        model_path: str | None = None,
        n_ctx: int = 256,
        n_gpu_layers: int = -1,
        max_tokens: int = 80,
        timeout_sec: float = 0.5,
    ):
        self._model_path = model_path
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._max_tokens = max_tokens
        self._timeout_sec = timeout_sec

        self._llm = None
        self._running = False
        self._thread: threading.Thread | None = None

        # Input queue: (text, timestamp)
        self._input_queue: queue.Queue[tuple[str, float]] = queue.Queue(maxsize=5)
        # Output queue: VoiceIntent
        self._output_queue: queue.Queue[VoiceIntent] = queue.Queue(maxsize=5)

        # Diagnostics
        self.is_active: bool = False
        self.inferences_done: int = 0
        self.avg_latency_ms: float = 0.0
        self._total_latency: float = 0.0

    def _find_model(self) -> str | None:
        """Locate a GGUF model file."""
        if self._model_path and os.path.isfile(self._model_path):
            return self._model_path

        # Search default cache directory
        if _DEFAULT_MODEL_DIR.is_dir():
            for f in _DEFAULT_MODEL_DIR.glob("*.gguf"):
                logger.info(f"Found GGUF model: {f}")
                return str(f)

        # Search current directory
        cwd = Path.cwd()
        for f in cwd.glob("*.gguf"):
            logger.info(f"Found GGUF model in cwd: {f}")
            return str(f)

        return None

    def start(self) -> bool:
        """
        Load the model and start the inference thread.
        Returns True on success, False if model not found or llama_cpp unavailable.
        """
        if self._running:
            return True

        try:
            from llama_cpp import Llama
        except ImportError:
            logger.warning(
                "llama-cpp-python is not installed. LLM intent disabled.\n"
                "Install with:  pip install llama-cpp-python\n"
                "For GPU support: CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python"
            )
            return False

        model_path = self._find_model()
        if model_path is None:
            logger.warning(
                f"No GGUF model found. LLM intent disabled.\n"
                f"Download a Gemma model and place it in: {_DEFAULT_MODEL_DIR}\n"
                f"Or specify path with --llm-model-path"
            )
            return False

        try:
            logger.info(f"Loading LLM model: {model_path}")
            self._llm = Llama(
                model_path=model_path,
                n_ctx=self._n_ctx,
                n_gpu_layers=self._n_gpu_layers,
                verbose=False,
            )
            logger.info("LLM model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            return False

        self._running = True
        self.is_active = True
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()
        logger.info("LLM intent engine started.")
        return True

    def _inference_loop(self):
        """Background loop: process submitted text through the LLM."""
        while self._running:
            try:
                text, submit_time = self._input_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            start = time.perf_counter()
            intent = self._run_inference(text)
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Update latency stats
            self.inferences_done += 1
            self._total_latency += elapsed_ms
            self.avg_latency_ms = self._total_latency / self.inferences_done

            logger.debug(
                f"LLM inference: '{text}' → {intent.intent} "
                f"({elapsed_ms:.0f} ms)"
            )

            try:
                self._output_queue.put_nowait(intent)
            except queue.Full:
                try:
                    self._output_queue.get_nowait()
                    self._output_queue.put_nowait(intent)
                except queue.Empty:
                    pass

    def _run_inference(self, text: str) -> VoiceIntent:
        """Run a single LLM inference and parse the JSON output."""
        prompt = GEMMA_PROMPT_TEMPLATE.format(text=text)
        now = time.perf_counter()

        try:
            output = self._llm(
                prompt,
                max_tokens=self._max_tokens,
                temperature=0.0,      # deterministic
                top_p=1.0,
                stop=["<end_of_turn>", "\n\n"],
            )

            response_text = output["choices"][0]["text"].strip()
            return self._parse_llm_response(response_text, text, now)

        except Exception as e:
            logger.debug(f"LLM inference error: {e}")
            return VoiceIntent(
                intent="none", confidence=0.0,
                raw_text=text, timestamp=now,
            )

    def _parse_llm_response(self, response: str, original_text: str,
                             timestamp: float) -> VoiceIntent:
        """Extract structured intent from LLM's JSON response."""
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)

            intent = data.get("intent", "none")
            valid_intents = {
                "click", "double_click", "scroll", "open",
                "drag", "drop", "stop", "none"
            }
            if intent not in valid_intents:
                intent = "none"

            return VoiceIntent(
                intent=intent,
                modifier=data.get("modifier"),
                target=data.get("target"),
                confidence=0.85,  # LLM results get moderate confidence
                raw_text=original_text,
                timestamp=timestamp,
            )

        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            logger.debug(f"Failed to parse LLM response: {response!r} — {e}")
            return VoiceIntent(
                intent="none", confidence=0.0,
                raw_text=original_text, timestamp=timestamp,
            )

    def submit(self, text: str):
        """Submit text for LLM classification (non-blocking)."""
        if not self._running or not text.strip():
            return
        try:
            self._input_queue.put_nowait((text.strip(), time.perf_counter()))
        except queue.Full:
            pass  # drop if overloaded

    def get_result(self, timeout: float = 0.0) -> VoiceIntent | None:
        """
        Retrieve the next LLM classification result.
        Returns VoiceIntent or None if not ready.
        """
        try:
            if timeout <= 0:
                return self._output_queue.get_nowait()
            return self._output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        """Stop the inference thread and release the model."""
        self._running = False
        self.is_active = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._llm = None
        logger.info(
            f"LLM intent stopped. "
            f"Inferences: {self.inferences_done}, "
            f"Avg latency: {self.avg_latency_ms:.0f} ms"
        )
