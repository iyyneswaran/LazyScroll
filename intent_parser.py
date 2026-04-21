"""
intent_parser.py — Fast, deterministic voice command parser.

Tier 1 parser: uses regex and keyword matching to convert recognised
speech text into structured VoiceIntent objects in <1 ms.  No ML
dependencies — pure stdlib.

Handles:
    "click here"     → intent=click, modifier=here
    "double click"   → intent=double_click
    "open this"      → intent=open, target=pointed_element
    "scroll faster"  → intent=scroll, modifier=faster
    "scroll slower"  → intent=scroll, modifier=slower
    "scroll up"      → intent=scroll, modifier=up
    "scroll down"    → intent=scroll, modifier=down
    "stop scrolling" → intent=stop
    "stop"           → intent=stop
    "drag this"      → intent=drag, target=pointed_element
    "drop" / "release" → intent=drop

Fuzzy matching: uses substring / token matching to handle noisy STT
output (e.g. "click hear" matches "click here").

Usage:
    parser = IntentParser()
    intent = parser.parse("scroll faster please")
    # VoiceIntent(intent='scroll', modifier='faster', ...)
"""

import re
import time
from dataclasses import dataclass, field


@dataclass(slots=True)
class VoiceIntent:
    """Structured representation of a parsed voice command."""
    intent: str = "none"           # click, double_click, scroll, open, drag, drop, stop, none
    modifier: str | None = None    # faster, slower, up, down, here, etc.
    target: str | None = None      # pointed_element, etc.
    confidence: float = 0.0        # 0.0–1.0
    timestamp: float = field(default_factory=time.perf_counter)
    raw_text: str = ""             # original recognised text

    def is_valid(self) -> bool:
        """True if this intent is actionable (not 'none')."""
        return self.intent != "none" and self.confidence > 0.3

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "modifier": self.modifier,
            "target": self.target,
            "confidence": self.confidence,
            "raw_text": self.raw_text,
        }


# ─────────────────────── Command pattern definitions ───────────────────────

# Each pattern: (compiled_regex, intent, modifier_extractor, target, confidence)
# modifier_extractor: None for static modifier, or a callable(match) → str|None

_COMMAND_PATTERNS: list[tuple[re.Pattern, str, str | None, str | None, float]] = []


def _pat(pattern: str, intent: str, modifier: str | None = None,
         target: str | None = None, confidence: float = 1.0):
    """Helper to register a command pattern."""
    _COMMAND_PATTERNS.append((
        re.compile(pattern, re.IGNORECASE),
        intent, modifier, target, confidence,
    ))


# ── Click commands ──
_pat(r"\bdouble\s*click\b", "double_click", None, "pointed_element", 1.0)
_pat(r"\bclick\s+here\b", "click", "here", "pointed_element", 1.0)
_pat(r"\bclick\s+this\b", "click", "here", "pointed_element", 0.95)
_pat(r"\bclick\s+that\b", "click", "here", "pointed_element", 0.90)
_pat(r"\bclick\b", "click", None, "pointed_element", 0.85)
_pat(r"\bpress\s+here\b", "click", "here", "pointed_element", 0.90)
_pat(r"\bpress\b", "click", None, "pointed_element", 0.75)
_pat(r"\btap\s+here\b", "click", "here", "pointed_element", 0.85)
_pat(r"\btap\b", "click", None, "pointed_element", 0.70)

# ── Open commands ──
_pat(r"\bopen\s+this\b", "open", None, "pointed_element", 1.0)
_pat(r"\bopen\s+that\b", "open", None, "pointed_element", 0.95)
_pat(r"\bopen\s+here\b", "open", None, "pointed_element", 0.90)
_pat(r"\bopen\s+it\b", "open", None, "pointed_element", 0.90)
_pat(r"\bopen\b", "open", None, "pointed_element", 0.80)
_pat(r"\blaunch\b", "open", None, "pointed_element", 0.75)
_pat(r"\bselect\s+this\b", "open", None, "pointed_element", 0.80)
_pat(r"\bselect\b", "open", None, "pointed_element", 0.65)

# ── Scroll commands ──
_pat(r"\bscroll\s+(?:much\s+)?faster\b", "scroll", "faster", None, 1.0)
_pat(r"\bscroll\s+(?:much\s+)?slower\b", "scroll", "slower", None, 1.0)
_pat(r"\bscroll\s+(?:speed\s+)?up\b", "scroll", "faster", None, 0.85)
_pat(r"\bfaster\b", "scroll", "faster", None, 0.70)
_pat(r"\bslower\b", "scroll", "slower", None, 0.70)
_pat(r"\bscroll\s+up\b", "scroll", "up", None, 0.95)
_pat(r"\bscroll\s+down\b", "scroll", "down", None, 0.95)
_pat(r"\bgo\s+up\b", "scroll", "up", None, 0.80)
_pat(r"\bgo\s+down\b", "scroll", "down", None, 0.80)
_pat(r"\bpage\s+up\b", "scroll", "up", None, 0.90)
_pat(r"\bpage\s+down\b", "scroll", "down", None, 0.90)
_pat(r"\bscroll\b", "scroll", None, None, 0.60)

# ── Stop commands ──
_pat(r"\bstop\s+scroll(?:ing)?\b", "stop", None, None, 1.0)
_pat(r"\bstop\s+moving\b", "stop", None, None, 0.90)
_pat(r"\bstop\s+that\b", "stop", None, None, 0.85)
_pat(r"\bstop\b", "stop", None, None, 0.80)
_pat(r"\bhalt\b", "stop", None, None, 0.75)
_pat(r"\bfreeze\b", "stop", None, None, 0.75)
_pat(r"\bpause\b", "stop", None, None, 0.70)

# ── Drag commands ──
_pat(r"\bdrag\s+this\b", "drag", None, "pointed_element", 1.0)
_pat(r"\bdrag\s+that\b", "drag", None, "pointed_element", 0.95)
_pat(r"\bdrag\s+here\b", "drag", None, "pointed_element", 0.90)
_pat(r"\bdrag\b", "drag", None, "pointed_element", 0.80)
_pat(r"\bmove\s+this\b", "drag", None, "pointed_element", 0.75)

# ── Drop / release commands ──
_pat(r"\bdrop\s+(this|that|it|here)\b", "drop", None, None, 1.0)
_pat(r"\bdrop\b", "drop", None, None, 0.85)
_pat(r"\brelease\b", "drop", None, None, 0.90)
_pat(r"\blet\s+go\b", "drop", None, None, 0.85)


class IntentParser:
    """
    Fast regex-based voice command parser (Tier 1).

    Matches input text against a priority-ordered list of patterns.
    The first matching pattern wins (patterns are ordered from most
    specific to most general within each category).

    Parameters
    ----------
    min_confidence : float
        Minimum confidence threshold.  Intents below this are returned
        as 'none'.
    cooldown_sec : float
        Minimum interval between identical consecutive intents to
        prevent repeat-triggering from STT stuttering.
    """

    def __init__(self, min_confidence: float = 0.5, cooldown_sec: float = 1.0):
        self.min_confidence = min_confidence
        self.cooldown_sec = cooldown_sec

        self._last_intent: str = "none"
        self._last_intent_time: float = 0.0

    def parse(self, text: str) -> VoiceIntent:
        """
        Parse a recognised text string into a VoiceIntent.

        Parameters
        ----------
        text : str
            Raw text from the speech-to-text engine.

        Returns
        -------
        VoiceIntent with the best matching intent, or intent='none'
        if no command is recognised.
        """
        if not text or not text.strip():
            return VoiceIntent(raw_text=text or "")

        cleaned = text.strip().lower()
        now = time.perf_counter()

        # Try each pattern in priority order
        for regex, intent, modifier, target, confidence in _COMMAND_PATTERNS:
            if regex.search(cleaned):
                if confidence < self.min_confidence:
                    continue

                # Cooldown: suppress identical intents within window
                if (intent == self._last_intent and
                        (now - self._last_intent_time) < self.cooldown_sec):
                    return VoiceIntent(
                        intent="none", confidence=0.0,
                        raw_text=text, timestamp=now,
                    )

                self._last_intent = intent
                self._last_intent_time = now

                return VoiceIntent(
                    intent=intent,
                    modifier=modifier,
                    target=target,
                    confidence=confidence,
                    raw_text=text,
                    timestamp=now,
                )

        # No match
        return VoiceIntent(intent="none", raw_text=text, timestamp=now)

    def reset(self):
        """Reset cooldown state."""
        self._last_intent = "none"
        self._last_intent_time = 0.0
