"""Claude CLI wrapper for LLM-based market agents. Uses claude -p (OAuth)."""

import json
import logging
import subprocess

logger = logging.getLogger(__name__)

HAIKU_MODEL = "claude-haiku-4-5"
SONNET_MODEL = "claude-sonnet-4-6"


class LLMClient:
    """Wrapper around Claude CLI for market agent decisions. Uses OAuth via claude -p."""

    def __init__(
        self,
        model: str = HAIKU_MODEL,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 200,
        system: str | None = None,
    ) -> str:
        """Chat call via claude -p subprocess. Returns response text."""
        model = model or self.model

        # Build prompt from messages
        prompt = ""
        if system:
            prompt += f"[System: {system}]\n\n"
        for msg in messages:
            prompt += msg["content"] + "\n"

        cmd = ["claude", "-p", prompt, "--model", model, "--max-turns", "1"]

        for attempt in range(self.max_retries):
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode != 0:
                    logger.warning(f"claude -p failed (attempt {attempt + 1}): {result.stderr[:200]}")
                    if attempt == self.max_retries - 1:
                        raise RuntimeError(f"claude -p failed: {result.stderr[:500]}")
                    continue

                self._call_count += 1
                return result.stdout.strip()
            except subprocess.TimeoutExpired:
                logger.warning(f"claude -p timed out (attempt {attempt + 1})")
                if attempt == self.max_retries - 1:
                    raise

        return ""

    def chat_json(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 200,
        system: str | None = None,
    ) -> dict:
        """Chat call that parses response as JSON."""
        text = self.chat(messages, model, temperature, max_tokens, system)
        return _parse_json(text)


def _parse_json(text: str) -> dict:
    """Extract JSON from LLM response text, handling markdown code blocks."""
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    if "```" in text:
        parts = text.split("```")
        for part in parts[1::2]:
            clean = part.strip()
            if clean.startswith("json"):
                clean = clean[4:].strip()
            try:
                return json.loads(clean)
            except json.JSONDecodeError:
                continue

    # Try finding JSON object in text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    logger.warning(f"Failed to parse JSON from: {text[:100]}...")
    return {}
