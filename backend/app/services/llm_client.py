"""Claude API wrapper for LLM-based market agents."""

import json
import asyncio
import logging
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

HAIKU_MODEL = "claude-haiku-4-5-20251001"
SONNET_MODEL = "claude-sonnet-4-5-20250514"


class LLMClient:
    """Wrapper around Anthropic SDK for market agent decisions."""

    def __init__(
        self,
        model: str = HAIKU_MODEL,
        max_retries: int = 3,
        timeout: float = 10.0,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = anthropic.Anthropic()
        self._async_client = anthropic.AsyncAnthropic()
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
        """Synchronous chat call. Returns response text."""
        model = model or self.model
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        for attempt in range(self.max_retries):
            try:
                response = self._client.messages.create(**kwargs)
                self._call_count += 1
                return response.content[0].text
            except (anthropic.RateLimitError, anthropic.APIConnectionError) as e:
                if attempt == self.max_retries - 1:
                    raise
                wait = 2 ** attempt
                logger.warning(f"API error (attempt {attempt + 1}): {e}. Retrying in {wait}s")
                import time
                time.sleep(wait)
            except anthropic.APIError as e:
                logger.error(f"API error: {e}")
                raise

        return ""  # Should not reach here

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

    async def achat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 200,
        system: str | None = None,
    ) -> str:
        """Async chat call. Returns response text."""
        model = model or self.model
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        for attempt in range(self.max_retries):
            try:
                response = await self._async_client.messages.create(**kwargs)
                self._call_count += 1
                return response.content[0].text
            except (anthropic.RateLimitError, anthropic.APIConnectionError) as e:
                if attempt == self.max_retries - 1:
                    raise
                wait = 2 ** attempt
                logger.warning(f"API error (attempt {attempt + 1}): {e}. Retrying in {wait}s")
                await asyncio.sleep(wait)
            except anthropic.APIError as e:
                logger.error(f"API error: {e}")
                raise

        return ""

    async def achat_json(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 200,
        system: str | None = None,
    ) -> dict:
        """Async chat call that parses response as JSON."""
        text = await self.achat(messages, model, temperature, max_tokens, system)
        return _parse_json(text)

    async def batch_achat_json(
        self,
        prompts: list[list[dict[str, str]]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 200,
        system: str | None = None,
        max_concurrent: int = 20,
    ) -> list[dict]:
        """
        Batch multiple async chat_json calls with concurrency control.

        Args:
            prompts: List of message lists.
            max_concurrent: Maximum concurrent API calls.

        Returns:
            List of parsed JSON responses.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _call(messages: list[dict[str, str]]) -> dict:
            async with semaphore:
                return await self.achat_json(messages, model, temperature, max_tokens, system)

        tasks = [_call(msgs) for msgs in prompts]
        return await asyncio.gather(*tasks)


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
        for part in parts[1::2]:  # odd-indexed parts are inside code blocks
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
