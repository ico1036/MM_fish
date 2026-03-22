"""Tests for LLM client wrapper."""

from unittest.mock import MagicMock, patch, AsyncMock
import pytest

from app.services.llm_client import LLMClient, _parse_json, HAIKU_MODEL


class TestParseJson:
    def test_parse_plain_json(self):
        result = _parse_json('{"action": "BUY", "quantity": 0.1}')
        assert result["action"] == "BUY"
        assert result["quantity"] == 0.1

    def test_parse_json_in_code_block(self):
        text = '```json\n{"action": "SELL"}\n```'
        result = _parse_json(text)
        assert result["action"] == "SELL"

    def test_parse_json_with_surrounding_text(self):
        text = 'Here is my decision: {"action": "HOLD", "reason": "uncertain"} based on analysis.'
        result = _parse_json(text)
        assert result["action"] == "HOLD"

    def test_parse_invalid_json_returns_empty(self):
        result = _parse_json("This is not JSON at all")
        assert result == {}

    def test_parse_empty_string(self):
        result = _parse_json("")
        assert result == {}


class TestLLMClientInit:
    def test_default_model(self):
        with patch("anthropic.Anthropic"), patch("anthropic.AsyncAnthropic"):
            client = LLMClient()
            assert client.model == HAIKU_MODEL

    def test_custom_model(self):
        with patch("anthropic.Anthropic"), patch("anthropic.AsyncAnthropic"):
            client = LLMClient(model="claude-sonnet-4-5-20250514")
            assert client.model == "claude-sonnet-4-5-20250514"

    def test_call_count_starts_zero(self):
        with patch("anthropic.Anthropic"), patch("anthropic.AsyncAnthropic"):
            client = LLMClient()
            assert client.call_count == 0


class TestLLMClientChat:
    def test_chat_returns_text(self):
        mock_anthropic = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"action": "BUY"}')]
        mock_anthropic.messages.create.return_value = mock_response

        with patch("anthropic.Anthropic", return_value=mock_anthropic), \
             patch("anthropic.AsyncAnthropic"):
            client = LLMClient()
            result = client.chat([{"role": "user", "content": "test"}])
            assert result == '{"action": "BUY"}'
            assert client.call_count == 1

    def test_chat_json_returns_dict(self):
        mock_anthropic = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"action": "SELL", "quantity": 0.5}')]
        mock_anthropic.messages.create.return_value = mock_response

        with patch("anthropic.Anthropic", return_value=mock_anthropic), \
             patch("anthropic.AsyncAnthropic"):
            client = LLMClient()
            result = client.chat_json([{"role": "user", "content": "test"}])
            assert result["action"] == "SELL"
            assert result["quantity"] == 0.5

    def test_chat_passes_system_prompt(self):
        mock_anthropic = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="ok")]
        mock_anthropic.messages.create.return_value = mock_response

        with patch("anthropic.Anthropic", return_value=mock_anthropic), \
             patch("anthropic.AsyncAnthropic"):
            client = LLMClient()
            client.chat(
                [{"role": "user", "content": "test"}],
                system="You are a trader"
            )
            call_kwargs = mock_anthropic.messages.create.call_args
            assert call_kwargs.kwargs["system"] == "You are a trader"


class TestLLMClientAsync:
    @pytest.mark.asyncio
    async def test_achat_returns_text(self):
        mock_async = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"action": "HOLD"}')]
        mock_async.messages.create.return_value = mock_response

        with patch("anthropic.Anthropic"), \
             patch("anthropic.AsyncAnthropic", return_value=mock_async):
            client = LLMClient()
            result = await client.achat([{"role": "user", "content": "test"}])
            assert result == '{"action": "HOLD"}'

    @pytest.mark.asyncio
    async def test_achat_json_returns_dict(self):
        mock_async = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"action": "BUY", "price": 100.0}')]
        mock_async.messages.create.return_value = mock_response

        with patch("anthropic.Anthropic"), \
             patch("anthropic.AsyncAnthropic", return_value=mock_async):
            client = LLMClient()
            result = await client.achat_json([{"role": "user", "content": "test"}])
            assert result["action"] == "BUY"
            assert result["price"] == 100.0

    @pytest.mark.asyncio
    async def test_batch_achat_json(self):
        mock_async = AsyncMock()
        call_count = 0

        async def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            mock_resp = MagicMock()
            mock_resp.content = [MagicMock(text=f'{{"id": {call_count}}}')]
            return mock_resp

        mock_async.messages.create = mock_create

        with patch("anthropic.Anthropic"), \
             patch("anthropic.AsyncAnthropic", return_value=mock_async):
            client = LLMClient()
            prompts = [
                [{"role": "user", "content": f"prompt {i}"}]
                for i in range(5)
            ]
            results = await client.batch_achat_json(prompts)
            assert len(results) == 5
            assert all("id" in r for r in results)
