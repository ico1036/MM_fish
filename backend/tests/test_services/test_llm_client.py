"""Tests for LLM client wrapper (claude -p subprocess)."""

from unittest.mock import patch, MagicMock
import subprocess
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
        client = LLMClient()
        assert client.model == HAIKU_MODEL

    def test_custom_model(self):
        client = LLMClient(model="claude-sonnet-4-6")
        assert client.model == "claude-sonnet-4-6"

    def test_call_count_starts_zero(self):
        client = LLMClient()
        assert client.call_count == 0


class TestLLMClientChat:
    def test_chat_returns_text(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"action": "BUY"}'

        with patch("subprocess.run", return_value=mock_result):
            client = LLMClient()
            result = client.chat([{"role": "user", "content": "test"}])
            assert result == '{"action": "BUY"}'
            assert client.call_count == 1

    def test_chat_json_returns_dict(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"action": "SELL", "quantity": 0.5}'

        with patch("subprocess.run", return_value=mock_result):
            client = LLMClient()
            result = client.chat_json([{"role": "user", "content": "test"}])
            assert result["action"] == "SELL"
            assert result["quantity"] == 0.5

    def test_chat_retries_on_failure(self):
        fail_result = MagicMock(returncode=1, stderr="error")
        ok_result = MagicMock(returncode=0, stdout='{"action": "HOLD"}')

        with patch("subprocess.run", side_effect=[fail_result, ok_result]):
            client = LLMClient()
            result = client.chat([{"role": "user", "content": "test"}])
            assert result == '{"action": "HOLD"}'

    def test_chat_raises_after_max_retries(self):
        fail_result = MagicMock(returncode=1, stderr="error")

        with patch("subprocess.run", return_value=fail_result):
            client = LLMClient(max_retries=2)
            with pytest.raises(RuntimeError, match="claude -p failed"):
                client.chat([{"role": "user", "content": "test"}])

    def test_chat_includes_system_in_prompt(self):
        mock_result = MagicMock(returncode=0, stdout="ok")

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            client = LLMClient()
            client.chat(
                [{"role": "user", "content": "test"}],
                system="You are a trader"
            )
            call_args = mock_run.call_args
            prompt = call_args[0][0][2]  # cmd[2] is the prompt
            assert "You are a trader" in prompt
