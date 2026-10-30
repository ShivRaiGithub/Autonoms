"""A thin client for a local Ollama model with tool calling.

The rest of the code only uses `LLM.chat(messages, tools)`."""

import json
import time
from typing import Protocol

import httpx


class LLMError(Exception):
    pass


class LLM(Protocol):
    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        """Send the conversation; return the assistant message
        {"role": "assistant", "content": str, "tool_calls": [{"name", "arguments"}],
        "usage": {"prompt_tokens", "completion_tokens"}}  (usage is optional)."""


class OllamaClient:
    def __init__(self, client: httpx.Client, model: str, think: bool | None = None):
        self._client = client
        self._model = model
        self._think = think

    @classmethod
    def create(cls, base_url: str, model: str, think: bool | None = None) -> "OllamaClient":
        # Generous timeout: a 9B model on this laptop writes about 15 tokens/second.
        return cls(httpx.Client(base_url=base_url, timeout=httpx.Timeout(600, connect=5)), model, think)

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        body = {
            "model": self._model,
            "messages": [self._to_wire(m) for m in messages],
            "tools": tools,
            "stream": False,
            "options": {"temperature": 0},  # same input -> (nearly) same output
        }
        if self._think is not None:
            body["think"] = self._think
        started = time.monotonic()
        try:
            response = self._client.post("/api/chat", json=body)
            response.raise_for_status()
            data = response.json()
            message = data["message"]
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise LLMError(f"Model call failed: {exc.__class__.__name__}") from exc
        return {
            "role": "assistant",
            "content": message.get("content") or "",
            # Ollama reports exactly how many tokens it read and wrote for this call.
            "usage": {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "seconds": round(time.monotonic() - started, 2),
                "model": data.get("model", self._model),
            },
            # message["thinking"] is deliberately dropped: it costs context and
            # the model does not need to read its own earlier reasoning.
            "tool_calls": [self._tool_call(c) for c in message.get("tool_calls") or []],
        }

    @staticmethod
    def _to_wire(message: dict) -> dict:
        """Our flat tool calls -> Ollama's {"function": {...}} shape, for echoing history back."""
        message = {k: v for k, v in message.items() if k != "usage"}  # bookkeeping, not conversation
        if not message.get("tool_calls"):
            return message
        calls = [{"function": {"name": c["name"], "arguments": c["arguments"]}} for c in message["tool_calls"]]
        return {**message, "tool_calls": calls}

    @staticmethod
    def _tool_call(call: dict) -> dict:
        function = call.get("function", {})
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):  # some models send JSON as a string
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = {}
        return {"name": function.get("name", ""), "arguments": arguments if isinstance(arguments, dict) else {}}
