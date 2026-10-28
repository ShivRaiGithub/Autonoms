"""A thin client for a local Ollama model with tool calling.

The rest of the code only uses `LLM.chat(messages, tools)`. Stage 9 will swap
the HTTP target for a gateway without touching the agent.
"""

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


class GatewayClient:
    """Talks to a LiteLLM gateway using the OpenAI-style chat API.

    The gateway, not this code, picks the real model, applies rate limits and falls
    back to a backup. The reply says which model answered, so a fallback is visible.
    """

    def __init__(self, client: httpx.Client, model: str):
        self._client = client
        self._model = model

    @classmethod
    def create(cls, base_url: str, model: str, api_key: str | None) -> "GatewayClient":
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        return cls(httpx.Client(base_url=base_url, headers=headers, timeout=httpx.Timeout(900, connect=5)), model)

    @staticmethod
    def to_openai(messages: list[dict]) -> list[dict]:
        """Our messages -> OpenAI format. Tool results must name the call they answer, so
        calls get ids made from their position (stable, so an unchanged prefix stays identical)."""
        converted, pending = [], []
        for index, message in enumerate(messages):
            if message["role"] == "assistant" and message.get("tool_calls"):
                calls = [
                    {"id": f"call_{index}_{k}", "type": "function",
                     "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                    for k, c in enumerate(message["tool_calls"])
                ]  # fmt: skip
                pending = [c["id"] for c in calls]
                converted.append({"role": "assistant", "content": message["content"] or None, "tool_calls": calls})
            elif message["role"] == "tool":
                call_id = pending.pop(0) if pending else f"call_{index}_x"
                converted.append({"role": "tool", "tool_call_id": call_id, "content": message["content"]})
            else:
                converted.append({"role": message["role"], "content": message["content"]})
        return converted

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        body = {"model": self._model, "messages": self.to_openai(messages), "tools": tools, "temperature": 0, "stream": False}
        started = time.monotonic()
        try:
            response = self._client.post("/v1/chat/completions", json=body)
            response.raise_for_status()
            data = response.json()
            message = data["choices"][0]["message"]
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"Model call failed: {exc.__class__.__name__}") from exc
        usage = data.get("usage") or {}
        headers = response.headers
        return {
            "role": "assistant",
            "content": message.get("content") or "",
            "tool_calls": [OllamaClient._tool_call(c) for c in message.get("tool_calls") or []],
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "seconds": round(time.monotonic() - started, 2),
                # The body echoes the public name ("autonoms"); the header names the model that really answered.
                "model": headers.get("x-litellm-model-name") or data.get("model", self._model),
                "fallbacks": int(headers.get("x-litellm-attempted-fallbacks", 0) or 0),
                "gateway_ms": float(headers.get("x-litellm-overhead-duration-ms", 0) or 0),
            },
        }
