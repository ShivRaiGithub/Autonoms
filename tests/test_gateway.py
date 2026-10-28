import json

import httpx
import pytest

from autonoms.config import Settings
from autonoms.llm import GatewayClient, LLMError
from test_agent import env  # noqa: F401 (fixture)
from autonoms.main import get_llm

TOOLS = [{"type": "function", "function": {"name": "read_file"}}]


def client(handler, model="autonoms"):
    return GatewayClient(httpx.Client(transport=httpx.MockTransport(handler), base_url="http://gw"), model)


def completion(message, usage=None, model="autonoms", headers=None):
    return httpx.Response(200, json={"model": model, "choices": [{"message": message}], "usage": usage or {}}, headers=headers or {})


def test_request_is_openai_shaped_and_tool_results_name_their_call():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        seen["path"] = request.url.path
        return completion({"content": "ok"})

    history = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "issue"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"name": "read_file", "arguments": {"path": "a.md"}}, {"name": "read_file", "arguments": {"path": "b.md"}}]},
        {"role": "tool", "tool_name": "read_file", "content": "A"},
        {"role": "tool", "tool_name": "read_file", "content": "B"},
        {"role": "assistant", "content": "", "tool_calls": [{"name": "finish", "arguments": {"summary": "x"}}]},
        {"role": "tool", "tool_name": "finish", "content": "x"},
    ]  # fmt: skip
    client(handler).chat(history, TOOLS)
    assert seen["path"] == "/v1/chat/completions"
    assert seen["model"] == "autonoms" and seen["temperature"] == 0 and seen["stream"] is False
    calls = seen["messages"][2]["tool_calls"]
    assert [c["function"]["arguments"] for c in calls] == ['{"path": "a.md"}', '{"path": "b.md"}']  # a JSON *string*
    assert seen["messages"][3] == {"role": "tool", "tool_call_id": calls[0]["id"], "content": "A"}
    assert seen["messages"][4] == {"role": "tool", "tool_call_id": calls[1]["id"], "content": "B"}
    assert seen["messages"][6]["tool_call_id"] == seen["messages"][5]["tool_calls"][0]["id"]
    assert len({c["id"] for c in calls}) == 2 and "tool_name" not in json.dumps(seen["messages"])


def test_ids_depend_only_on_position_so_an_unchanged_prefix_stays_identical():
    history = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
               {"role": "assistant", "content": "", "tool_calls": [{"name": "read_file", "arguments": {"path": "a"}}]},
               {"role": "tool", "tool_name": "read_file", "content": "A"}]  # fmt: skip
    assert GatewayClient.to_openai(history) == GatewayClient.to_openai(list(history) + [{"role": "user", "content": "more"}])[:4]


def test_reply_is_parsed_with_usage_latency_and_serving_model():
    reply = client(lambda r: completion(
        {"content": "", "tool_calls": [{"id": "x", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.md"}'}}]},
        usage={"prompt_tokens": 900, "completion_tokens": 30},
        headers={"x-litellm-model-name": "ollama_chat/qwen3.5:9b", "x-litellm-attempted-fallbacks": "1", "x-litellm-overhead-duration-ms": "4.9"},
    )).chat([{"role": "user", "content": "hi"}], TOOLS)  # fmt: skip
    assert reply["tool_calls"] == [{"name": "read_file", "arguments": {"path": "a.md"}}]
    usage = reply["usage"]
    assert (usage["prompt_tokens"], usage["completion_tokens"], usage["model"]) == (900, 30, "ollama_chat/qwen3.5:9b")
    assert usage["seconds"] >= 0
    assert (usage["fallbacks"], usage["gateway_ms"]) == (1, 4.9)  # read from the gateway's headers


def test_without_gateway_headers_the_body_model_and_zero_overhead_are_used():
    usage = client(lambda r: completion({"content": "x"}, model="autonoms")).chat([{"role": "user", "content": "hi"}], TOOLS)["usage"]
    assert (usage["model"], usage["fallbacks"], usage["gateway_ms"]) == ("autonoms", 0, 0.0)


def test_a_text_only_reply_has_no_tool_calls():
    assert client(lambda r: completion({"content": "I am done"})).chat([{"role": "user", "content": "hi"}], TOOLS)["tool_calls"] == []


@pytest.mark.parametrize("handler", [
    lambda r: httpx.Response(429, json={"error": "rate limit"}),  # the gateway's rate limit
    lambda r: httpx.Response(500, json={}),
    lambda r: httpx.Response(200, text="<html>"),
    lambda r: httpx.Response(200, json={"choices": []}),
    lambda r: (_ for _ in ()).throw(httpx.ConnectError("gateway down")),
])  # fmt: skip
def test_gateway_failures_become_llmerror(handler):
    with pytest.raises(LLMError):
        client(handler).chat([{"role": "user", "content": "hi"}], TOOLS)


def test_the_master_key_is_sent_but_never_shown():
    sent = {}
    gateway = GatewayClient.create("http://gw", "autonoms", "sk-secret-key")
    gateway._client = httpx.Client(transport=httpx.MockTransport(lambda r: sent.update(auth=r.headers["authorization"]) or completion({"content": ""})),
                                   base_url="http://gw", headers={"Authorization": "Bearer sk-secret-key"})  # fmt: skip
    gateway.chat([{"role": "user", "content": "hi"}], TOOLS)
    assert sent["auth"] == "Bearer sk-secret-key"
    settings = Settings(litellm_master_key="sk-secret-key")
    assert "sk-secret-key" not in repr(settings) and "sk-secret-key" not in str(settings)


def test_settings_choose_the_backend(monkeypatch):
    from autonoms import main
    from autonoms.llm import OllamaClient

    for backend, expected in (("ollama", OllamaClient), ("gateway", GatewayClient)):
        monkeypatch.setattr(main, "get_settings", lambda b=backend: Settings(llm_backend=b, litellm_master_key="sk-x"))
        assert isinstance(get_llm(), expected)


def test_the_run_reports_which_model_answered_and_how_long_it_took(env, memory_checkpointer):
    from test_memory_flow import flow_for, with_usage
    from test_agent import call

    replies = [with_usage(call("read_file", path="bodmas.md"), 900, 40), with_usage(call("finish", summary="x"), 1000, 20)]
    replies[0]["usage"] |= {"seconds": 2.5, "model": "ollama_chat/qwen3.5:9b-8k"}
    replies[1]["usage"] |= {"seconds": 1.5, "model": "ollama_chat/qwen3.5:9b", "fallbacks": 1, "gateway_ms": 5.0}  # served by the backup
    flow, _, task = flow_for(env, memory_checkpointer, replies)
    result = flow.start(task, 8, 1)
    assert (result.usage.model_seconds, result.usage.fallbacks, result.usage.gateway_overhead_ms) == (4.0, 1, 5.0)
    assert result.models_used == ["ollama_chat/qwen3.5:9b-8k", "ollama_chat/qwen3.5:9b"]
