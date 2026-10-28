import httpx
import pytest

from autonoms import agent as agent_module
from autonoms.agent import Agent
from autonoms.llm import LLMError, OllamaClient
from autonoms.main import app, get_checker, get_indexes, get_llm, get_workspaces
from autonoms.models import CheckProblem, CheckResult, Task
from autonoms.search import SearchIndexes
from autonoms.workspace import Workspaces
from conftest import make_issue

TASK_ID = "11111111-1111-4111-8111-111111111111"


class FakeLLM:
    """Returns scripted replies and records every conversation it was shown."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen: list[list[dict]] = []

    def chat(self, messages, tools):
        self.seen.append([dict(m) for m in messages])
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeChecker:
    """Returns scripted check results in order (the last one repeats)."""

    def __init__(self, *results):
        self.results = list(results) or [CheckResult(status="passed", equations=5)]
        self.calls = 0

    def run(self, task_id):
        self.calls += 1
        return self.results.pop(0) if len(self.results) > 1 else self.results[0]


FAILED = CheckResult(
    status="failed", equations=5,
    problems=[CheckProblem(file="bodmas.md", line=11, text="2 + 12 = 20", claimed="20", actual="14")],
)  # fmt: skip


def call(name, **arguments):
    return {"role": "assistant", "content": "", "tool_calls": [{"name": name, "arguments": arguments}]}


def say(text):
    return {"role": "assistant", "content": text, "tool_calls": []}


@pytest.fixture
def env(tmp_path, notes_server):
    workspaces = Workspaces(tmp_path / "ws", notes_server)
    workspaces.create(TASK_ID, "o", "r")
    indexes = SearchIndexes(tmp_path / "idx", workspaces, "BAAI/bge-small-en-v1.5")
    indexes.build(TASK_ID)
    issue = {
        "owner": "o", "repo": "r", "number": 1, "title": "Circle area is wrong",
        "body": "formulas.md says pi times the radius squared but should say pi r squared",
        "state": "open", "author": "a", "labels": [], "url": "u",
        "created_at": "2026-01-01T00:00:00Z", "comments": [],
    }  # fmt: skip
    task = Task(
        id=TASK_ID, owner="o", repo="r", issue_number=1, issue_title=issue["title"],
        status="created", created_at="2026-01-01T00:00:00Z", updated_at="2026-01-01T00:00:00Z", issue=issue,
    )  # fmt: skip
    return workspaces, indexes, task


def run(env, *replies, max_steps=8, checker=None):
    workspaces, indexes, task = env
    llm = FakeLLM(*replies)
    return Agent(llm, workspaces, indexes, checker or FakeChecker()).run(task, max_steps), llm


def test_happy_path_search_read_edit_finish(env):
    workspaces, indexes, _ = env
    result, llm = run(
        env,
        call("search_code", query="circle area"),
        call("read_file", path="formulas.md"),
        call("replace_in_file", path="formulas.md", old_text="radius squared", new_text="zebra squared"),
        call("finish", summary="Reworded the circle formula."),
    )
    assert result.status == "finished"
    assert result.summary == "Reworded the circle formula."
    assert [s.tool for s in result.steps] == ["search_code", "read_file", "replace_in_file", "finish"]
    assert "-The area of a circle is pi times the radius squared." in result.diff
    assert "+The area of a circle is pi times the zebra squared." in result.diff
    # the index was rebuilt, so search now sees the new text
    assert indexes.search(TASK_ID, "zebra", "keyword").hits[0].path == "formulas.md"


def test_first_prompt_contains_issue_files_and_hints_but_no_thinking(env):
    _, llm = run(env, call("finish", summary="nothing"))
    system, user = llm.seen[0]
    assert system["role"] == "system"
    assert "Circle area is wrong" in user["content"]
    assert "formulas.md" in user["content"] and "pythagoras.md" in user["content"]
    assert "may be relevant" in user["content"]


def test_tool_results_are_fed_back(env):
    _, llm = run(env, call("read_file", path="bodmas.md"), call("finish", summary="done"))
    last = llm.seen[1][-1]
    assert last["role"] == "tool" and last["tool_name"] == "read_file"
    assert "BODMAS" in last["content"]


@pytest.mark.parametrize(
    "bad_call",
    [
        call("rm_rf", path="/"),
        call("read_file"),  # missing argument
        call("read_file", path="../../etc/passwd"),
        call("read_file", path="nope.md"),
        call("read_file", path="image.bin"),
        call("replace_in_file", path="formulas.md", old_text="not in file", new_text="x"),
        call("replace_in_file", path="formulas.md", old_text="", new_text="x"),
        call("replace_in_file", path="../../tmp/x", old_text="a", new_text="b"),
    ],
)
def test_bad_tool_calls_become_error_text_not_crashes(env, bad_call):
    result, llm = run(env, bad_call, call("finish", summary="gave up"))
    assert result.steps[0].result.startswith("Error")
    assert result.status == "finished"  # the loop carried on
    assert result.diff == ""  # nothing was changed


def test_ambiguous_replace_is_refused(env):
    # "the" appears twice in pythagoras.md
    result, _ = run(env, call("replace_in_file", path="pythagoras.md", old_text="the", new_text="THE"), call("finish", summary="x"))
    assert "appears" in result.steps[0].result and result.diff == ""


def test_step_limit(env):
    result, llm = run(env, *[call("read_file", path="bodmas.md")] * 3, max_steps=3)
    assert result.status == "max_steps" and len(result.steps) == 3 and len(llm.seen) == 3


def test_model_that_never_calls_a_tool(env):
    result, _ = run(env, say("I think it is fine."))
    assert result.status == "no_tool_call" and result.summary == "I think it is fine."


def test_llm_failure_is_reported_not_raised(env):
    result, _ = run(env, LLMError("Model call failed: ConnectError"))
    assert result.status == "llm_error" and "ConnectError" in result.summary


def test_long_tool_output_is_cut(env, monkeypatch):
    monkeypatch.setattr(agent_module, "MAX_TOOL_OUTPUT_CHARS", 40)
    result, _ = run(env, call("read_file", path="pythagoras.md"), call("finish", summary="x"))
    assert "[... cut:" in result.steps[0].result and len(result.steps[0].result) < 120


def test_conversation_budget_stops_the_run(env, monkeypatch):
    monkeypatch.setattr(agent_module, "MAX_CONVERSATION_CHARS", 100)
    result, llm = run(env, call("finish", summary="x"))
    assert result.status == "context_limit" and llm.seen == []


def test_two_tool_calls_in_one_reply_run_in_order(env):
    both = {"role": "assistant", "content": "", "tool_calls": [
        {"name": "read_file", "arguments": {"path": "bodmas.md"}},
        {"name": "finish", "arguments": {"summary": "ok"}},
    ]}  # fmt: skip
    result, _ = run(env, both)
    assert [s.tool for s in result.steps] == ["read_file", "finish"]


# --- the Ollama client, with a fake HTTP server -----------------------------------


def ollama(handler, think=None):
    return OllamaClient(httpx.Client(transport=httpx.MockTransport(handler), base_url="http://x"), "m", think)


def test_ollama_request_and_response_shape():
    seen = {}

    def handler(request):
        seen.update(__import__("json").loads(request.content))
        return httpx.Response(200, json={"message": {
            "role": "assistant", "content": "", "thinking": "long private reasoning",
            "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "a.md"}}}],
        }})  # fmt: skip

    reply = ollama(handler, think=False).chat(
        [{"role": "assistant", "content": "", "tool_calls": [{"name": "t", "arguments": {"a": 1}}]}], [{"tools": 1}]
    )
    usage = reply.pop("usage")
    assert reply == {"role": "assistant", "content": "", "tool_calls": [{"name": "read_file", "arguments": {"path": "a.md"}}]}
    assert (usage["prompt_tokens"], usage["completion_tokens"], usage["model"]) == (0, 0, "m") and usage["seconds"] >= 0
    assert seen["model"] == "m" and seen["stream"] is False and seen["think"] is False
    assert seen["options"]["temperature"] == 0
    assert seen["messages"][0]["tool_calls"] == [{"function": {"name": "t", "arguments": {"a": 1}}}]


def test_ollama_accepts_arguments_sent_as_a_json_string():
    reply = ollama(lambda r: httpx.Response(200, json={"message": {"content": "", "tool_calls": [
        {"function": {"name": "read_file", "arguments": '{"path": "a.md"}'}}]}})).chat([], [])  # fmt: skip
    assert reply["tool_calls"][0]["arguments"] == {"path": "a.md"}


@pytest.mark.parametrize("handler", [
    lambda r: httpx.Response(500, json={"error": "boom"}),
    lambda r: httpx.Response(200, text="not json"),
    lambda r: httpx.Response(200, json={"nothing": 1}),
    lambda r: (_ for _ in ()).throw(httpx.ConnectError("down")),
])  # fmt: skip
def test_ollama_failures_become_llmerror(handler):
    with pytest.raises(LLMError):
        ollama(handler).chat([], [])


# --- through the HTTP API ---------------------------------------------------------


@pytest.fixture
def api(fake_github, db_session, env):
    workspaces, indexes, _ = env
    llm = FakeLLM()
    app.dependency_overrides[get_checker] = lambda: FakeChecker()
    app.dependency_overrides[get_workspaces] = lambda: workspaces
    app.dependency_overrides[get_indexes] = lambda: indexes
    app.dependency_overrides[get_llm] = lambda: llm

    def handler(request):
        if request.url.path.endswith("/comments"):
            return httpx.Response(200, json=[])
        return httpx.Response(200, json=make_issue(number=1, title="Circle area is wrong"))

    client = fake_github(handler)
    client.llm = llm
    yield client
    for dep in (get_workspaces, get_indexes, get_llm, get_checker):
        app.dependency_overrides.pop(dep, None)


def test_api_run(api, env):
    workspaces, _, _ = env
    task_id = api.post("/tasks", json={"owner": "o", "repo": "r", "number": 1}).json()["id"]
    workspaces.create(task_id, "o", "r")
    api.llm.replies = [call("finish", summary="nothing to do")]
    body = api.post(f"/tasks/{task_id}/run", json={"max_steps": 3}).json()
    assert body["status"] == "finished" and body["summary"] == "nothing to do"


def test_api_run_needs_task_and_workspace_and_valid_limit(api):
    assert api.post("/tasks/22222222-2222-4222-8222-222222222222/run").status_code == 404
    task_id = api.post("/tasks", json={"owner": "o", "repo": "r", "number": 1}).json()["id"]
    assert api.post(f"/tasks/{task_id}/run").status_code == 404  # no workspace
    assert api.post(f"/tasks/{task_id}/run", json={"max_steps": 99}).status_code == 422


def test_api_rejects_a_second_run_at_the_same_time(api, env):
    from autonoms import main

    workspaces, _, _ = env
    task_id = api.post("/tasks", json={"owner": "o", "repo": "r", "number": 1}).json()["id"]
    workspaces.create(task_id, "o", "r")
    main._running.add(task_id)
    try:
        assert api.post(f"/tasks/{task_id}/run").status_code == 409
    finally:
        main._running.discard(task_id)


# --- checks ---------------------------------------------------------------------------


def test_run_check_tool_shows_problems_to_the_model(env):
    result, llm = run(env, call("run_check"), call("finish", summary="x"), checker=FakeChecker(FAILED))
    text = result.steps[0].result
    assert text.startswith("CHECK FAILED: 1 wrong of 5")
    assert "bodmas.md line 11: 2 + 12 = 20 (the result should be 14)" in text


def test_before_and_after_checks_are_recorded(env):
    checker = FakeChecker(FAILED, CheckResult(status="passed", equations=5))
    result, _ = run(env, call("finish", summary="x"), checker=checker)
    assert result.check_before.status == "failed" and result.check_after.status == "passed"
    assert checker.calls == 2


def test_a_check_that_did_not_run_is_never_described_as_a_pass(env):
    broken = CheckResult(status="not_run", detail="Docker is not available (OSError)")
    result, _ = run(env, call("run_check"), call("finish", summary="x"), checker=FakeChecker(broken))
    assert result.steps[0].result == "CHECK NOT RUN: Docker is not available (OSError). Nothing was verified."
    assert "PASSED" not in result.steps[0].result
    assert result.check_after.status == "not_run"


def test_passed_check_with_zero_equations_says_so(env):
    from autonoms.agent import describe

    assert describe(CheckResult(status="passed", equations=0)) == "CHECK PASSED: 0 equations checked, all correct."


def test_api_check_endpoint(api, env):
    workspaces, _, _ = env
    task_id = api.post("/tasks", json={"owner": "o", "repo": "r", "number": 1}).json()["id"]
    from autonoms.checker import Checker
    from autonoms.sandbox import Sandbox

    # the real Checker refuses before it ever reaches Docker when there is no workspace
    app.dependency_overrides[get_checker] = lambda: Checker(Sandbox("unused"), workspaces)
    assert api.post(f"/tasks/{task_id}/check").status_code == 404
    workspaces.create(task_id, "o", "r")
    app.dependency_overrides[get_checker] = lambda: FakeChecker(FAILED)
    body = api.post(f"/tasks/{task_id}/check").json()
    assert body["status"] == "failed" and body["problems"][0]["line"] == 11
    assert api.post("/tasks/33333333-3333-4333-8333-333333333333/check").status_code == 404
