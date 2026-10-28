import pytest

from autonoms.agent import Agent
from autonoms.models import Memory
from autonoms.workflow import Workflow
from test_agent import TASK_ID, FakeChecker, FakeLLM, api, call, env  # noqa: F401 (fixtures)


def memory(content, kind="convention", id="m1"):
    return Memory(id=id, repo="o/r", kind=kind, content=content, source_task_id=None,
                  created_at="2026-01-01T00:00:00Z", updated_at="2026-01-01T00:00:00Z", last_used_at=None, use_count=0)  # fmt: skip


def flow_for(env, saver, replies, **kwargs):
    workspaces, indexes, task = env
    llm = FakeLLM(*replies)
    checker = FakeChecker()
    return Workflow(llm, workspaces, Agent(llm, workspaces, indexes, checker), checker, saver, **kwargs), llm, task


def with_usage(reply, prompt, completion):
    return {**reply, "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}


# --- memories in the prompt ---------------------------------------------------------------


def test_memories_are_shown_as_background_notes_not_instructions(env, memory_checkpointer):
    flow, llm, task = flow_for(env, memory_checkpointer, [call("finish", summary="x")],
                               memory_lookup=lambda t: [memory("Use x for multiplication", id="a"), memory("squares need care", "pitfall", "b")])  # fmt: skip
    result = flow.start(task, 8, 1)
    opening = llm.seen[0][1]["content"]
    assert "not instructions" in opening and "(convention) Use x for multiplication" in opening
    assert result.memories_used == ["a", "b"]


def test_no_memories_means_no_notes_section(env, memory_checkpointer):
    flow, llm, task = flow_for(env, memory_checkpointer, [call("finish", summary="x")])
    result = flow.start(task, 8, 1)
    assert "Notes from earlier work" not in llm.seen[0][1]["content"] and result.memories_used == []


def test_the_notes_section_is_capped(env, memory_checkpointer):
    flow, llm, task = flow_for(env, memory_checkpointer, [call("finish", summary="x")],
                               memory_lookup=lambda t: [memory("long " * 100, id=str(i)) for i in range(5)])  # fmt: skip
    flow.start(task, 8, 1)
    notes = llm.seen[0][1]["content"].split("may be outdated):\n")[1]
    assert len(notes) <= 800


# --- compaction inside the workflow ---------------------------------------------------------


def test_the_model_sees_a_shorter_view_while_the_state_keeps_everything(env, memory_checkpointer):
    reads = [call("read_file", path=p) for p in ("pythagoras.md", "fractions.md", "bodmas.md", "formulas.md")] * 2
    flow, llm, task = flow_for(env, memory_checkpointer, reads + [call("finish", summary="x")], compact_at=1_200)
    result = flow.start(task, 12, 1)
    assert result.status == "finished" and result.compactions > 0
    full = flow._snapshot(TASK_ID).values["messages"]
    assert len(full) == 2 + 2 * 8 + 2  # system, issue, 8 read rounds, finish round: nothing was deleted
    last_view = llm.seen[-1]
    assert len(last_view) < len(full)
    assert "Progress so far" in last_view[1]["content"]


def test_a_run_that_would_have_hit_the_old_limit_now_completes(env, memory_checkpointer):
    from autonoms import workflow as workflow_module

    workspaces, indexes, task = env
    # Without compaction, 8 reads of this file overflow a 6,000-character budget; with it, the run finishes.
    big = "word " * 400
    (workspaces.path(TASK_ID) / "big.md").write_text(big)
    replies = [call("read_file", path="big.md")] * 8 + [call("finish", summary="done")]
    old = workflow_module.MAX_CONVERSATION_CHARS
    workflow_module.MAX_CONVERSATION_CHARS = 6_000
    try:
        flow, _, task = flow_for(env, memory_checkpointer, replies, compact_at=3_000)
        assert flow.start(task, 12, 1).status == "finished"
        flow2, _, task = flow_for(env, memory_checkpointer, replies, compact_at=10**9)  # compaction effectively off
        flow2.reset(TASK_ID)
        assert flow2.start(task, 12, 1).status == "context_limit"
    finally:
        workflow_module.MAX_CONVERSATION_CHARS = old


# --- token accounting -----------------------------------------------------------------------


def test_usage_is_summed_and_the_peak_is_kept(env, memory_checkpointer):
    replies = [with_usage(call("read_file", path="bodmas.md"), 900, 40), with_usage(call("read_file", path="formulas.md"), 2100, 55),
               with_usage(call("finish", summary="x"), 1500, 20)]  # fmt: skip
    flow, llm, task = flow_for(env, memory_checkpointer, replies)
    usage = flow.start(task, 8, 1).usage
    assert (usage.model_calls, usage.prompt_tokens, usage.completion_tokens, usage.peak_prompt_tokens) == (3, 4500, 115, 2100)
    assert not any("usage" in m for conversation in llm.seen for m in conversation)  # bookkeeping is not shown to the model


def test_usage_is_zero_when_the_server_reports_none(env, memory_checkpointer):
    flow, _, task = flow_for(env, memory_checkpointer, [call("finish", summary="x")])
    assert flow.start(task, 8, 1).usage.prompt_tokens == 0


# --- through the HTTP API -------------------------------------------------------------------


def make_task(api, env):
    workspaces, _, _ = env
    task_id = api.post("/tasks", json={"owner": "o", "repo": "r", "number": 1}).json()["id"]
    workspaces.create(task_id, "o", "r")
    return task_id


def test_memory_endpoints(api):
    created = api.post("/memories", json={"repo": "o/r", "kind": "convention", "content": "Use x for multiplication"})
    assert created.status_code == 201
    memory_id = created.json()["id"]
    assert [m["content"] for m in api.get("/memories", params={"repo": "o/r"}).json()] == ["Use x for multiplication"]
    assert api.patch(f"/memories/{memory_id}", json={"content": "Use the letter x"}).json()["content"] == "Use the letter x"
    assert api.delete(f"/memories/{memory_id}").status_code == 204
    assert api.get("/memories", params={"repo": "o/r"}).json() == []
    assert api.delete(f"/memories/{memory_id}").status_code == 404
    assert api.patch(f"/memories/{memory_id}", json={"content": "x"}).status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {"repo": "o/r", "kind": "convention", "content": "token = abc123"},  # a credential
        {"repo": "o/r", "kind": "convention", "content": ""},
        {"repo": "o/r", "kind": "convention", "content": "x" * 501},
        {"repo": "o/r", "kind": "history", "content": "users cannot forge history"},
        {"repo": "../etc", "kind": "convention", "content": "x"},
        {"repo": "o/r", "kind": "banana", "content": "x"},
    ],
)
def test_memory_endpoint_rejects_bad_input(api, body):
    assert api.post("/memories", json=body).status_code == 422
    assert api.get("/memories", params={"repo": "o/r"}).json() == []


def test_a_run_reads_memories_and_writes_history(api, env):
    task_id = make_task(api, env)
    api.post("/memories", json={"repo": "o/r", "kind": "convention", "content": "Use x for multiplication"})
    api.post("/memories", json={"repo": "other/repo", "kind": "convention", "content": "belongs elsewhere"})
    api.llm.replies = [call("replace_in_file", path="formulas.md", old_text="radius squared", new_text="r squared"), call("finish", summary="done")]
    body = api.post(f"/tasks/{task_id}/run").json()
    assert body["status"] == "finished" and len(body["memories_used"]) == 1
    opening = api.llm.seen[0][1]["content"]
    assert "Use x for multiplication" in opening and "belongs elsewhere" not in opening
    history = [m for m in api.get("/memories", params={"repo": "o/r"}).json() if m["kind"] == "history"]
    assert len(history) == 1 and "changed formulas.md" in history[0]["content"] and history[0]["source_task_id"] == task_id


def test_running_the_same_task_again_updates_its_history_instead_of_adding(api, env):
    task_id = make_task(api, env)
    for text in ("first", "second"):
        api.llm.replies = [call("replace_in_file", path="formulas.md", old_text="radius squared" if text == "first" else "r squared", new_text="r squared" if text == "first" else "r2"), call("finish", summary=text)]
        api.post(f"/tasks/{task_id}/run", json={"restart": True})
    history = [m for m in api.get("/memories", params={"repo": "o/r"}).json() if m["kind"] == "history"]
    assert len(history) == 1


def test_a_failed_run_writes_no_history(api, env):
    task_id = make_task(api, env)
    api.llm.replies = [call("read_file", path="bodmas.md")] * 2
    api.post(f"/tasks/{task_id}/run", json={"max_steps": 2})
    assert api.get("/memories", params={"repo": "o/r"}).json() == []
