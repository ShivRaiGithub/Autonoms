import os
import subprocess
import sys

import pytest

from autonoms.agent import Agent
from autonoms.main import app
from autonoms.models import CheckProblem, CheckResult
from autonoms.workflow import Workflow, WorkspaceChanged
from test_agent import FAILED, FakeChecker, FakeLLM, TASK_ID, api, call, env, say  # noqa: F401 (fixtures)

PASSED = CheckResult(status="passed", equations=5)
NOT_RUN = CheckResult(status="not_run", detail="Docker is not available (OSError)")


def flow_for(env, saver, *replies, checker=None):
    workspaces, indexes, task = env
    llm = FakeLLM(*replies)
    checker = checker or FakeChecker()
    agent = Agent(llm, workspaces, indexes, checker)
    return Workflow(llm, workspaces, agent, checker, saver), llm, task, workspaces


def run(env, saver, *replies, checker=None, max_steps=8, max_repairs=1):
    flow, llm, task, _ = flow_for(env, saver, *replies, checker=checker)
    return flow.start(task, max_steps, max_repairs), llm, flow


# --- same behaviour as the Stage 5 loop -------------------------------------------------


def test_graph_gives_the_same_result_as_the_plain_loop(env, memory_checkpointer):
    workspaces, indexes, task = env
    script = [call("search_code", query="circle"), call("read_file", path="formulas.md"), call("finish", summary="nothing")]
    plain = Agent(FakeLLM(*script), workspaces, indexes, FakeChecker()).run(task)
    graph, _, _ = run(env, memory_checkpointer, *script)
    assert (graph.status, graph.summary, graph.steps, graph.diff) == (plain.status, plain.summary, plain.steps, plain.diff)


def test_happy_path_edit_and_finish(env, memory_checkpointer):
    result, llm, _ = run(
        env, memory_checkpointer,
        call("replace_in_file", path="formulas.md", old_text="radius squared", new_text="r squared"),
        call("finish", summary="done"),
    )  # fmt: skip
    assert result.status == "finished" and "+The area of a circle is pi times the r squared." in result.diff
    assert result.check_before.status == "passed" and result.check_after.status == "passed"


@pytest.mark.parametrize(
    "replies, kwargs, status",
    [
        ([say("fine as it is")], {}, "no_tool_call"),
        ([call("read_file", path="bodmas.md")] * 3, {"max_steps": 3}, "max_steps"),
    ],
)
def test_other_ways_a_run_ends(env, memory_checkpointer, replies, kwargs, status):
    result, _, _ = run(env, memory_checkpointer, *replies, **kwargs)
    assert result.status == status


def test_llm_error_ends_the_run_cleanly(env, memory_checkpointer):
    from autonoms.llm import LLMError

    result, _, _ = run(env, memory_checkpointer, LLMError("Model call failed: ConnectError"))
    assert result.status == "llm_error" and "ConnectError" in result.summary


# --- the conditional routing: repair --------------------------------------------------


def test_new_wrong_equation_sends_the_agent_back_to_repair(env, memory_checkpointer):
    checker = FakeChecker(PASSED, FAILED, PASSED)  # baseline ok, first verify finds a regression, second ok
    result, llm, _ = run(env, memory_checkpointer, call("finish", summary="first try"), call("finish", summary="fixed"), checker=checker)
    assert result.status == "finished" and result.summary == "fixed"
    assert [s.tool for s in result.steps] == ["finish", "finish"]
    repair_prompt = llm.seen[1][-1]
    assert repair_prompt["role"] == "user"
    assert "correct before your edits" in repair_prompt["content"] and "bodmas.md line 11" in repair_prompt["content"]


def test_repairs_are_bounded(env, memory_checkpointer):
    checker = FakeChecker(PASSED, FAILED)  # the problem never goes away
    result, llm, flow = run(
        env, memory_checkpointer, call("finish", summary="a"), call("finish", summary="b"), call("finish", summary="c"),
        checker=checker, max_repairs=1,
    )  # fmt: skip
    assert result.status == "finished_with_regressions" and len(llm.seen) == 2  # one repair round, then stop
    assert flow.status(TASK_ID, running=False).repairs == 1


def test_no_repair_when_repairs_are_disabled(env, memory_checkpointer):
    result, llm, _ = run(env, memory_checkpointer, call("finish", summary="a"), checker=FakeChecker(PASSED, FAILED), max_repairs=0)
    assert result.status == "finished_with_regressions" and len(llm.seen) == 1


def test_a_problem_that_was_already_there_is_not_a_regression(env, memory_checkpointer):
    result, llm, _ = run(env, memory_checkpointer, call("finish", summary="a"), checker=FakeChecker(FAILED))
    assert result.status == "finished" and len(llm.seen) == 1


def test_a_check_that_did_not_run_does_not_trigger_repair(env, memory_checkpointer):
    result, _, _ = run(env, memory_checkpointer, call("finish", summary="a"), checker=FakeChecker(PASSED, NOT_RUN))
    assert result.status == "finished" and result.check_after.status == "not_run"


# --- checkpoints: crash and resume --------------------------------------------------


def crash_midway(env, saver):
    """Two good model calls, then the process 'dies' during the third."""
    flow, llm, task, workspaces = flow_for(
        env, saver, call("read_file", path="bodmas.md"), RuntimeError("power cut"), call("finish", summary="done")
    )
    with pytest.raises(RuntimeError):
        flow.start(task, 8, 1)
    return flow, llm, workspaces


def test_status_before_during_and_after_a_crash(env, memory_checkpointer):
    flow, _, task, _ = flow_for(env, memory_checkpointer, call("read_file", path="bodmas.md"), RuntimeError("power cut"))
    assert flow.status(TASK_ID, running=False).state == "not_started"
    with pytest.raises(RuntimeError):
        flow.start(task, 8, 1)
    status = flow.status(TASK_ID, running=False)
    assert (status.state, status.next, status.model_calls, len(status.steps)) == ("interrupted", ["agent"], 1, 1)
    assert flow.status(TASK_ID, running=True).state == "running"


def test_resume_continues_without_repeating_finished_work(env, memory_checkpointer):
    flow, llm, _ = crash_midway(env, memory_checkpointer)
    result = flow.resume(TASK_ID)
    assert result.status == "finished"
    assert [s.tool for s in result.steps] == ["read_file", "finish"]  # read_file ran once, not twice
    assert len(llm.seen) == 3  # model calls: read, the crashed one, finish
    assert flow.status(TASK_ID, running=False).state == "finished"


def test_resume_refuses_when_the_workspace_changed_meanwhile(env, memory_checkpointer):
    flow, _, workspaces = crash_midway(env, memory_checkpointer)
    workspaces.replace_in_file(TASK_ID, "formulas.md", "radius squared", "tampered")
    with pytest.raises(WorkspaceChanged):
        flow.resume(TASK_ID)


def test_finished_runs_can_be_detected_and_reset(env, memory_checkpointer):
    result, _, flow = run(env, memory_checkpointer, call("finish", summary="a"))
    assert flow.exists(TASK_ID) and flow.finished(TASK_ID)
    flow.reset(TASK_ID)
    assert not flow.exists(TASK_ID)


# --- through the HTTP API -----------------------------------------------------------


def make_task(api, env):
    workspaces, _, _ = env
    task_id = api.post("/tasks", json={"owner": "o", "repo": "r", "number": 1}).json()["id"]
    workspaces.create(task_id, "o", "r")
    return task_id


def test_api_run_status_rerun_and_restart(api, env):
    task_id = make_task(api, env)
    assert api.get(f"/tasks/{task_id}/run").json()["state"] == "not_started"
    api.llm.replies = [call("finish", summary="one")]
    assert api.post(f"/tasks/{task_id}/run").json()["summary"] == "one"
    status = api.get(f"/tasks/{task_id}/run").json()
    assert status["state"] == "finished" and status["status"] == "finished" and status["model_calls"] == 1
    assert api.post(f"/tasks/{task_id}/run").status_code == 409  # already ran
    assert api.post(f"/tasks/{task_id}/resume").status_code == 409  # nothing to resume
    api.llm.replies = [call("finish", summary="two")]
    assert api.post(f"/tasks/{task_id}/run", json={"restart": True}).json()["summary"] == "two"


def test_api_crash_then_resume(api, env):
    task_id = make_task(api, env)
    api.llm.replies = [call("read_file", path="bodmas.md"), RuntimeError("power cut")]
    with pytest.raises(RuntimeError):
        api.post(f"/tasks/{task_id}/run")
    status = api.get(f"/tasks/{task_id}/run").json()
    assert status["state"] == "interrupted" and status["next"] == ["agent"] and len(status["steps"]) == 1
    assert api.post(f"/tasks/{task_id}/run").status_code == 409  # must resume or restart
    api.llm.replies = [call("finish", summary="recovered")]
    body = api.post(f"/tasks/{task_id}/resume").json()
    assert body["status"] == "finished" and body["summary"] == "recovered"
    assert [s["tool"] for s in body["steps"]] == ["read_file", "finish"]


def test_api_resume_errors(api, env):
    task_id = make_task(api, env)
    assert api.post(f"/tasks/{task_id}/resume").status_code == 404  # no run at all
    assert api.post("/tasks/44444444-4444-4444-8444-444444444444/resume").status_code == 404


def test_api_resume_after_workspace_edit_is_409(api, env):
    task_id = make_task(api, env)
    workspaces, _, _ = env
    api.llm.replies = [call("read_file", path="bodmas.md"), RuntimeError("power cut")]
    with pytest.raises(RuntimeError):
        api.post(f"/tasks/{task_id}/run")
    workspaces.replace_in_file(task_id, "formulas.md", "radius squared", "changed by hand")
    response = api.post(f"/tasks/{task_id}/resume")
    assert response.status_code == 409 and "workspace changed" in response.json()["detail"]


def test_deleting_a_task_deletes_its_checkpoints(api, env, memory_checkpointer):
    task_id = make_task(api, env)
    api.llm.replies = [call("finish", summary="a")]
    api.post(f"/tasks/{task_id}/run")
    config = {"configurable": {"thread_id": task_id}}
    assert memory_checkpointer.get_tuple(config) is not None
    api.delete(f"/tasks/{task_id}")
    assert memory_checkpointer.get_tuple(config) is None


# --- privacy -----------------------------------------------------------------------


def test_tracing_is_forced_off_even_if_the_environment_enables_it():
    code = "import autonoms, os; print(os.environ['LANGSMITH_TRACING'], os.environ['LANGCHAIN_TRACING_V2'])"
    out = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "LANGSMITH_TRACING": "true", "LANGCHAIN_TRACING_V2": "true"},
        capture_output=True, text=True,
    )  # fmt: skip
    assert out.stdout.split() == ["false", "false"]


def test_concurrent_first_requests_create_the_checkpointer_once(monkeypatch):
    """Regression: two threads calling get_checkpointer at once both ran setup()."""
    import threading
    import time

    from autonoms import main

    created = []

    def slow_create():
        created.append(1)
        time.sleep(0.2)  # long enough for the other threads to arrive
        return object()

    monkeypatch.setattr(main, "_checkpointer", None)
    monkeypatch.setattr(main, "_create_checkpointer", slow_create)
    results = []
    threads = [threading.Thread(target=lambda: results.append(main.get_checkpointer())) for _ in range(5)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(created) == 1 and len({id(r) for r in results}) == 1


def test_run_check_tells_the_model_what_was_already_wrong(env, memory_checkpointer):
    """The real run showed a model fixing every listed problem. The text now separates old from new."""
    old = CheckProblem(file="arithmetic.md", line=8, text="12 / 4 = 4", claimed="4", actual="3")
    new = CheckProblem(file="bodmas.md", line=11, text="2 + 12 = 20", claimed="20", actual="14")
    before = CheckResult(status="failed", equations=11, problems=[old])
    after = CheckResult(status="failed", equations=11, problems=[old, new])
    result, _, _ = run(env, memory_checkpointer, call("run_check"), call("finish", summary="x"),
                       checker=FakeChecker(before, after), max_repairs=0)  # fmt: skip
    text = result.steps[0].result
    assert text.index("NEW since you started") < text.index("bodmas.md line 11") < text.index("Already wrong before you started")
    assert text.index("Already wrong before you started") < text.index("arithmetic.md line 8")


def test_run_check_says_when_the_edits_broke_nothing(env, memory_checkpointer):
    old = CheckProblem(file="arithmetic.md", line=8, text="12 / 4 = 4", claimed="4", actual="3")
    failed = CheckResult(status="failed", equations=11, problems=[old])
    result, _, _ = run(env, memory_checkpointer, call("run_check"), call("finish", summary="x"), checker=FakeChecker(failed))
    assert "Your edits broke nothing." in result.steps[0].result and "NEW since" not in result.steps[0].result
