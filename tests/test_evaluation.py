from autonoms.evaluation import evaluate
from autonoms.models import CheckProblem, CheckResult
from test_agent import api, call, env  # noqa: F401 (fixtures)
from test_memory import run as make_run
from test_memory import task as make_task_obj


def problem(file, text="1 + 1 = 3"):
    return CheckProblem(file=file, line=1, text=text, claimed="3", actual="2")


def test_a_good_run_scores_as_completed_clean_and_in_scope():
    run = make_run(files=("formulas.md",)).model_copy(update={
        "check_before": CheckResult(status="failed", equations=5, problems=[problem("formulas.md"), problem("a.md", "2 + 2 = 5")]),
        "check_after": CheckResult(status="failed", equations=5, problems=[problem("a.md", "2 + 2 = 5")]),
    })  # fmt: skip
    evaluation = evaluate(make_task_obj(title="Fix formulas.md"), run)
    assert (evaluation.completed, evaluation.clean, evaluation.in_scope) == (True, True, True)
    assert (evaluation.equations_fixed, evaluation.equations_broken, evaluation.files_changed) == (1, 0, 1)


def test_a_run_that_breaks_something_or_strays_is_not_clean_or_in_scope():
    run = make_run(status="finished_with_regressions", files=("formulas.md", "other.md")).model_copy(update={
        "check_before": CheckResult(status="passed", equations=5),
        "check_after": CheckResult(status="failed", equations=5, problems=[problem("other.md")]),
    })  # fmt: skip
    evaluation = evaluate(make_task_obj(title="Fix formulas.md"), run)
    assert (evaluation.completed, evaluation.clean, evaluation.in_scope, evaluation.equations_broken) == (True, False, False, 1)


def test_unknown_is_not_zero():
    run = make_run().model_copy(update={"check_after": CheckResult(status="not_run", detail="Docker is not available")})
    evaluation = evaluate(make_task_obj(), run)
    assert evaluation.equations_fixed is None and evaluation.equations_broken is None


def test_unfinished_or_empty_runs_did_not_complete():
    assert evaluate(make_task_obj(), make_run(status="max_steps")).completed is False
    assert evaluate(make_task_obj(), make_run(files=())).completed is False


def test_cost_and_speed_come_from_the_run():
    run = make_run().model_copy(update={"seconds": 12.5})
    run.usage.model_calls, run.usage.prompt_tokens, run.usage.completion_tokens, run.usage.model_seconds = 4, 3000, 200, 11.0
    evaluation = evaluate(make_task_obj(), run)
    assert (evaluation.seconds, evaluation.model_seconds, evaluation.model_calls, evaluation.prompt_tokens, evaluation.completion_tokens) == (12.5, 11.0, 4, 3000, 200)


def test_the_evaluation_endpoint(api, env):
    workspaces, _, _ = env
    task_id = api.post("/tasks", json={"owner": "o", "repo": "r", "number": 1}).json()["id"]
    assert api.get(f"/tasks/{task_id}/evaluation").status_code == 409  # no run yet
    workspaces.create(task_id, "o", "r")
    api.llm.replies = [call("replace_in_file", path="formulas.md", old_text="radius squared", new_text="r squared"), call("finish", summary="done")]
    api.post(f"/tasks/{task_id}/run")
    body = api.get(f"/tasks/{task_id}/evaluation").json()
    assert body["completed"] is True and body["clean"] is True and body["files_changed"] == 1
    assert api.get("/tasks/55555555-5555-4555-8555-555555555555/evaluation").status_code == 404
