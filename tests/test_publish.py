import json
import subprocess
import uuid
from pathlib import Path

import httpx
import pytest

from autonoms import report
from autonoms.github import GitHubClient, GitHubError, PullRequestExists
from autonoms.main import app, get_checker, get_indexes, get_llm, get_workspaces
from autonoms.models import AgentRun, CheckProblem, CheckResult
from autonoms.workspace import GitError, InvalidPath
from conftest import GIT_ENV, make_issue, notes_server, server  # noqa: F401 (fixtures)
from test_agent import FAILED, FakeChecker, FakeLLM, call, env  # noqa: F401 (fixtures)
from test_memory import run as make_run
from test_memory import task as make_task_obj

PASSED = CheckResult(status="passed", equations=5)
AUTHOR = ("Autonoms", "autonoms@users.noreply.github.com")
BRANCH = "autonoms/issue-1-abcdef12"


def origin_path(workspaces) -> Path:
    return Path(workspaces.clone_base.removeprefix("file://")) / "o" / "r.git"


def git_out(cwd, *args) -> str:
    return subprocess.run(["git", *args], cwd=cwd, env=GIT_ENV, check=True, capture_output=True, text=True).stdout.strip()


# --- the report -------------------------------------------------------------------------------


def test_review_contents():
    review = report.build_review(make_task_obj(), make_run(files=("formulas.md",)))
    assert review.title == 'Fix #5: Fix circle area'
    assert review.diff == make_run(files=("formulas.md",)).diff  # the reader is shown exactly what the fingerprint covers
    assert review.body.startswith("Closes #5") and "`formulas.md` (+0 -0)" in review.body
    assert review.files == ["formulas.md"] and review.branch.startswith("autonoms/issue-5-11111111")
    assert len(review.fingerprint) == 64 and review.can_publish and review.blockers == []


def test_diffstat_counts_added_and_removed_lines():
    diff = "diff --git a/a.md b/a.md\n--- a/a.md\n+++ b/a.md\n@@ -1 +1 @@\n-old\n+new\n+more\ndiff --git a/b.md b/b.md\n--- a/b.md\n+++ b/b.md\n-x\n"
    assert report.diffstat(diff) == {"a.md": (2, 1), "b.md": (0, 1)}


def test_mentions_from_outside_cannot_ping_people():
    review = report.build_review(make_task_obj(title="Fix @octocat and @everyone"), make_run())
    assert "@octocat" not in review.title + review.body and "@​octocat" in review.title


def test_the_fingerprint_changes_when_anything_published_changes():
    task, run = make_task_obj(), make_run()
    base = report.build_review(task, run).fingerprint
    assert report.build_review(task, run).fingerprint == base
    assert report.build_review(task, run.model_copy(update={"diff": run.diff + "+x\n"})).fingerprint != base
    assert report.build_review(task, run.model_copy(update={"summary": "something else"})).fingerprint != base
    assert report.build_review(make_task_obj(title="Other"), run).fingerprint != base


def risks(review, level=None):
    return [r.text for r in review.risks if level is None or r.level == level]


def test_regressions_unrun_checks_and_unmentioned_files_are_warnings():
    new = CheckProblem(file="bodmas.md", line=3, text="2 + 2 = 5", claimed="5", actual="4")
    run = make_run(files=("formulas.md", "other.md")).model_copy(update={"check_after": CheckResult(status="failed", equations=5, problems=[new])})
    review = report.build_review(make_task_obj(title="Fix circle in formulas.md"), run)
    warnings = " ".join(risks(review, "warn"))
    assert "made 1 equation(s) wrong" in warnings and "bodmas.md line 3: 2 + 2 = 5" in warnings
    assert "other.md" in warnings and "formulas.md" not in warnings.split("does not name:")[1]
    assert review.needs_acknowledgement
    unrun = make_run().model_copy(update={"check_after": CheckResult(status="not_run", detail="Docker is not available")})
    assert any("did not run after" in t for t in risks(report.build_review(make_task_obj(), unrun), "warn"))


def test_pre_existing_problems_and_the_coverage_limit_are_information_only():
    review = report.build_review(make_task_obj(title="Fix formulas.md"), make_run(files=("formulas.md",), before=3, after=3))
    assert not review.needs_acknowledgement
    assert any("already wrong before" in t for t in risks(review, "info"))
    assert any("cannot judge wording" in t for t in risks(review, "info"))


@pytest.mark.parametrize("bad", [make_run(status="max_steps"), make_run(status="llm_error"), make_run(files=())])
def test_runs_that_did_not_finish_or_changed_nothing_are_blocked(bad):
    review = report.build_review(make_task_obj(), bad)
    assert review.blockers and not review.can_publish


def test_final_body_names_the_approver_safely():
    review = report.build_review(make_task_obj(), make_run())
    assert "approval by **Alice**" in report.final_body(review, "Alice")
    assert "@​mallory" in report.final_body(review, "@mallory")


# --- git: commit and push -------------------------------------------------------------------------


@pytest.fixture
def ws(env):
    workspaces, _, _ = env
    task_id = str(uuid.uuid4())
    workspaces.create(task_id, "o", "r")
    return workspaces, task_id


def test_commit_includes_edits_to_tracked_files_only(ws):
    workspaces, task_id = ws
    path = workspaces.path(task_id)
    workspaces.replace_in_file(task_id, "formulas.md", "radius squared", "r squared")
    (path / "stray.txt").write_text("an untracked file the agent must not publish")
    sha = workspaces.commit_to_branch(task_id, BRANCH, "Fix it", AUTHOR)
    assert git_out(path, "show", "--name-only", "--format=", "HEAD") == "formulas.md"
    assert git_out(path, "log", "-1", "--format=%an <%ae>") == "Autonoms <autonoms@users.noreply.github.com>"
    assert git_out(path, "status", "--porcelain") == "?? stray.txt" and sha == git_out(path, "rev-parse", "HEAD")


def test_committing_twice_changes_nothing(ws):
    workspaces, task_id = ws
    workspaces.replace_in_file(task_id, "formulas.md", "radius squared", "r squared")
    first = workspaces.commit_to_branch(task_id, BRANCH, "Fix it", AUTHOR)
    assert workspaces.commit_to_branch(task_id, BRANCH, "Fix it", AUTHOR) == first
    assert git_out(workspaces.path(task_id), "rev-list", "--count", "main..HEAD") == "1"


def test_push_puts_the_branch_on_the_remote_and_base_is_main(ws, env):
    workspaces, task_id = ws
    workspaces.replace_in_file(task_id, "formulas.md", "radius squared", "r squared")
    sha = workspaces.commit_to_branch(task_id, BRANCH, "Fix it", AUTHOR)
    workspaces.push_branch(task_id, BRANCH)
    assert git_out(origin_path(workspaces), "rev-parse", BRANCH) == sha
    assert workspaces.base_branch(task_id) == "main"
    assert git_out(origin_path(workspaces), "rev-parse", "main") != sha  # main itself is untouched


@pytest.mark.parametrize("bad", ["main", "autonoms/x", "../evil", "autonoms/issue-1-ZZZZZZZZ", "autonoms/issue-1-abcdef12; rm -rf", "--force"])
def test_only_our_own_branch_names_reach_git(ws, bad):
    workspaces, task_id = ws
    for action in (lambda: workspaces.commit_to_branch(task_id, bad, "m", AUTHOR), lambda: workspaces.push_branch(task_id, bad)):
        with pytest.raises(InvalidPath):
            action()


def test_push_is_never_forced(ws, env):
    workspaces, first = ws
    second = str(uuid.uuid4())
    workspaces.create(second, "o", "r")
    workspaces.replace_in_file(first, "formulas.md", "radius squared", "one")
    workspaces.replace_in_file(second, "formulas.md", "radius squared", "two")
    for task_id in (first, second):
        workspaces.commit_to_branch(task_id, BRANCH, "Fix", AUTHOR)
    workspaces.push_branch(first, BRANCH)
    before = git_out(origin_path(workspaces), "rev-parse", BRANCH)
    with pytest.raises(GitError):
        workspaces.push_branch(second, BRANCH)  # different history on the same branch name
    assert git_out(origin_path(workspaces), "rev-parse", BRANCH) == before  # the remote branch was not overwritten


# --- GitHub: opening the pull request ------------------------------------------------------------


def gh(handler):
    return GitHubClient(httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.github.com"))


def test_create_pull_request_sends_a_draft_by_default_shape_and_parses_the_answer():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content), path=request.url.path, method=request.method)
        return httpx.Response(201, json={"number": 7, "html_url": "https://github.com/o/r/pull/7"})

    assert gh(handler).create_pull_request("o", "r", BRANCH, "main", "T", "B") == (7, "https://github.com/o/r/pull/7")
    assert seen == {"title": "T", "head": BRANCH, "base": "main", "body": "B", "draft": True, "path": "/repos/o/r/pulls", "method": "POST"}


@pytest.mark.parametrize("status", [401, 403, 404])
def test_a_refused_token_gets_an_actionable_message(status):
    with pytest.raises(GitHubError, match="Pull requests: write"):
        gh(lambda r: httpx.Response(status, json={"message": "no"})).create_pull_request("o", "r", BRANCH, "main", "T", "B")


def test_pull_request_failures():
    with pytest.raises(PullRequestExists):
        gh(lambda r: httpx.Response(422, json={"errors": [{"message": "A pull request already exists for o:x"}]})).create_pull_request("o", "r", BRANCH, "main", "T", "B")
    with pytest.raises(GitHubError, match="no difference|difference"):
        gh(lambda r: httpx.Response(422, json={"errors": [{"message": "No commits between main and x"}]})).create_pull_request("o", "r", BRANCH, "main", "T", "B")
    with pytest.raises(GitHubError):
        gh(lambda r: httpx.Response(500)).create_pull_request("o", "r", BRANCH, "main", "T", "B")
    with pytest.raises(GitHubError):
        gh(lambda r: (_ for _ in ()).throw(httpx.ConnectError("down"))).create_pull_request("o", "r", BRANCH, "main", "T", "B")


def test_find_pull_request():
    found = gh(lambda r: httpx.Response(200, json=[{"number": 3, "html_url": "u"}]) if r.url.params["head"] == f"o:{BRANCH}" else httpx.Response(200, json=[]))
    assert found.find_pull_request("o", "r", BRANCH) == (3, "u")
    assert found.find_pull_request("o", "r", "autonoms/issue-2-aaaaaaaa") is None


# --- the whole flow through the API ---------------------------------------------------------------


@pytest.fixture
def papi(fake_github, db_session, env):
    workspaces, indexes, _ = env
    llm = FakeLLM()
    state = {"checker": FakeChecker(), "pulls": [], "pr_answers": []}
    app.dependency_overrides[get_workspaces] = lambda: workspaces
    app.dependency_overrides[get_indexes] = lambda: indexes
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_checker] = lambda: state["checker"]

    def handler(request):
        if request.url.path.endswith("/pulls"):
            if request.method == "GET":
                return httpx.Response(200, json=[{"number": 7, "html_url": "https://github.com/o/r/pull/7"}])
            state["pulls"].append(json.loads(request.content))
            answer = state["pr_answers"].pop(0) if state["pr_answers"] else httpx.Response(201, json={"number": 7, "html_url": "https://github.com/o/r/pull/7"})
            return answer
        if request.url.path.endswith("/comments"):
            return httpx.Response(200, json=[])
        return httpx.Response(200, json=make_issue(number=1, title="Circle area is wrong", body="formulas.md says radius squared"))

    client = fake_github(handler)
    client.llm, client.state, client.workspaces = llm, state, workspaces
    yield client
    for dep in (get_workspaces, get_indexes, get_llm, get_checker):
        app.dependency_overrides.pop(dep, None)


EDIT = [call("replace_in_file", path="formulas.md", old_text="radius squared", new_text="r squared"), call("finish", summary="Reworded the formula.")]


def finished_task(api, replies=EDIT, **run_body):
    task_id = api.post("/tasks", json={"owner": "o", "repo": "r", "number": 1}).json()["id"]
    api.workspaces.create(task_id, "o", "r")
    api.llm.replies = list(replies)
    assert api.post(f"/tasks/{task_id}/run", json=run_body or None).status_code == 200
    return task_id


def approve(api, task_id, **extra):
    review = api.get(f"/tasks/{task_id}/review").json()
    return api.post(f"/tasks/{task_id}/publish", json={"fingerprint": review["fingerprint"], "approved_by": "Alice", **extra})


def test_review_needs_a_finished_run(papi):
    task_id = papi.post("/tasks", json={"owner": "o", "repo": "r", "number": 1}).json()["id"]
    assert papi.get(f"/tasks/{task_id}/review").status_code == 409
    assert papi.post(f"/tasks/{task_id}/publish", json={"fingerprint": "0" * 64, "approved_by": "A"}).status_code == 409
    assert papi.get(f"/tasks/{uuid.uuid4()}/review").status_code == 404


def test_reading_the_review_publishes_nothing(papi):
    task_id = finished_task(papi)
    review = papi.get(f"/tasks/{task_id}/review").json()
    assert review["can_publish"] and review["files"] == ["formulas.md"] and review["publication"] is None
    assert papi.state["pulls"] == [] and git_out(origin_path(papi.workspaces), "branch", "--list", "autonoms/*") == ""


def test_publishing_after_approval(papi):
    task_id = finished_task(papi)
    response = approve(papi, task_id)
    assert response.status_code == 201
    publication = response.json()
    assert (publication["pr_number"], publication["base"], publication["draft"], publication["approved_by"]) == (7, "main", True, "Alice")
    pull = papi.state["pulls"][0]
    assert pull["head"] == publication["branch"] and pull["base"] == "main" and pull["draft"] is True
    assert pull["title"] == "Fix #1: Circle area is wrong" and "Closes #1" in pull["body"] and "approval by **Alice**" in pull["body"]
    assert git_out(origin_path(papi.workspaces), "rev-parse", publication["branch"])  # the branch is on the remote
    assert papi.get(f"/tasks/{task_id}/review").json()["publication"]["pr_url"] == "https://github.com/o/r/pull/7"
    assert approve(papi, task_id).status_code == 409  # never twice
    assert len(papi.state["pulls"]) == 1


def test_a_ready_for_review_pr_only_when_asked(papi):
    task_id = finished_task(papi)
    approve(papi, task_id, draft=False)
    assert papi.state["pulls"][0]["draft"] is False


def test_approval_must_quote_the_exact_fingerprint(papi):
    task_id = finished_task(papi)
    body = {"approved_by": "Alice"}
    assert papi.post(f"/tasks/{task_id}/publish", json={**body, "fingerprint": "short"}).status_code == 422
    assert papi.post(f"/tasks/{task_id}/publish", json=body).status_code == 422
    assert papi.post(f"/tasks/{task_id}/publish", json={**body, "fingerprint": "0" * 64}).status_code == 409
    assert papi.post(f"/tasks/{task_id}/publish", json={"fingerprint": papi.get(f"/tasks/{task_id}/review").json()["fingerprint"], "approved_by": ""}).status_code == 422
    assert papi.state["pulls"] == [] and git_out(origin_path(papi.workspaces), "branch", "--list", "autonoms/*") == ""


def test_an_approval_does_not_survive_a_changed_run(papi):
    task_id = finished_task(papi)
    old = papi.get(f"/tasks/{task_id}/review").json()["fingerprint"]
    papi.llm.replies = [call("finish", summary="A different summary.")]
    papi.post(f"/tasks/{task_id}/run", json={"restart": True})
    response = papi.post(f"/tasks/{task_id}/publish", json={"fingerprint": old, "approved_by": "Alice"})
    assert response.status_code == 409 and "Review again" in response.json()["detail"]


def test_warnings_must_be_acknowledged(papi):
    PASSED_THEN_FAILED = FakeChecker(PASSED, FAILED)  # baseline fine, afterwards something is newly wrong
    papi.state["checker"] = PASSED_THEN_FAILED
    task_id = finished_task(papi, max_repairs=0)
    review = papi.get(f"/tasks/{task_id}/review").json()
    assert review["needs_acknowledgement"] and any(r["level"] == "warn" for r in review["risks"])
    assert approve(papi, task_id).status_code == 409
    assert papi.state["pulls"] == []
    assert approve(papi, task_id, acknowledge_warnings=True).status_code == 201
    assert "**Warning:**" in papi.state["pulls"][0]["body"]


def test_runs_that_did_not_finish_cannot_be_published(papi):
    task_id = finished_task(papi, replies=[call("read_file", path="bodmas.md")] * 2, max_steps=2)
    review = papi.get(f"/tasks/{task_id}/review").json()
    assert not review["can_publish"] and review["blockers"]
    response = papi.post(f"/tasks/{task_id}/publish", json={"fingerprint": review["fingerprint"], "approved_by": "Alice"})
    assert response.status_code == 409 and "Cannot publish" in response.json()["detail"]


def test_files_changed_after_the_run_are_not_published(papi):
    task_id = finished_task(papi)
    review = papi.get(f"/tasks/{task_id}/review").json()
    papi.workspaces.replace_in_file(task_id, "bodmas.md", "BODMAS", "SNEAKY")  # an edit that was never reviewed
    response = papi.post(f"/tasks/{task_id}/publish", json={"fingerprint": review["fingerprint"], "approved_by": "Alice"})
    assert response.status_code == 409 and "workspace changed" in response.json()["detail"]
    assert papi.state["pulls"] == [] and git_out(origin_path(papi.workspaces), "branch", "--list", "autonoms/*") == ""


def test_a_failed_pull_request_can_be_retried_without_a_second_commit(papi):
    papi.state["pr_answers"] = [httpx.Response(403, json={"message": "Resource not accessible"})]
    task_id = finished_task(papi)
    first = approve(papi, task_id)
    assert first.status_code == 502 and "Pull requests: write" in first.json()["detail"] and "safe" in first.json()["detail"]
    branch = papi.get(f"/tasks/{task_id}/review").json()["branch"]
    assert git_out(origin_path(papi.workspaces), "rev-parse", branch)  # the push already happened
    assert papi.get(f"/tasks/{task_id}/review").json()["publication"] is None
    assert approve(papi, task_id).status_code == 201
    assert git_out(papi.workspaces.path(task_id), "rev-list", "--count", "main..HEAD") == "1"


def test_an_already_open_pull_request_is_adopted(papi):
    papi.state["pr_answers"] = [httpx.Response(422, json={"errors": [{"message": "A pull request already exists for o:x"}]})]
    task_id = finished_task(papi)
    assert approve(papi, task_id).json()["pr_number"] == 7


def test_the_token_never_appears_in_any_response(papi):
    task_id = finished_task(papi)
    text = papi.get(f"/tasks/{task_id}/review").text + approve(papi, task_id).text
    assert "ghp_" not in text and "Authorization" not in text


def test_a_permission_denied_push_says_how_to_fix_the_token(papi, monkeypatch):
    """Found in the real run: the token could read the repository but GitHub answered 403 to a push."""
    def denied(self, task_id, branch):
        raise GitError("git command failed", denied=True)

    monkeypatch.setattr(type(papi.workspaces), "push_branch", denied)
    task_id = finished_task(papi)
    response = approve(papi, task_id)
    assert response.status_code == 502 and "Contents: Read and write" in response.json()["detail"]
    assert papi.state["pulls"] == [] and papi.get(f"/tasks/{task_id}/review").json()["publication"] is None


@pytest.mark.parametrize(
    "stderr, denied",
    [
        ("remote: Permission to o/r.git denied to someone.\nfatal: unable to access: The requested URL returned error: 403", True),
        ("fatal: unable to access 'https://github.com/o/r.git/': The requested URL returned error: 403", True),
        ("fatal: could not resolve host", False),
    ],
)
def test_git_errors_are_classified(monkeypatch, tmp_path, stderr, denied):
    from autonoms import workspace as w

    monkeypatch.setattr(w.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 128, "", stderr))
    with pytest.raises(GitError) as err:
        w.Workspaces(tmp_path, "file:///x")._git(["push"])
    assert err.value.denied is denied
