import uuid

import httpx
import pytest

from autonoms.main import app, get_workspaces
from autonoms.workspace import Workspaces
from conftest import make_issue

TASK = str(uuid.uuid4())


def test_clone_and_summary(workspaces):
    assert workspaces.create(TASK, "o", "r") is True
    summary = workspaces.summary(TASK)
    assert summary.branch == "main"
    assert len(summary.head) == 40
    assert {f.path for f in summary.files} >= {"arithmetic.md", "notes/bodmas.md"}
    assert summary.manifests == ["requirements.txt"]
    assert [c.subject for c in summary.commits] == ["fix division", "first commit"]
    assert summary.commits[0].author == "Tester"


def test_second_create_is_a_noop(workspaces):
    assert workspaces.create(TASK, "o", "r") is True
    assert workspaces.create(TASK, "o", "r") is False


def test_missing_repo_raises_and_leaves_no_folder(workspaces):
    from autonoms.workspace import GitError

    with pytest.raises(GitError) as err:
        workspaces.create(TASK, "o", "nope")
    assert err.value.not_found
    assert not workspaces.exists(TASK)


def test_read_file(workspaces):
    workspaces.create(TASK, "o", "r")
    assert workspaces.read_file(TASK, "notes/bodmas.md") == "2 + 3 x 4 = 20\n"


@pytest.mark.parametrize(
    "bad", ["../../etc/passwd", "/etc/passwd", ".git/config", "notes/../../x", "", "link-outside", "nope.md", "notes"]
)
def test_bad_paths_are_rejected(workspaces, bad):
    from autonoms.workspace import InvalidPath

    workspaces.create(TASK, "o", "r")
    with pytest.raises(InvalidPath):
        workspaces.read_file(TASK, bad)


def test_binary_and_big_files(workspaces):
    from autonoms.workspace import BinaryFile, FileTooLarge

    workspaces.create(TASK, "o", "r")
    with pytest.raises(BinaryFile):
        workspaces.read_file(TASK, "image.bin")
    with pytest.raises(FileTooLarge):
        workspaces.read_file(TASK, "big.txt")


def test_no_workspace_and_remove(workspaces):
    from autonoms.workspace import NoWorkspace

    with pytest.raises(NoWorkspace):
        workspaces.summary(TASK)
    workspaces.create(TASK, "o", "r")
    assert workspaces.remove(TASK) is True
    assert workspaces.remove(TASK) is False


def test_task_id_cannot_escape_the_workspace_folder(workspaces):
    with pytest.raises(ValueError):
        workspaces.path("../../tmp")


def test_token_only_goes_to_github():
    github = Workspaces("/tmp/x", "https://github.com", token="ghp_secret")
    other = Workspaces("/tmp/x", "file:///somewhere", token="ghp_secret")
    assert "GIT_CONFIG_VALUE_0" in github._env()
    assert "GIT_CONFIG_VALUE_0" not in other._env()
    # never on a command line: it is only in the environment dict
    assert "ghp_secret" not in repr(github.__dict__.get("root"))


# --- through the HTTP API ------------------------------------------------------


def issue_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/comments"):
        return httpx.Response(200, json=[])
    return httpx.Response(200, json=make_issue(number=1))


@pytest.fixture
def api(fake_github, db_session, workspaces):
    app.dependency_overrides[get_workspaces] = lambda: workspaces
    client = fake_github(issue_handler)
    yield client
    app.dependency_overrides.pop(get_workspaces, None)


def new_task(api):
    return api.post("/tasks", json={"owner": "o", "repo": "r", "number": 1}).json()["id"]


def test_api_workspace_flow(api, workspaces):
    task_id = new_task(api)
    assert api.get(f"/tasks/{task_id}/repo").status_code == 404  # no clone yet
    first = api.post(f"/tasks/{task_id}/workspace")
    assert first.status_code == 201
    assert first.json()["branch"] == "main"
    assert api.post(f"/tasks/{task_id}/workspace").status_code == 200
    assert api.get(f"/tasks/{task_id}/repo").json()["file_count"] == 6
    body = api.get(f"/tasks/{task_id}/files", params={"path": "arithmetic.md"}).json()
    assert body["content"] == "12 / 4 = 3\n"
    assert api.delete(f"/tasks/{task_id}/workspace").status_code == 204
    assert not workspaces.exists(task_id)


def test_api_file_errors(api):
    task_id = new_task(api)
    api.post(f"/tasks/{task_id}/workspace")
    get = lambda p: api.get(f"/tasks/{task_id}/files", params={"path": p}).status_code  # noqa: E731
    assert get("../../etc/passwd") == 400
    assert get("link-outside") == 400
    assert get("image.bin") == 415
    assert get("big.txt") == 413


def test_api_unknown_task_and_missing_repo(api):
    assert api.post(f"/tasks/{uuid.uuid4()}/workspace").status_code == 404


def test_deleting_a_task_removes_its_workspace(api, workspaces):
    task_id = new_task(api)
    api.post(f"/tasks/{task_id}/workspace")
    assert workspaces.exists(task_id)
    assert api.delete(f"/tasks/{task_id}").status_code == 204
    assert not workspaces.exists(task_id)


def test_a_workspace_does_not_exist_until_its_clone_is_complete(workspaces, monkeypatch):
    """Found through the web UI: a request arriving mid-clone saw an empty workspace."""
    seen = []
    real_git = Workspaces._git

    def spy(self, args, cwd=None):
        if args and args[0] == "clone":
            seen.append(self.exists(TASK))  # asked while git is cloning
        return real_git(self, args, cwd)

    monkeypatch.setattr(Workspaces, "_git", spy)
    workspaces.create(TASK, "o", "r")
    assert seen == [False] and workspaces.exists(TASK)
    assert not (workspaces.root / f".{TASK}.partial").exists()  # the temporary folder is gone


def test_a_failed_clone_leaves_neither_a_workspace_nor_a_temporary_folder(workspaces):
    from autonoms.workspace import GitError

    with pytest.raises(GitError):
        workspaces.create(TASK, "o", "nope")
    assert not workspaces.exists(TASK) and not (workspaces.root / f".{TASK}.partial").exists()
