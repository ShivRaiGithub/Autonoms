import os
import subprocess

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from autonoms.db import Base, get_session
from autonoms.github import GitHubClient
from autonoms.main import app, get_checkpointer, get_github
from autonoms.workspace import Workspaces


def make_issue(**overrides):
    data = {
        "number": 7,
        "title": "median is wrong",
        "body": "median([1,2,3,4]) returns 3",
        "state": "open",
        "user": {"login": "alice"},
        "labels": [{"name": "bug"}],
        "html_url": "https://github.com/o/r/issues/7",
        "created_at": "2026-01-01T10:00:00Z",
    }
    return data | overrides


@pytest.fixture
def fake_github():
    """Build an app client whose GitHub is a fake: pass a handler(request) -> Response."""

    def build(handler):
        github = GitHubClient(
            httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.github.com")
        )
        app.dependency_overrides[get_github] = lambda: github
        return TestClient(app)

    yield build
    app.dependency_overrides.clear()


@pytest.fixture
def db_session(tmp_path):
    """A SQLite database standing in for Postgres: no server needed.

    Like the real app, every request gets its OWN session. (A single shared session was fine
    while one request ran at a time, but the web UI's polling timer overlaps other requests, and a
    Session must not be used from two threads at once.) A file, not ":memory:", so that separate
    connections see the same data.
    """
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)

    def override():
        with Session(engine) as per_request:
            yield per_request

    app.dependency_overrides[get_session] = override
    with Session(engine) as session:  # for tests that want to look at the database directly
        yield session
    app.dependency_overrides.pop(get_session, None)
    engine.dispose()


GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "Tester", "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "Tester", "GIT_COMMITTER_EMAIL": "t@example.com",
    "GIT_CONFIG_GLOBAL": os.devnull,
}  # fmt: skip


def git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, env=GIT_ENV, check=True, capture_output=True)


@pytest.fixture
def server(tmp_path):
    """A local folder acting as GitHub: <server>/o/r.git is a real repository."""
    repo = tmp_path / "server" / "o" / "r.git"
    repo.mkdir(parents=True)
    git(repo, "init", "-q", "-b", "main")
    (repo / "arithmetic.md").write_text("12 / 4 = 4\n")
    (repo / "requirements.txt").write_text("pytest\n")
    (repo / "notes").mkdir()
    (repo / "notes" / "bodmas.md").write_text("2 + 3 x 4 = 20\n")
    (repo / "image.bin").write_bytes(b"\xff\xfe\x00\x01")
    (repo / "big.txt").write_text("x" * 300_000)
    os.symlink("/etc/hostname", repo / "link-outside")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "first commit")
    (repo / "arithmetic.md").write_text("12 / 4 = 3\n")
    git(repo, "commit", "-q", "-am", "fix division")
    return f"file://{tmp_path / 'server'}"


@pytest.fixture
def workspaces(tmp_path, server):
    return Workspaces(tmp_path / "workspaces", server)


@pytest.fixture
def notes_server(tmp_path):
    """Like `server`, but with small math notes to search."""
    repo = tmp_path / "notes_server" / "o" / "r.git"
    repo.mkdir(parents=True)
    git(repo, "init", "-q", "-b", "main")
    notes = {
        "pythagoras.md": "In a right-angled triangle the hypotenuse is the longest side. "
        "The square of the hypotenuse equals the sum of the squares of the other two sides.",
        "fractions.md": "To add fractions, first make the denominators the same, "
        "then add the numerators. For example 1/2 + 1/3 = 5/6.",
        "bodmas.md": "BODMAS is the order of operations: brackets, orders, division and "
        "multiplication, then addition and subtraction.",
        "formulas.md": "The area of a circle is pi times the radius squared.",
    }
    for name, text in notes.items():
        (repo / name).write_text(text + "\n")
    (repo / "image.bin").write_bytes(b"\xff\xfe\x00\x01")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "notes")
    return f"file://{tmp_path / 'notes_server'}"


@pytest.fixture(autouse=True)
def memory_checkpointer():
    """Every test gets an in-memory checkpoint store, so none can touch real Postgres."""
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    app.dependency_overrides[get_checkpointer] = lambda: saver
    yield saver
    app.dependency_overrides.pop(get_checkpointer, None)

