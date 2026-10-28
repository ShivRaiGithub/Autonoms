import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from autonoms.db import Base, get_session
from autonoms.github import GitHubClient
from autonoms.main import app, get_github


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
def db_session():
    """An in-memory SQLite database standing in for Postgres: no server needed."""
    # StaticPool keeps one connection, otherwise each connection sees a fresh empty DB.
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    session = Session(engine)

    def override():
        yield session

    app.dependency_overrides[get_session] = override
    yield session
    app.dependency_overrides.pop(get_session, None)
    session.close()
