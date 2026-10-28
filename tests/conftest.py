import httpx
import pytest
from fastapi.testclient import TestClient

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
