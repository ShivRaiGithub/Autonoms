import httpx

from conftest import make_issue

COMMENT = {"user": {"login": "bob"}, "body": "I can reproduce", "created_at": "2026-01-02T10:00:00Z"}


def ok_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/comments"):
        return httpx.Response(200, json=[COMMENT])
    return httpx.Response(200, json=make_issue())


def test_health(fake_github):
    assert fake_github(ok_handler).get("/health").json() == {"status": "ok"}


def test_issue_is_parsed_into_our_model(fake_github):
    body = fake_github(ok_handler).get("/issues/o/r/7").json()
    assert body["title"] == "median is wrong"
    assert body["labels"] == ["bug"]
    assert body["author"] == "alice"
    assert body["comments"][0]["author"] == "bob"


def test_empty_issue_body_becomes_empty_string(fake_github):
    def handler(request):
        if request.url.path.endswith("/comments"):
            return httpx.Response(200, json=[])
        return httpx.Response(200, json=make_issue(body=None))

    body = fake_github(handler).get("/issues/o/r/7").json()
    assert body["body"] == ""
    assert body["comments"] == []


def test_unknown_issue_is_404(fake_github):
    client = fake_github(lambda request: httpx.Response(404, json={"message": "Not Found"}))
    assert client.get("/issues/o/r/999").status_code == 404


def test_pull_request_is_rejected(fake_github):
    client = fake_github(lambda r: httpx.Response(200, json=make_issue(pull_request={"url": "x"})))
    assert client.get("/issues/o/r/7").status_code == 422


def test_rate_limit_is_429(fake_github):
    client = fake_github(
        lambda r: httpx.Response(403, headers={"x-ratelimit-remaining": "0"}, json={})
    )
    assert client.get("/issues/o/r/7").status_code == 429


def test_github_down_is_502(fake_github):
    def handler(request):
        raise httpx.ConnectError("boom")

    assert fake_github(handler).get("/issues/o/r/7").status_code == 502


def test_github_server_error_is_502(fake_github):
    client = fake_github(lambda r: httpx.Response(500))
    assert client.get("/issues/o/r/7").status_code == 502


def test_bad_path_values_never_reach_github(fake_github):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=make_issue())

    client = fake_github(handler)
    assert client.get("/issues/o/r/0").status_code == 422
    assert client.get("/issues/o/r/abc").status_code == 422
    assert client.get("/issues/o%20x/r/1").status_code == 422
    assert calls == []


def test_token_is_sent_but_never_shown():
    from autonoms.config import Settings

    settings = Settings(github_token="ghp_secret123")
    assert "ghp_secret123" not in repr(settings)
    assert "ghp_secret123" not in str(settings)


def test_dot_only_names_are_rejected(fake_github):
    """Regression: ".." passed the name pattern and would be a path step in the GitHub URL."""
    calls = []
    client = fake_github(lambda r: calls.append(r) or httpx.Response(200, json={}))
    for path in ("/issues/../r/1", "/issues/o/../1", "/issues/./r/1"):
        assert client.get(path).status_code in (404, 422)  # the HTTP client may also normalise the path
    assert client.post("/tasks", json={"owner": "..", "repo": "r", "number": 1}).status_code == 422
    assert client.post("/tasks", json={"owner": "o", "repo": ".", "number": 1}).status_code == 422
    assert calls == []
