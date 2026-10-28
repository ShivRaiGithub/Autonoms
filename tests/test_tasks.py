import httpx
from sqlalchemy.exc import OperationalError

from autonoms.db import get_session
from autonoms.main import app
from conftest import make_issue


def handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/comments"):
        return httpx.Response(200, json=[])
    number = int(request.url.path.rsplit("/", 1)[1])
    if number == 404:
        return httpx.Response(404, json={})
    return httpx.Response(200, json=make_issue(number=number))


def test_create_task_stores_issue_snapshot(fake_github, db_session):
    client = fake_github(handler)
    response = client.post("/tasks", json={"owner": "o", "repo": "r", "number": 7})
    assert response.status_code == 201
    task = response.json()
    assert task["status"] == "created"
    assert task["issue_title"] == "median is wrong"
    assert task["issue"]["labels"] == ["bug"]
    assert client.get(f"/tasks/{task['id']}").json() == task


def test_tasks_are_listed_newest_first(fake_github, db_session):
    client = fake_github(handler)
    first = client.post("/tasks", json={"owner": "o", "repo": "r", "number": 1}).json()
    second = client.post("/tasks", json={"owner": "o", "repo": "r", "number": 2}).json()
    assert [t["id"] for t in client.get("/tasks").json()] == [second["id"], first["id"]]


def test_empty_list(fake_github, db_session):
    assert fake_github(handler).get("/tasks").json() == []


def test_unknown_issue_stores_nothing(fake_github, db_session):
    client = fake_github(handler)
    assert client.post("/tasks", json={"owner": "o", "repo": "r", "number": 404}).status_code == 404
    assert client.get("/tasks").json() == []


def test_invalid_body_is_422_and_never_calls_github(fake_github, db_session):
    calls = []
    client = fake_github(lambda r: calls.append(r) or httpx.Response(200, json={}))
    for body in (
        {"owner": "o", "repo": "r", "number": 0},
        {"owner": "../x", "repo": "r", "number": 1},
        {"owner": "o", "repo": "r"},
    ):
        assert client.post("/tasks", json=body).status_code == 422
    assert calls == []


def test_unknown_task_is_404(fake_github, db_session):
    client = fake_github(handler)
    assert client.get("/tasks/nope").status_code == 404
    assert client.delete("/tasks/nope").status_code == 404


def test_delete_removes_only_that_task(fake_github, db_session):
    client = fake_github(handler)
    keep = client.post("/tasks", json={"owner": "o", "repo": "r", "number": 1}).json()
    gone = client.post("/tasks", json={"owner": "o", "repo": "r", "number": 2}).json()
    assert client.delete(f"/tasks/{gone['id']}").status_code == 204
    assert [t["id"] for t in client.get("/tasks").json()] == [keep["id"]]


def test_database_down_is_503_without_leaking_details(fake_github):
    def broken():
        raise OperationalError("select 1", {}, Exception("password=hunter2 host=db"))
        yield

    app.dependency_overrides[get_session] = broken
    try:
        response = fake_github(handler).get("/tasks")
    finally:
        app.dependency_overrides.pop(get_session, None)
    assert response.status_code == 503
    assert "hunter2" not in response.text


def test_database_url_uses_psycopg3_driver_and_is_secret():
    from autonoms.config import Settings

    settings = Settings(postgres_connection_string="postgresql://u:pw123@h:5432/d")
    assert settings.database_url == "postgresql+psycopg://u:pw123@h:5432/d"
    assert "pw123" not in repr(settings)
