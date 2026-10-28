"""FastAPI application: the HTTP interface to Autonoms."""

from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Path, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from autonoms import tasks
from autonoms.config import get_settings
from autonoms.db import get_session, init_db
from autonoms.github import GitHubClient, GitHubError, IssueNotFound, NotAnIssue, RateLimited
from autonoms.models import NAME, Issue, Task, TaskCreate


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()  # runs once when the server starts
    yield


app = FastAPI(title="Autonoms", lifespan=lifespan)


@app.exception_handler(OperationalError)
def database_unavailable(request: Request, exc: OperationalError) -> JSONResponse:
    # Don't echo exc: its text can contain the connection details.
    return JSONResponse({"detail": "Database unavailable"}, status_code=503)


def get_github() -> Iterator[GitHubClient]:
    settings = get_settings()
    token = settings.github_token.get_secret_value() if settings.github_token else None
    client = GitHubClient.create(settings.github_api_url, token)
    try:
        yield client
    finally:
        client.close()


NameParam = Annotated[str, Path(pattern=NAME, max_length=100)]
SessionDep = Annotated[Session, Depends(get_session)]
GitHubDep = Annotated[GitHubClient, Depends(get_github)]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/issues/{owner}/{repo}/{number}", response_model=Issue)
def read_issue(
    owner: NameParam,
    repo: NameParam,
    number: Annotated[int, Path(ge=1)],
    github: GitHubDep,
) -> Issue:
    return fetch_issue(github, owner, repo, number)


def fetch_issue(github: GitHubClient, owner: str, repo: str, number: int) -> Issue:
    """Fetch an issue, translating GitHub failures into HTTP errors."""
    try:
        return github.get_issue(owner, repo, number)
    except IssueNotFound:
        raise HTTPException(404, "Issue not found")
    except NotAnIssue:
        raise HTTPException(422, "That number is a pull request, not an issue")
    except RateLimited:
        raise HTTPException(429, "GitHub rate limit exceeded")
    except GitHubError as exc:
        raise HTTPException(502, str(exc))


@app.post("/tasks", response_model=Task, status_code=201)
def create_task(body: TaskCreate, github: GitHubDep, session: SessionDep) -> Task:
    issue = fetch_issue(github, body.owner, body.repo, body.number)
    return tasks.create_task(session, issue)


@app.get("/tasks", response_model=list[Task])
def read_tasks(session: SessionDep) -> list[Task]:
    return tasks.list_tasks(session)


@app.get("/tasks/{task_id}", response_model=Task)
def read_task(task_id: str, session: SessionDep) -> Task:
    task = tasks.get_task(session, task_id)
    if task is None:
        raise HTTPException(404, "Task not found")
    return task


@app.delete("/tasks/{task_id}", status_code=204)
def remove_task(task_id: str, session: SessionDep) -> Response:
    if not tasks.delete_task(session, task_id):
        raise HTTPException(404, "Task not found")
    return Response(status_code=204)
