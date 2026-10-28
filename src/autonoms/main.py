"""FastAPI application: the HTTP interface to Autonoms."""

from collections.abc import Iterator
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from autonoms import search, tasks, workspace
from autonoms.agent import Agent
from autonoms.checker import Checker
from autonoms.llm import LLM, OllamaClient
from autonoms.config import get_settings
from autonoms.db import get_session, init_db
from autonoms.github import GitHubClient, GitHubError, IssueNotFound, NotAnIssue, RateLimited
from autonoms.models import (
    NAME,
    AgentRun,
    CheckResult,
    IndexInfo,
    Issue,
    RepoSummary,
    RunRequest,
    SearchResult,
    Task,
    TaskCreate,
)
from autonoms.sandbox import Sandbox
from autonoms.search import SearchIndexes
from autonoms.workspace import Workspaces


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


def get_workspaces() -> Workspaces:
    settings = get_settings()
    token = settings.github_token.get_secret_value() if settings.github_token else None
    return Workspaces(settings.workspaces_dir, settings.github_clone_url, token)


def get_indexes(workspaces: Annotated[Workspaces, Depends(get_workspaces)]) -> SearchIndexes:
    settings = get_settings()
    return get_search_indexes(settings.indexes_dir, workspaces, settings.embedding_model)


@lru_cache
def _cached_indexes(root, workspaces_root, clone_base, model) -> SearchIndexes:
    return SearchIndexes(root, Workspaces(workspaces_root, clone_base), model)


def get_search_indexes(root, workspaces: Workspaces, model: str) -> SearchIndexes:
    # One SearchIndexes per process so loaded indexes stay in memory between requests.
    return _cached_indexes(root, workspaces.root, workspaces.clone_base, model)


def get_checker(workspaces: Annotated[Workspaces, Depends(get_workspaces)]) -> Checker:
    settings = get_settings()
    sandbox = Sandbox(settings.sandbox_image, settings.sandbox_timeout_seconds, settings.sandbox_memory)
    return Checker(sandbox, workspaces)


def get_llm() -> LLM:
    settings = get_settings()
    return OllamaClient.create(settings.ollama_base_url, settings.ollama_model, settings.ollama_think)


NameParam = Annotated[str, Path(pattern=NAME, max_length=100)]
SessionDep = Annotated[Session, Depends(get_session)]
GitHubDep = Annotated[GitHubClient, Depends(get_github)]
WorkspacesDep = Annotated[Workspaces, Depends(get_workspaces)]
IndexesDep = Annotated[SearchIndexes, Depends(get_indexes)]


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
def remove_task(
    task_id: str, session: SessionDep, workspaces: WorkspacesDep, indexes: IndexesDep
) -> Response:
    if not tasks.delete_task(session, task_id):
        raise HTTPException(404, "Task not found")
    workspaces.remove(task_id)  # a task's workspace and index go with it
    indexes.remove(task_id)
    return Response(status_code=204)


def _task_or_404(session: Session, task_id: str) -> Task:
    task = tasks.get_task(session, task_id)
    if task is None:
        raise HTTPException(404, "Task not found")
    return task


@app.post("/tasks/{task_id}/workspace", response_model=RepoSummary)
def create_workspace(
    task_id: str, response: Response, session: SessionDep, workspaces: WorkspacesDep
) -> RepoSummary:
    task = _task_or_404(session, task_id)
    try:
        created = workspaces.create(task_id, task.owner, task.repo)
        response.status_code = 201 if created else 200  # 200: it was already there
        return workspaces.summary(task_id)
    except workspace.GitError as exc:
        if exc.not_found:
            raise HTTPException(404, "Repository not found")
        raise HTTPException(502, "Could not clone the repository")


@app.get("/tasks/{task_id}/repo", response_model=RepoSummary)
def read_repo(task_id: str, session: SessionDep, workspaces: WorkspacesDep) -> RepoSummary:
    _task_or_404(session, task_id)
    try:
        return workspaces.summary(task_id)
    except workspace.NoWorkspace:
        raise HTTPException(404, "No workspace yet: POST /tasks/{id}/workspace first")
    except workspace.GitError:
        raise HTTPException(502, "git failed while inspecting the workspace")


@app.get("/tasks/{task_id}/files")
def read_file(task_id: str, path: str, session: SessionDep, workspaces: WorkspacesDep) -> dict:
    _task_or_404(session, task_id)
    try:
        return {"path": path, "content": workspaces.read_file(task_id, path)}
    except workspace.NoWorkspace:
        raise HTTPException(404, "No workspace yet: POST /tasks/{id}/workspace first")
    except workspace.InvalidPath:
        raise HTTPException(400, "Invalid path")
    except workspace.FileTooLarge:
        raise HTTPException(413, "File too large")
    except workspace.BinaryFile:
        raise HTTPException(415, "Not a text file")


@app.delete("/tasks/{task_id}/workspace", status_code=204)
def remove_workspace(
    task_id: str, session: SessionDep, workspaces: WorkspacesDep, indexes: IndexesDep
) -> Response:
    _task_or_404(session, task_id)
    indexes.remove(task_id)  # an index of files that no longer exist is misleading
    if not workspaces.remove(task_id):
        raise HTTPException(404, "No workspace")
    return Response(status_code=204)


@app.post("/tasks/{task_id}/index", response_model=IndexInfo)
def build_index(
    task_id: str, session: SessionDep, indexes: IndexesDep
) -> IndexInfo:
    _task_or_404(session, task_id)
    try:
        return indexes.build(task_id)
    except workspace.NoWorkspace:
        raise HTTPException(404, "No workspace yet: POST /tasks/{id}/workspace first")


def _run_search(indexes: SearchIndexes, task_id: str, query: str, mode: str, top_k: int) -> SearchResult:
    try:
        return indexes.search(task_id, query, mode, top_k)
    except search.NoIndex:
        raise HTTPException(404, "No index yet: POST /tasks/{id}/index first")


@app.get("/tasks/{task_id}/search", response_model=SearchResult)
def search_task(
    task_id: str,
    session: SessionDep,
    indexes: IndexesDep,
    q: Annotated[str, Query(min_length=1, max_length=2000)],
    mode: Annotated[str, Query(pattern="^(keyword|semantic|hybrid)$")] = "hybrid",
    top_k: Annotated[int, Query(ge=1, le=20)] = 5,
) -> SearchResult:
    _task_or_404(session, task_id)
    return _run_search(indexes, task_id, q, mode, top_k)


@app.get("/tasks/{task_id}/context", response_model=SearchResult)
def issue_context(
    task_id: str,
    session: SessionDep,
    indexes: IndexesDep,
    top_k: Annotated[int, Query(ge=1, le=20)] = 3,
) -> SearchResult:
    """The code most relevant to this task's issue: the issue text is the query."""
    task = _task_or_404(session, task_id)
    query = f"{task.issue.title}\n{task.issue.body}"[:2000]
    return _run_search(indexes, task_id, query, "hybrid", top_k)


_running: set[str] = set()  # task ids with a run in progress (this process only)


@app.post("/tasks/{task_id}/run", response_model=AgentRun)
def run_agent(
    task_id: str,
    session: SessionDep,
    workspaces: WorkspacesDep,
    indexes: IndexesDep,
    llm: Annotated[LLM, Depends(get_llm)],
    checker: Annotated[Checker, Depends(get_checker)],
    body: RunRequest | None = None,
) -> AgentRun:
    """Let the model try to resolve the task's issue. Takes minutes on a local model."""
    task = _task_or_404(session, task_id)
    if not workspaces.exists(task_id):
        raise HTTPException(404, "No workspace yet: POST /tasks/{id}/workspace first")
    if task_id in _running:
        raise HTTPException(409, "A run is already in progress for this task")
    _running.add(task_id)
    try:
        indexes.build(task_id)  # always start from an index that matches the files
        return Agent(llm, workspaces, indexes, checker).run(task, (body or RunRequest()).max_steps)
    finally:
        _running.discard(task_id)


@app.post("/tasks/{task_id}/check", response_model=CheckResult)
def check_task(
    task_id: str, session: SessionDep, checker: Annotated[Checker, Depends(get_checker)]
) -> CheckResult:
    """Run the arithmetic check on the task's workspace, inside the sandbox."""
    _task_or_404(session, task_id)
    try:
        return checker.run(task_id)
    except workspace.NoWorkspace:
        raise HTTPException(404, "No workspace yet: POST /tasks/{id}/workspace first")
