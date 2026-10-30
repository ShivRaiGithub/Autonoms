"""FastAPI application: the HTTP interface to Autonoms."""

from collections.abc import Iterator
import threading
from datetime import UTC, datetime
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request, Response
from fastapi.responses import JSONResponse
import psycopg
from langgraph.checkpoint.base import BaseCheckpointSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool, PoolTimeout
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from autonoms import memory, report, search, tasks, workspace
from autonoms.agent import Agent
from autonoms.checker import Checker
from autonoms.evaluation import Evaluation, evaluate
from autonoms.llm import LLM, OllamaClient
from autonoms.config import get_settings
from autonoms.db import PublicationRow, get_session, init_db
from autonoms.github import GitHubClient, GitHubError, IssueNotFound, NotAnIssue, PullRequestExists, RateLimited
from autonoms.models import (
    NAME,
    AgentRun,
    CheckResult,
    IndexInfo,
    Issue,
    Memory,
    MemoryCreate,
    MemoryUpdate,
    Publication,
    PublishRequest,
    Review,
    RepoSummary,
    RunRequest,
    RunStatus,
    SearchResult,
    Task,
    TaskCreate,
)
from autonoms.sandbox import Sandbox
from autonoms.search import SearchIndexes
from autonoms.workflow import Workflow, WorkspaceChanged
from autonoms.workspace import Workspaces


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()  # runs once when the server starts
    get_checkpointer()  # create the checkpoint tables now, before concurrent requests arrive
    yield


app = FastAPI(title="Autonoms", lifespan=lifespan)


@app.exception_handler(OperationalError)
@app.exception_handler(psycopg.OperationalError)
@app.exception_handler(PoolTimeout)
def database_unavailable(request: Request, exc: Exception) -> JSONResponse:
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


_checkpointer: BaseCheckpointSaver | None = None
_checkpointer_lock = threading.Lock()


def get_checkpointer() -> BaseCheckpointSaver:
    """Saves workflow state in Postgres (tables are created on first use).

    The lock matters: requests run in parallel threads, and two of them creating
    the tables at the same moment crashed with a "duplicate key" error (found in
    the real run, invisible to tests that use an in-memory store).
    """
    global _checkpointer
    with _checkpointer_lock:
        if _checkpointer is None:
            _checkpointer = _create_checkpointer()
        return _checkpointer


def _create_checkpointer() -> BaseCheckpointSaver:
    from langgraph.checkpoint.postgres import PostgresSaver

    url = get_settings().postgres_connection_string
    if url is None:
        raise RuntimeError("POSTGRES_CONNECTION_STRING is not set in .env")
    pool = ConnectionPool(
        url.get_secret_value(),
        max_size=5,
        # the settings LangGraph's saver requires
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=True,
    )
    saver = PostgresSaver(pool)
    saver.setup()  # creates the checkpoint tables if missing; never alters existing ones
    return saver


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
    task_id: str,
    session: SessionDep,
    workspaces: WorkspacesDep,
    indexes: IndexesDep,
    checkpointer: Annotated[BaseCheckpointSaver, Depends(get_checkpointer)],
) -> Response:
    if not tasks.delete_task(session, task_id):
        raise HTTPException(404, "Task not found")
    workspaces.remove(task_id)  # a task's workspace and index go with it
    indexes.remove(task_id)
    checkpointer.delete_thread(task_id)  # and so does its saved run history
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


def get_workflow(
    llm: Annotated[LLM, Depends(get_llm)],
    workspaces: WorkspacesDep,
    indexes: IndexesDep,
    checker: Annotated[Checker, Depends(get_checker)],
    checkpointer: Annotated[BaseCheckpointSaver, Depends(get_checkpointer)],
    session: SessionDep,
) -> Workflow:
    def memories_for(task: Task) -> list[Memory]:
        text = f"{task.issue.title}\n{task.issue.body}"
        return memory.MemoryStore(session).relevant(f"{task.owner}/{task.repo}", text)

    return Workflow(
        llm, workspaces, Agent(llm, workspaces, indexes, checker), checker, checkpointer,
        compact_at=get_settings().context_compact_chars, memory_lookup=memories_for,
    )  # fmt: skip


WorkflowDep = Annotated[Workflow, Depends(get_workflow)]


@app.post("/tasks/{task_id}/run", response_model=AgentRun)
def run_agent(
    task_id: str,
    session: SessionDep,
    workspaces: WorkspacesDep,
    indexes: IndexesDep,
    flow: WorkflowDep,
    body: RunRequest | None = None,
) -> AgentRun:
    """Let the model try to resolve the task's issue. Takes minutes on a local model."""
    body = body or RunRequest()
    task = _task_or_404(session, task_id)
    if not workspaces.exists(task_id):
        raise HTTPException(404, "No workspace yet: POST /tasks/{id}/workspace first")
    if task_id in _running:
        raise HTTPException(409, "A run is already in progress for this task")
    if flow.exists(task_id):
        if not body.restart:
            raise HTTPException(
                409,
                "This task already has a run. POST /tasks/{id}/resume to continue an interrupted one, "
                "or run again with restart=true (the workspace keeps its edits).",
            )
        flow.reset(task_id)
    _running.add(task_id)
    try:
        indexes.build(task_id)  # always start from an index that matches the files
        result = flow.start(task, body.max_steps, body.max_repairs)
        memory.MemoryStore(session).record_history(task, result)
        return result
    finally:
        _running.discard(task_id)


@app.get("/tasks/{task_id}/run", response_model=RunStatus)
def run_status(task_id: str, session: SessionDep, flow: WorkflowDep) -> RunStatus:
    """Progress of the run, read from its latest checkpoint (works while it is running)."""
    _task_or_404(session, task_id)
    return flow.status(task_id, running=task_id in _running)


@app.post("/tasks/{task_id}/resume", response_model=AgentRun)
def resume_run(
    task_id: str, session: SessionDep, workspaces: WorkspacesDep, indexes: IndexesDep, flow: WorkflowDep
) -> AgentRun:
    """Continue an interrupted run from its last checkpoint."""
    task = _task_or_404(session, task_id)
    if not flow.exists(task_id):
        raise HTTPException(404, "This task has no run to resume")
    if task_id in _running:
        raise HTTPException(409, "A run is already in progress for this task")
    if flow.finished(task_id):
        raise HTTPException(409, "The run already finished; nothing to resume")
    if not workspaces.exists(task_id):
        raise HTTPException(409, "The workspace is gone; start a new run with restart=true")
    _running.add(task_id)
    try:
        indexes.build(task_id)
        result = flow.resume(task_id)
        memory.MemoryStore(session).record_history(task, result)
        return result
    except WorkspaceChanged:
        raise HTTPException(
            409, "The workspace changed since the last checkpoint, so resuming is unsafe. Run again with restart=true."
        )
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


# --- long-term memory ---------------------------------------------------------------


@app.post("/memories", response_model=Memory, status_code=201)
def create_memory(body: MemoryCreate, session: SessionDep) -> Memory:
    try:
        return memory.MemoryStore(session).create(body.repo, body.kind, body.content)
    except memory.SecretDetected as exc:
        raise HTTPException(422, str(exc))
    except memory.TooManyMemories:
        raise HTTPException(409, "Too many memories for this repository: delete some first")


@app.get("/memories", response_model=list[Memory])
def list_memories(
    session: SessionDep, repo: Annotated[str, Query(pattern=r"^[A-Za-z0-9_.-]*[A-Za-z0-9_-][A-Za-z0-9_.-]*/[A-Za-z0-9_.-]*[A-Za-z0-9_-][A-Za-z0-9_.-]*$")]
) -> list[Memory]:
    """Memories that apply to this repository (including the "*" ones)."""
    return memory.MemoryStore(session).for_repo(repo)


@app.patch("/memories/{memory_id}", response_model=Memory)
def update_memory(memory_id: str, body: MemoryUpdate, session: SessionDep) -> Memory:
    try:
        return memory.MemoryStore(session).update(memory_id, body.content)
    except memory.MemoryNotFound:
        raise HTTPException(404, "Memory not found")
    except memory.SecretDetected as exc:
        raise HTTPException(422, str(exc))


@app.delete("/memories/{memory_id}", status_code=204)
def delete_memory(memory_id: str, session: SessionDep) -> Response:
    try:
        memory.MemoryStore(session).delete(memory_id)
    except memory.MemoryNotFound:
        raise HTTPException(404, "Memory not found")
    return Response(status_code=204)


# --- review and publish -----------------------------------------------------------------

AUTHOR = ("Autonoms", "autonoms@users.noreply.github.com")  # commits say plainly that a tool made them
_publishing: set[str] = set()


def _publication(session: Session, task_id: str) -> Publication | None:
    row = session.get(PublicationRow, task_id)
    return Publication.model_validate(row, from_attributes=True) if row else None


def _finished_run(flow: Workflow, task_id: str) -> AgentRun:
    if not flow.exists(task_id) or not flow.finished(task_id):
        raise HTTPException(409, "This task has no finished run to review")
    return flow.result(task_id)


@app.get("/tasks/{task_id}/review", response_model=Review)
def review_task(task_id: str, session: SessionDep, flow: WorkflowDep) -> Review:
    """The report a human reads before approving. Nothing is published by reading it."""
    task = _task_or_404(session, task_id)
    review = report.build_review(task, _finished_run(flow, task_id))
    review.publication = _publication(session, task_id)
    return review


@app.post("/tasks/{task_id}/publish", response_model=Publication, status_code=201)
def publish_task(
    task_id: str, body: PublishRequest, session: SessionDep, flow: WorkflowDep, workspaces: WorkspacesDep, github: GitHubDep
) -> Publication:
    """Push a branch and open a pull request, but only for exactly the content that was reviewed."""
    task = _task_or_404(session, task_id)
    if existing := _publication(session, task_id):
        raise HTTPException(409, f"Already published: {existing.pr_url}")
    run = _finished_run(flow, task_id)
    review = report.build_review(task, run)
    if review.blockers:
        raise HTTPException(409, "Cannot publish: " + " ".join(review.blockers))
    if body.fingerprint != review.fingerprint:
        raise HTTPException(409, "What you approved is not what would be published (the run or the report changed). Review again.")
    if review.needs_acknowledgement and not body.acknowledge_warnings:
        raise HTTPException(409, "This review has warnings. Read them in GET /review and send acknowledge_warnings=true to proceed.")
    if task_id in _publishing:
        raise HTTPException(409, "Publishing is already in progress for this task")
    _publishing.add(task_id)
    try:
        # The edits must still be exactly the ones that were reviewed (unless we already committed them).
        if workspaces.current_branch(task_id) != review.branch and workspaces.diff(task_id) != run.diff:
            raise HTTPException(409, "The workspace changed since the run, so it is no longer what was reviewed. Run again.")
        message = f"{review.title}\n\nGenerated by Autonoms. Published after approval by {report.clean(body.approved_by, 80)}."
        try:
            workspaces.commit_to_branch(task_id, review.branch, message, AUTHOR)
            base = workspaces.base_branch(task_id)
            workspaces.push_branch(task_id, review.branch)  # a plain push: never forced
            try:
                number, url = github.create_pull_request(
                    task.owner, task.repo, review.branch, base, review.title, report.final_body(review, body.approved_by), body.draft
                )
            except PullRequestExists:  # an earlier attempt got this far
                found = github.find_pull_request(task.owner, task.repo, review.branch)
                if found is None:
                    raise HTTPException(502, "GitHub says a pull request exists but it could not be found")
                number, url = found
        except workspace.GitError as exc:
            if exc.denied:
                raise HTTPException(
                    502,
                    "GitHub refused the push: the token can read this repository but not write to it. Give the token "
                    "'Contents: Read and write' and 'Pull requests: Read and write' for this repository, then publish again.",
                )
            raise HTTPException(502, "git failed while publishing (is the branch already on GitHub with different history?)")
        except GitHubError as exc:
            raise HTTPException(502, f"The branch was pushed, but opening the pull request failed: {exc}. Publishing again is safe.")
        row = PublicationRow(
            task_id=task_id, branch=review.branch, base=base, pr_number=number, pr_url=url,
            approved_by=body.approved_by, fingerprint=review.fingerprint, draft=body.draft, created_at=datetime.now(UTC),
        )  # fmt: skip
        session.add(row)
        session.commit()
        return Publication.model_validate(row, from_attributes=True)
    finally:
        _publishing.discard(task_id)


@app.get("/tasks/{task_id}/evaluation", response_model=Evaluation)
def evaluate_task(task_id: str, session: SessionDep, flow: WorkflowDep) -> Evaluation:
    """Did the run complete, stay clean and in scope, and what did it cost? Computed locally."""
    task = _task_or_404(session, task_id)
    return evaluate(task, _finished_run(flow, task_id))
