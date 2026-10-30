"""Pydantic models: the shape of the data Autonoms works with."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


# GitHub names only allow these characters. Validating also blocks path tricks
# like "..", because owner/repo go straight into a URL path.
# At least one non-dot character: "." and ".." are path steps, not names. (No lookahead:
# pydantic's regex engine does not support it.)
NAME = r"^[A-Za-z0-9_.-]*[A-Za-z0-9_-][A-Za-z0-9_.-]*$"


class Comment(BaseModel):
    author: str
    body: str
    created_at: datetime


class Issue(BaseModel):
    owner: str
    repo: str
    number: int
    title: str
    body: str
    state: str
    author: str
    labels: list[str]
    url: str
    created_at: datetime
    comments: list[Comment]


class TaskStatus(StrEnum):
    CREATED = "created"


class TaskCreate(BaseModel):
    owner: str = Field(pattern=NAME, max_length=100)
    repo: str = Field(pattern=NAME, max_length=100)
    number: int = Field(ge=1)


class Task(BaseModel):
    """One piece of work: 'try to resolve this issue'. Later stages add more statuses."""

    id: str
    owner: str
    repo: str
    issue_number: int
    issue_title: str
    status: TaskStatus
    created_at: datetime
    updated_at: datetime
    issue: Issue


class RepoFile(BaseModel):
    path: str
    size: int


class Commit(BaseModel):
    sha: str
    author: str
    date: datetime
    subject: str


class RepoSummary(BaseModel):
    branch: str
    head: str
    file_count: int
    files: list[RepoFile]  # capped, see files_truncated
    files_truncated: bool
    manifests: list[str]  # dependency files found, e.g. pyproject.toml
    commits: list[Commit]


class IndexInfo(BaseModel):
    files: int
    chunks: int
    embedding_model: str


class SearchHit(BaseModel):
    path: str
    score: float
    text: str


class SearchResult(BaseModel):
    query: str
    mode: str
    hits: list[SearchHit]


class RunRequest(BaseModel):
    max_steps: int = Field(default=8, ge=1, le=15)  # model calls
    max_repairs: int = Field(default=1, ge=0, le=3)  # extra rounds after a failed check
    restart: bool = False  # discard a finished/interrupted run of this task and start again


class CheckProblem(BaseModel):
    file: str
    line: int
    text: str
    claimed: str
    actual: str


class CheckResult(BaseModel):
    # passed: ran, nothing wrong. failed: ran, found wrong equations.
    # not_run: it did not run to completion (Docker down, timeout, crash).
    # "not_run" is never the same as "passed".
    status: str
    equations: int = 0
    problems: list[CheckProblem] = []
    detail: str = ""  # why it did not run, if it did not
    seconds: float = 0


class AgentStep(BaseModel):
    tool: str
    arguments: dict
    result: str  # what the model was told (already shortened)


class Usage(BaseModel):
    model_calls: int = 0
    prompt_tokens: int = 0  # total tokens the model read, summed over calls
    completion_tokens: int = 0
    peak_prompt_tokens: int = 0  # the largest single prompt: compare with the 8192 window
    model_seconds: float = 0  # time spent waiting for the model, summed over calls


class AgentRun(BaseModel):
    # finished: the model called `finish`. The others are why we stopped it.
    status: str  # finished | no_tool_call | max_steps | context_limit | llm_error
    summary: str
    steps: list[AgentStep]
    diff: str  # `git diff` of the workspace: the proposed patch
    check_before: CheckResult  # the check on the untouched clone
    check_after: CheckResult  # the check on the result
    usage: Usage = Usage()
    compactions: int = 0  # calls where older steps were collapsed to fit the context
    memories_used: list[str] = []  # ids of long-term memories shown to the model
    models_used: list[str] = []  # which model(s) actually answered; more than one means a fallback happened
    seconds: float


class RunStatus(BaseModel):
    # not_started | running | interrupted (can be resumed) | finished
    state: str
    status: str = ""  # the run's own status once it has one
    next: list[str] = []  # graph nodes still to run
    model_calls: int = 0
    repairs: int = 0
    steps: list[AgentStep] = []


MEMORY_KINDS = ("convention", "preference", "pitfall", "finding")


class MemoryCreate(BaseModel):
    repo: str = Field(pattern=r"^(\*|[A-Za-z0-9_.-]*[A-Za-z0-9_-][A-Za-z0-9_.-]*/[A-Za-z0-9_.-]*[A-Za-z0-9_-][A-Za-z0-9_.-]*)$", max_length=201)  # "owner/name", or "*" for everywhere
    kind: str = Field(pattern=f"^({'|'.join(MEMORY_KINDS)})$")
    content: str = Field(min_length=1, max_length=500)


class MemoryUpdate(BaseModel):
    content: str = Field(min_length=1, max_length=500)


class Memory(BaseModel):
    id: str
    repo: str
    kind: str  # the four above, or "history" (written by Autonoms after a run)
    content: str
    source_task_id: str | None
    created_at: datetime
    updated_at: datetime
    last_used_at: datetime | None
    use_count: int


class Risk(BaseModel):
    level: str  # "info" = good to know; "warn" = must be acknowledged before publishing
    text: str


class Publication(BaseModel):
    task_id: str
    branch: str
    base: str
    pr_number: int
    pr_url: str
    approved_by: str
    fingerprint: str
    draft: bool
    created_at: datetime


class Review(BaseModel):
    """What a human reads before approving. `fingerprint` identifies exactly this content."""

    title: str
    body: str
    diff: str  # the exact change; covered by the fingerprint, so what you read is what you approve
    branch: str
    files: list[str]
    risks: list[Risk]
    blockers: list[str]  # reasons publishing is not possible at all
    needs_acknowledgement: bool  # True if any risk is a warning
    fingerprint: str
    can_publish: bool
    publication: Publication | None = None


class PublishRequest(BaseModel):
    fingerprint: str = Field(min_length=64, max_length=64)  # copied from GET /review: "I read exactly this"
    approved_by: str = Field(min_length=1, max_length=80)
    acknowledge_warnings: bool = False
    draft: bool = True
