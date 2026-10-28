"""Pydantic models: the shape of the data Autonoms works with."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


# GitHub names only allow these characters. Validating also blocks path tricks
# like "..", because owner/repo go straight into a URL path.
NAME = r"^[A-Za-z0-9_.-]+$"


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
    max_steps: int = Field(default=8, ge=1, le=15)


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


class AgentRun(BaseModel):
    # finished: the model called `finish`. The others are why we stopped it.
    status: str  # finished | no_tool_call | max_steps | context_limit | llm_error
    summary: str
    steps: list[AgentStep]
    diff: str  # `git diff` of the workspace: the proposed patch
    check_before: CheckResult  # the check on the untouched clone
    check_after: CheckResult  # the check on the result
    seconds: float
