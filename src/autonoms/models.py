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
