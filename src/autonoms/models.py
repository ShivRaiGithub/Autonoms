"""Pydantic models: the shape of the data Autonoms works with."""

from datetime import datetime

from pydantic import BaseModel


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
