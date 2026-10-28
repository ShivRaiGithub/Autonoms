"""Storing and loading tasks. Pure database code: knows nothing about HTTP."""

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from autonoms.db import TaskRow
from autonoms.models import Issue, Task, TaskStatus


def _to_task(row: TaskRow) -> Task:
    return Task(
        id=row.id,
        owner=row.owner,
        repo=row.repo,
        issue_number=row.issue_number,
        issue_title=row.issue_title,
        status=TaskStatus(row.status),
        created_at=row.created_at,
        updated_at=row.updated_at,
        issue=Issue.model_validate(row.issue),
    )


def create_task(session: Session, issue: Issue) -> Task:
    now = datetime.now(UTC)
    row = TaskRow(
        id=str(uuid.uuid4()),
        owner=issue.owner,
        repo=issue.repo,
        issue_number=issue.number,
        issue_title=issue.title,
        status=TaskStatus.CREATED,
        issue=issue.model_dump(mode="json"),  # mode="json": dates become strings
        created_at=now,
        updated_at=now,
    )
    session.add(row)
    session.commit()
    return _to_task(row)


def list_tasks(session: Session) -> list[Task]:
    rows = session.scalars(select(TaskRow).order_by(TaskRow.created_at.desc()))
    return [_to_task(row) for row in rows]


def get_task(session: Session, task_id: str) -> Task | None:
    row = session.get(TaskRow, task_id)
    return _to_task(row) if row else None


def delete_task(session: Session, task_id: str) -> bool:
    row = session.get(TaskRow, task_id)
    if row is None:
        return False
    session.delete(row)
    session.commit()
    return True
