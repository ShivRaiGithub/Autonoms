"""Long-term memory: what Autonoms remembers between tasks.

Design rules (from the project spec):
  - scoped: a memory belongs to one repository ("owner/name") or to "*" (everywhere)
  - small and structured: kind + at most 500 characters, not transcripts
  - never secrets: text that looks like a credential is refused, not stored
  - retrieved only when relevant: conventions/preferences always (few), the rest by word overlap
  - removable: every memory can be edited or deleted; history is capped and pruned
"""

import re
import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from autonoms.db import MemoryRow
from autonoms.models import AgentRun, Memory, Task

ALWAYS_KINDS = ("convention", "preference")  # short and general: always shown
MAX_ALWAYS = 5
MAX_RELEVANT = 3
MIN_SHARED_WORDS = 2
MAX_PER_REPO = 100  # manual memories
MAX_HISTORY_PER_REPO = 50
STOPWORDS = frozenset(
    "the and for are was with that this from have has not but you your all can will its into when then than "
    "does did should would could about they them there their what which who how why also only just "
    # words that appear in almost every issue or in our own history template say nothing about the topic
    "fix fixed wrong error issue bug add new change changed changes update make need needs "
    "finished failed passed check equations example answer correct incorrect".split()
)  # fmt: skip
SECRET_PATTERNS = [
    re.compile(p)
    for p in (
        r"ghp_[A-Za-z0-9]{16,}", r"github_pat_[A-Za-z0-9_]{16,}", r"gh[ousr]_[A-Za-z0-9]{16,}",
        r"sk-[A-Za-z0-9_-]{16,}", r"AKIA[0-9A-Z]{16}", r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        r"://[^/\s:@]+:[^@\s]+@",  # user:password@host in a URL
        r"(?i)\b(password|passwd|secret|token|api[_-]?key)\s*[=:]\s*\S+",
        r"\b[A-Za-z0-9+/_-]{40,}={0,2}\b",  # a long unbroken token-like string
    )
]  # fmt: skip


class SecretDetected(Exception):
    pass


class MemoryNotFound(Exception):
    pass


class TooManyMemories(Exception):
    pass


def looks_secret(text: str) -> bool:
    return any(p.search(text) for p in SECRET_PATTERNS)


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]{3,}", text.lower()) if w not in STOPWORDS}


def _to_memory(row: MemoryRow) -> Memory:
    return Memory(
        id=row.id, repo=row.repo, kind=row.kind, content=row.content, source_task_id=row.source_task_id,
        created_at=row.created_at, updated_at=row.updated_at, last_used_at=row.last_used_at, use_count=row.use_count,
    )  # fmt: skip


class MemoryStore:
    def __init__(self, session: Session):
        self._session = session

    def _repo_rows(self, repo: str, kinds: tuple[str, ...] | None = None) -> list[MemoryRow]:
        query = select(MemoryRow).where(MemoryRow.repo.in_([repo, "*"])).order_by(MemoryRow.updated_at.desc())
        rows = list(self._session.scalars(query))
        return [r for r in rows if kinds is None or r.kind in kinds]

    # --- create / read / update / delete ---------------------------------------------

    def create(self, repo: str, kind: str, content: str, source_task_id: str | None = None) -> Memory:
        if looks_secret(content):
            raise SecretDetected("That text looks like a credential, so it was not stored.")
        if kind != "history":
            manual = [r for r in self._repo_rows(repo) if r.repo == repo and r.kind != "history"]
            if len(manual) >= MAX_PER_REPO:
                raise TooManyMemories(repo)
        now = datetime.now(UTC)
        row = MemoryRow(
            id=str(uuid.uuid4()), repo=repo, kind=kind, content=content, source_task_id=source_task_id,
            created_at=now, updated_at=now, last_used_at=None, use_count=0,
        )  # fmt: skip
        self._session.add(row)
        self._session.commit()
        return _to_memory(row)

    def for_repo(self, repo: str) -> list[Memory]:
        return [_to_memory(r) for r in self._repo_rows(repo)]

    def update(self, memory_id: str, content: str) -> Memory:
        row = self._session.get(MemoryRow, memory_id)
        if row is None:
            raise MemoryNotFound(memory_id)
        if looks_secret(content):
            raise SecretDetected("That text looks like a credential, so it was not stored.")
        row.content, row.updated_at = content, datetime.now(UTC)
        self._session.commit()
        return _to_memory(row)

    def delete(self, memory_id: str) -> None:
        row = self._session.get(MemoryRow, memory_id)
        if row is None:
            raise MemoryNotFound(memory_id)
        self._session.delete(row)
        self._session.commit()

    # --- using memories ----------------------------------------------------------------

    def relevant(self, repo: str, text: str) -> list[Memory]:
        """Memories worth showing for a task whose issue says `text`. Also marks them as used."""
        rows = self._repo_rows(repo)
        always = [r for r in rows if r.kind in ALWAYS_KINDS][:MAX_ALWAYS]
        issue_words = words(text)
        scored = sorted(
            ((len(words(r.content) & issue_words), r) for r in rows if r.kind not in ALWAYS_KINDS),
            key=lambda pair: pair[0], reverse=True,
        )  # fmt: skip
        chosen = always + [r for score, r in scored if score >= MIN_SHARED_WORDS][:MAX_RELEVANT]
        now = datetime.now(UTC)
        for row in chosen:
            row.last_used_at, row.use_count = now, row.use_count + 1
        self._session.commit()
        return [_to_memory(r) for r in chosen]

    def record_history(self, task: Task, run: AgentRun) -> Memory | None:
        """Remember what a finished run changed. Facts come from the diff and the checks, not from the model.

        One history memory per task: a resumed or repeated run replaces it instead of adding another.
        """
        files = list(dict.fromkeys(re.findall(r"^diff --git a/(\S+)", run.diff, flags=re.M)))
        if not run.status.startswith("finished") or not files:
            return None
        content = (
            f"Issue #{task.issue_number} \"{task.issue_title[:80]}\" ({run.status}): changed {', '.join(files)}. "
            f"Wrong equations {len(run.check_before.problems)} -> {len(run.check_after.problems)} "
            f"({run.check_after.status})."
        )  # fmt: skip
        if looks_secret(content):  # an issue title is written by someone else: never trust it
            return None
        repo = f"{task.owner}/{task.repo}"
        existing = self._session.scalars(
            select(MemoryRow).where(MemoryRow.kind == "history", MemoryRow.source_task_id == task.id)
        ).first()
        if existing:
            existing.content, existing.updated_at = content, datetime.now(UTC)
            self._session.commit()
            return _to_memory(existing)
        memory = self.create(repo, "history", content, source_task_id=task.id)
        old = [r for r in self._repo_rows(repo, ("history",)) if r.repo == repo][MAX_HISTORY_PER_REPO:]
        for row in old:  # retention: forget the oldest history beyond the cap
            self._session.delete(row)
        self._session.commit()
        return memory
