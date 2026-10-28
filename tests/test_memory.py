import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from autonoms.db import Base
from autonoms.memory import MAX_HISTORY_PER_REPO, MAX_PER_REPO, MemoryNotFound, MemoryStore, SecretDetected, TooManyMemories, looks_secret
from autonoms.models import AgentRun, CheckProblem, CheckResult, Task
from conftest import make_issue

REPO = "Ssr4803/TestingPythonRepo"


@pytest.fixture
def store():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield MemoryStore(session)


def task(number=5, title="Fix circle area", task_id="11111111-1111-4111-8111-111111111111"):
    issue = {"owner": "Ssr4803", "repo": "TestingPythonRepo", "number": number, "title": title, "body": "b", "state": "open",
             "author": "a", "labels": [], "url": "u", "created_at": "2026-01-01T00:00:00Z", "comments": []}  # fmt: skip
    return Task(id=task_id, owner="Ssr4803", repo="TestingPythonRepo", issue_number=number, issue_title=title, status="created",
                created_at="2026-01-01T00:00:00Z", updated_at="2026-01-01T00:00:00Z", issue=issue)  # fmt: skip


def run(status="finished", files=("formulas.md",), before=3, after=3):
    problems = lambda n: [CheckProblem(file="a.md", line=1, text="x", claimed="1", actual="2")] * n  # noqa: E731
    return AgentRun(
        status=status, summary="s", steps=[], seconds=1,
        diff="".join(f"diff --git a/{f} b/{f}\n--- a/{f}\n" for f in files),
        check_before=CheckResult(status="failed", equations=11, problems=problems(before)),
        check_after=CheckResult(status="failed", equations=11, problems=problems(after)),
    )  # fmt: skip


# --- safety ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "use ghp_abcdefghijklmnopqrstuvwxyz0123456789",
        "github_pat_11ABCDEFG0abcdefghijklmnop",
        "token = abc123",
        "PASSWORD: hunter2",
        "postgresql://user:pw123@localhost:5432/db",
        "sk-abcdefghijklmnopqrstuvwxyz",
        "AKIAABCDEFGHIJKLMNOP",
        "-----BEGIN RSA PRIVATE KEY-----",
        "x" * 20 + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9t0",
    ],
)
def test_credentials_are_refused(store, text):
    assert looks_secret(text)
    with pytest.raises(SecretDetected):
        store.create(REPO, "pitfall", text)
    assert store.for_repo(REPO) == []


@pytest.mark.parametrize("text", ["Examples use x for multiplication, not the dot.", "Keep answers as whole numbers.", "bodmas.md and fractions.md share examples"])
def test_ordinary_notes_are_accepted(store, text):
    assert not looks_secret(text)
    assert store.create(REPO, "convention", text).content == text


def test_updating_to_a_credential_is_refused_too(store):
    memory = store.create(REPO, "convention", "fine")
    with pytest.raises(SecretDetected):
        store.update(memory.id, "token = abc123")
    assert store.for_repo(REPO)[0].content == "fine"


def test_a_secret_in_an_issue_title_stops_the_history_memory(store):
    assert store.record_history(task(title="my key is ghp_abcdefghijklmnopqrstuvwxyz0123456789"), run()) is None
    assert store.for_repo(REPO) == []


# --- scope, edit, delete, limits ---------------------------------------------------------


def test_scope_repo_memories_stay_in_their_repo_and_star_applies_everywhere(store):
    store.create(REPO, "convention", "mine")
    store.create("other/repo", "convention", "theirs")
    store.create("*", "preference", "everywhere")
    assert {m.content for m in store.for_repo(REPO)} == {"mine", "everywhere"}
    assert {m.content for m in store.for_repo("other/repo")} == {"theirs", "everywhere"}


def test_update_and_delete(store):
    memory = store.create(REPO, "pitfall", "old text")
    assert store.update(memory.id, "new text").content == "new text"
    store.delete(memory.id)
    assert store.for_repo(REPO) == []
    for action in (lambda: store.update(memory.id, "x"), lambda: store.delete(memory.id)):
        with pytest.raises(MemoryNotFound):
            action()


def test_manual_memories_are_capped_per_repo(store):
    for i in range(MAX_PER_REPO):
        store.create(REPO, "finding", f"note {i}")
    with pytest.raises(TooManyMemories):
        store.create(REPO, "finding", "one too many")
    store.create("other/repo", "finding", "other repos have their own budget")


# --- retrieval ---------------------------------------------------------------------------


def test_conventions_and_preferences_are_always_included(store):
    store.create(REPO, "convention", "Use x for multiplication")
    store.create("*", "preference", "Keep summaries short")
    got = {m.content for m in store.relevant(REPO, "something completely unrelated about weather")}
    assert got == {"Use x for multiplication", "Keep summaries short"}


def test_other_kinds_are_included_only_when_the_issue_shares_words(store):
    store.create(REPO, "pitfall", "formulas.md squares are written as radius x radius, not radius x 2")
    store.create(REPO, "finding", "pythagoras.md rule needs squares on every side")
    got = store.relevant(REPO, "Fix circle radius formula in formulas.md")
    assert [m.kind for m in got] == ["pitfall"]
    assert store.relevant(REPO, "hello world") == []


def test_one_shared_word_is_not_enough(store):
    store.create(REPO, "finding", "pythagoras.md rule needs squares")
    assert store.relevant(REPO, "add a new squares section") == []  # only "squares" in common


def test_using_a_memory_is_recorded(store):
    memory = store.create(REPO, "convention", "Use x")
    store.relevant(REPO, "anything")
    store.relevant(REPO, "anything")
    used = store.for_repo(REPO)[0]
    assert used.id == memory.id and used.use_count == 2 and used.last_used_at is not None


# --- history -----------------------------------------------------------------------------


def test_history_is_built_from_the_diff_and_checks_not_from_the_model(store):
    memory = store.record_history(task(), run(files=("formulas.md", "bodmas.md"), before=3, after=2))
    assert memory.kind == "history" and memory.source_task_id == task().id
    assert memory.content == 'Issue #5 "Fix circle area" (finished): changed formulas.md, bodmas.md. Wrong equations 3 -> 2 (failed).'


@pytest.mark.parametrize("bad_run", [run(status="max_steps"), run(status="llm_error"), run(files=())])
def test_failed_or_empty_runs_leave_no_history(store, bad_run):
    assert store.record_history(task(), bad_run) is None


def test_the_same_task_never_creates_two_history_entries(store):
    store.record_history(task(), run(after=3))
    store.record_history(task(), run(after=1))  # e.g. after a resume
    entries = store.for_repo(REPO)
    assert len(entries) == 1 and "3 -> 1" in entries[0].content


def test_old_history_is_forgotten_beyond_the_cap(store):
    for i in range(MAX_HISTORY_PER_REPO + 5):
        store.record_history(task(number=i + 1, task_id=f"00000000-0000-4000-8000-{i:012d}"), run())
    kept = store.for_repo(REPO)
    assert len(kept) == MAX_HISTORY_PER_REPO
    assert not any('Issue #1 "' in m.content for m in kept)  # the oldest went first


def test_boilerplate_words_do_not_make_unrelated_history_relevant(store):
    """Found in the real run: "Fix ... Wrong equations" matched any issue saying "wrong" and "fix"."""
    store.record_history(task(number=5, title="Fix circle area and add a triangle to formulas.md"), run(files=("formulas.md",)))
    unrelated = "Example 1 in bodmas.md has the wrong final answer. Fix the example."
    assert store.relevant(REPO, unrelated) == []
    related = "The circle area formula in formulas.md uses the wrong radius"
    assert len(store.relevant(REPO, related)) == 1
