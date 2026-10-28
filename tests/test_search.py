import uuid

import httpx
import pytest

from autonoms.main import app, get_indexes, get_workspaces
from autonoms.search import MODES, NoIndex, SearchIndexes
from autonoms.workspace import NoWorkspace, Workspaces
from conftest import make_issue

TASK = str(uuid.uuid4())
MODEL = "BAAI/bge-small-en-v1.5"


@pytest.fixture
def notes(tmp_path, notes_server):
    workspaces = Workspaces(tmp_path / "ws", notes_server)
    workspaces.create(TASK, "o", "r")
    return SearchIndexes(tmp_path / "idx", workspaces, MODEL)


@pytest.fixture
def built(notes):
    notes.build(TASK)
    return notes


def top(result):
    return result.hits[0].path if result.hits else None


def test_build_counts_text_files_only(notes):
    info = notes.build(TASK)
    assert info.files == 4  # image.bin is skipped
    assert info.chunks == 4  # each note is shorter than one chunk
    assert info.embedding_model == MODEL


def test_build_without_workspace_fails(tmp_path, notes_server):
    indexes = SearchIndexes(tmp_path / "idx", Workspaces(tmp_path / "ws", notes_server), MODEL)
    with pytest.raises(NoWorkspace):
        indexes.build(TASK)


def test_search_before_build_fails(notes):
    with pytest.raises(NoIndex):
        notes.search(TASK, "anything")


def test_keyword_finds_exact_words(built):
    assert top(built.search(TASK, "hypotenuse", "keyword")) == "pythagoras.md"


def test_keyword_returns_nothing_when_no_word_matches(built):
    assert built.search(TASK, "xyzzy", "keyword").hits == []


def test_semantic_matches_meaning_without_shared_words(built):
    # No word in these queries appears in the note they should find. They need a
    # clear concept: vague ones ("which calculation first?") score 0.57-0.65 for
    # every note and the small model cannot separate them (see the stage doc).
    assert top(built.search(TASK, "space inside a round shape", "semantic")) == "formulas.md"
    assert top(built.search(TASK, "how to add 1/2 and 1/3", "semantic")) == "fractions.md"


def test_keyword_search_misses_what_semantic_finds(built):
    query = "space inside a round shape"
    assert built.search(TASK, query, "keyword").hits == []
    assert top(built.search(TASK, query, "semantic")) == "formulas.md"


def test_hybrid_finds_both_kinds_of_query(built):
    assert top(built.search(TASK, "hypotenuse", "hybrid")) == "pythagoras.md"
    assert top(built.search(TASK, "space inside a round shape", "hybrid")) == "formulas.md"


def test_top_k_larger_than_corpus_is_fine(built):
    for mode in MODES:
        assert len(built.search(TASK, "squares", mode, top_k=20).hits) <= 4


def test_unknown_mode(built):
    with pytest.raises(ValueError):
        built.search(TASK, "x", "magic")


def test_index_survives_restart(built, tmp_path, notes_server):
    fresh = SearchIndexes(tmp_path / "idx", Workspaces(tmp_path / "ws", notes_server), MODEL)
    assert top(fresh.search(TASK, "hypotenuse", "hybrid")) == "pythagoras.md"


def test_rebuild_replaces_and_remove_deletes(built):
    built.build(TASK)
    assert built.remove(TASK) is True
    assert built.remove(TASK) is False
    with pytest.raises(NoIndex):
        built.search(TASK, "x")


def test_task_id_cannot_escape_the_index_folder(notes):
    with pytest.raises(ValueError):
        notes.remove("../../tmp")


# --- through the HTTP API ----------------------------------------------------


def issue_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/comments"):
        return httpx.Response(200, json=[])
    return httpx.Response(
        200,
        json=make_issue(
            number=1,
            title="Pythagoras rule is wrong",
            body="The rule for the hypotenuse is missing squares.",
        ),
    )


@pytest.fixture
def api(fake_github, db_session, notes):
    app.dependency_overrides[get_workspaces] = lambda: notes._workspaces
    app.dependency_overrides[get_indexes] = lambda: notes
    client = fake_github(issue_handler)
    yield client
    app.dependency_overrides.pop(get_workspaces, None)
    app.dependency_overrides.pop(get_indexes, None)


def make_task(api, notes):
    task_id = api.post("/tasks", json={"owner": "o", "repo": "r", "number": 1}).json()["id"]
    notes._workspaces.create(task_id, "o", "r")
    return task_id


def test_api_index_search_and_context(api, notes):
    task_id = make_task(api, notes)
    assert api.get(f"/tasks/{task_id}/search", params={"q": "x"}).status_code == 404  # no index yet
    assert api.post(f"/tasks/{task_id}/index").json()["files"] == 4
    hits = api.get(f"/tasks/{task_id}/search", params={"q": "hypotenuse", "mode": "keyword"}).json()["hits"]
    assert hits[0]["path"] == "pythagoras.md"
    context = api.get(f"/tasks/{task_id}/context").json()
    assert context["hits"][0]["path"] == "pythagoras.md"  # found from the issue text alone


def test_api_validation_and_missing(api, notes):
    task_id = make_task(api, notes)
    api.post(f"/tasks/{task_id}/index")
    assert api.get(f"/tasks/{task_id}/search", params={"q": ""}).status_code == 422
    assert api.get(f"/tasks/{task_id}/search", params={"q": "x", "mode": "magic"}).status_code == 422
    assert api.get(f"/tasks/{task_id}/search", params={"q": "x", "top_k": 99}).status_code == 422
    assert api.post(f"/tasks/{uuid.uuid4()}/index").status_code == 404


def test_api_index_needs_a_workspace(api):
    task_id = api.post("/tasks", json={"owner": "o", "repo": "r", "number": 1}).json()["id"]
    assert api.post(f"/tasks/{task_id}/index").status_code == 404


def test_deleting_a_task_removes_its_index(api, notes):
    task_id = make_task(api, notes)
    api.post(f"/tasks/{task_id}/index")
    assert api.delete(f"/tasks/{task_id}").status_code == 204
    with pytest.raises(NoIndex):
        notes.search(task_id, "x")
