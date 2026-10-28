"""The web page, exercised by simulated users against the real API (in-process, with fakes behind it)."""

import asyncio

import httpx
import pytest
from nicegui import ui
from nicegui.testing import User

from autonoms.main import app
from autonoms.ui.api import Api
from autonoms.ui.app import register
from test_agent import FAILED, FakeChecker, call, env  # noqa: F401 (fixtures)
from test_publish import PASSED, papi  # noqa: F401 (fixtures)


@pytest.fixture
def api_for_ui(papi):  # noqa: F811
    """An API client wired straight into the FastAPI app, with the scripted fakes from test_publish behind it."""
    # raise_app_exceptions=bool(__import__("os").environ.get("UI_TEST_RAISE")): a crash inside the API becomes an HTTP 500 response, as it would over a real network
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=bool(__import__("os").environ.get("UI_TEST_RAISE"))), base_url="http://test", timeout=None)
    api = Api("http://test", client)
    papi.ui_api = api
    return papi


async def test_the_page_loads_and_shows_an_empty_state(user: User, api_for_ui):
    register(api_for_ui.ui_api)
    await user.open("/")
    await see(user, "Autonoms")
    await see(user, "API connected")
    await see(user, "No tasks yet.")
    await see(user, "Create a task, or select one on the left.")


EDIT = [call("replace_in_file", path="formulas.md", old_text="radius squared", new_text="r squared"), call("finish", summary="Reworded the formula.")]


async def see(user: User, text: str) -> None:
    """should_see with patience: the first run in a test loads the embedding model and builds an index."""
    await user.should_see(text, retries=150)


async def settle(user: User) -> None:
    """Wait until the page is idle: no spinner, and the redraw that follows an action has happened.

    A simulated user is far faster than a person: it can find a button and click it in the few
    milliseconds before the page replaces that button with a fresh one, and the click lands on
    nothing. A person cannot, so the tests wait where a person naturally would.
    """
    for _ in range(200):
        if not any(isinstance(e, ui.spinner) for e in user.client.elements.values()):
            break
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.4)


async def open_page(user: User, api_for_ui) -> None:
    register(api_for_ui.ui_api)
    await user.open("/")


async def create_task(user: User) -> None:
    user.find("Owner").clear().type("o")
    user.find("Repository").clear().type("r")
    user.find("Issue number").clear().type("1")
    user.find("Create task").click()
    await see(user, "Circle area is wrong")
    await settle(user)


async def test_creating_a_task_shows_the_issue_and_lists_it(user: User, api_for_ui):
    await open_page(user, api_for_ui)
    await create_task(user)
    await see(user, "formulas.md says radius squared")  # the issue text
    await see(user, "o/r")
    await see(user, "1. Get the repository")
    await user.should_not_see("No tasks yet.")


async def test_a_bad_repository_name_shows_the_servers_message_instead_of_crashing(user: User, api_for_ui):
    await open_page(user, api_for_ui)
    user.find("Owner").clear().type("../evil")
    user.find("Create task").click()
    await see(user, "owner")  # the validation message names the field
    await see(user, "Create a task, or select one on the left.")


async def wait_for_notification(user: User, text: str) -> None:
    for _ in range(100):
        if any(text in message for message in user.notify.messages):
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"no notification containing {text!r}: {list(user.notify.messages)}")


async def clone_and_run(user: User, api_for_ui, replies=EDIT, expect="The model finished.") -> None:
    user.find("1. Get the repository").click()
    await wait_for_notification(user, "Repository ready")
    await settle(user)
    api_for_ui.llm.replies = list(replies)
    user.find("2. Run the agent").click()
    await see(user, expect)
    await settle(user)


async def test_run_then_read_the_result(user: User, api_for_ui):
    await open_page(user, api_for_ui)
    await create_task(user)
    await clone_and_run(user, api_for_ui)
    await see(user, "completed")
    await see(user, "in scope")
    await see(user, "-The area of a circle is pi times the radius squared.")
    await see(user, "+The area of a circle is pi times the r squared.")


async def test_the_whole_journey_ends_in_a_draft_pull_request_only_after_confirmation(user: User, api_for_ui):
    await open_page(user, api_for_ui)
    await create_task(user)
    await clone_and_run(user, api_for_ui)
    user.find("Review & publish").click()
    await settle(user)
    await see(user, "Fix #1: Circle area is wrong")
    await see(user, "Your approval")
    user.find("Your name").type("Alice")
    user.find("Approve and publish as draft PR").click()
    await see(user, "Publish this as a draft pull request?")
    assert api_for_ui.state["pulls"] == []  # asking is not publishing
    user.find(marker="publish-confirm").click()
    await see(user, "Published as draft pull request #7")
    pull = api_for_ui.state["pulls"][0]
    assert pull["draft"] is True and "approval by **Alice**" in pull["body"] and pull["title"] == "Fix #1: Circle area is wrong"


async def test_cancelling_the_confirmation_publishes_nothing(user: User, api_for_ui):
    await open_page(user, api_for_ui)
    await create_task(user)
    await clone_and_run(user, api_for_ui)
    user.find("Review & publish").click()
    await settle(user)
    user.find("Your name").type("Alice")
    user.find("Approve and publish as draft PR").click()
    await see(user, "Publish this as a draft pull request?")
    user.find(marker="publish-cancel").click()
    await user.should_not_see("Publish this as a draft pull request?")
    assert api_for_ui.state["pulls"] == []
    await user.should_not_see("Published as")


async def test_publishing_needs_the_approvers_name(user: User, api_for_ui):
    await open_page(user, api_for_ui)
    await create_task(user)
    await clone_and_run(user, api_for_ui)
    user.find("Review & publish").click()
    await settle(user)
    user.find("Approve and publish as draft PR").click()
    await user.should_not_see("Publish this as a draft pull request?")
    user.notify.contains("type your name")
    assert api_for_ui.state["pulls"] == []


async def test_warnings_must_be_ticked_before_publishing(user: User, api_for_ui):
    api_for_ui.state["checker"] = FakeChecker(PASSED, FAILED)  # the edit "breaks" an equation
    await open_page(user, api_for_ui)
    await create_task(user)
    await clone_and_run(user, api_for_ui, replies=[*EDIT, call("finish", summary="Tried again.")], expect="Finished, but the edits broke something")
    user.find("Review & publish").click()
    await settle(user)
    await see(user, "The edits made 1 equation(s) wrong that were correct before")
    user.find("Your name").type("Alice")
    user.find("Approve and publish as draft PR").click()
    await user.should_not_see("Publish this as a draft pull request?")
    user.notify.contains("warnings")
    user.find("I have read the warnings above and still want to publish").click()
    user.find("Approve and publish as draft PR").click()
    await see(user, "Publish this as a draft pull request?")
    user.find(marker="publish-confirm").click()
    await see(user, "Published as draft pull request #7")
    assert "**Warning:**" in api_for_ui.state["pulls"][0]["body"]


async def test_a_refused_publish_shows_what_to_fix(user: User, api_for_ui):
    import httpx as _httpx

    api_for_ui.state["pr_answers"] = [_httpx.Response(403, json={"message": "Resource not accessible"})]
    await open_page(user, api_for_ui)
    await create_task(user)
    await clone_and_run(user, api_for_ui)
    user.find("Review & publish").click()
    await settle(user)
    user.find("Your name").type("Alice")
    user.find("Approve and publish as draft PR").click()
    await see(user, "Publish this as a draft pull request?")
    user.find(marker="publish-confirm").click()
    await see(user, "Pull requests: write")
    assert any("Pull requests: write" in message for message in user.notify.messages)  # visible wherever you are scrolled to
    await user.should_not_see("Published as")


async def test_running_before_getting_the_repository_explains_what_to_do(user: User, api_for_ui):
    await open_page(user, api_for_ui)
    await create_task(user)
    user.find("2. Run the agent").click()
    await see(user, "No workspace yet")


def closed_port() -> int:
    """A local port nothing listens on (connecting is refused at once)."""
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def test_an_unreachable_api_is_shown_not_hidden(user: User):
    register(Api(f"http://127.0.0.1:{closed_port()}"))
    await user.open("/")
    await see(user, "API unreachable")


async def test_existing_tasks_are_listed_and_can_be_selected(user: User, api_for_ui):
    await api_for_ui.ui_api.create_task("o", "r", 1)
    await open_page(user, api_for_ui)
    await see(user, "#1 Circle area is wrong")
    user.find(kind=ui.item).click()  # a real browser bubbles a click on the label up to the item; the simulation does not
    await see(user, "1. Get the repository")


async def test_an_interrupted_run_can_be_resumed_from_the_page(user: User, api_for_ui):
    await open_page(user, api_for_ui)
    await create_task(user)
    user.find("1. Get the repository").click()
    await wait_for_notification(user, "Repository ready")
    await settle(user)
    api_for_ui.llm.replies = [call("read_file", path="bodmas.md"), RuntimeError("power cut")]
    user.find("2. Run the agent").click()
    await see(user, "interrupted: can be resumed")
    await see(user, "Resume the interrupted run")
    api_for_ui.llm.replies = [call("finish", summary="Recovered.")]
    user.find("Resume the interrupted run").click()
    await see(user, "The model finished.")


async def test_memories_can_be_added_and_deleted_and_secrets_are_refused(user: User, api_for_ui):
    await open_page(user, api_for_ui)
    await create_task(user)
    user.find("Memories").click()
    await see(user, "Nothing yet.")
    user.find("A short note to remember (no secrets)").type("Use x for multiplication")
    user.find("Remember").click()
    for _ in range(100):  # the text is already visible in the input box, so wait for it to be *stored*
        if await api_for_ui.ui_api.memories("o/r"):
            break
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.3)  # let the page redraw its list
    user.find("A short note to remember (no secrets)").clear().type("token = abc123")
    user.find("Remember").click()
    await see(user, "looks like a credential")
    stored = [m["content"] for m in await api_for_ui.ui_api.memories("o/r")]
    assert stored == ["Use x for multiplication"]  # the credential-like text was never stored


async def test_deleting_a_task_asks_first(user: User, api_for_ui):
    await open_page(user, api_for_ui)
    await create_task(user)
    user.find("Delete task").click()
    await see(user, "Delete this task?")
    user.find(marker="delete-cancel").click()
    await user.should_not_see("Delete this task?")
    await see(user, "Circle area is wrong")  # still there
    assert len(await api_for_ui.ui_api.tasks()) == 1
    user.find("Delete task").click()
    await see(user, "Delete this task?")
    user.find(marker="delete-confirm").click()
    await see(user, "No tasks yet.")
    await user.should_not_see("Circle area is wrong")
    assert await api_for_ui.ui_api.tasks() == []


async def test_a_tab_is_selected_from_the_start(user: User, api_for_ui):
    """Seen in a real browser: no tab was selected and the panel below was empty until the first click."""
    await open_page(user, api_for_ui)
    await create_task(user)
    tabs = next(iter(user.find(kind=ui.tabs).elements))
    panels = next(iter(user.find(kind=ui.tab_panels).elements))
    assert tabs.value == "Result" and panels.value == "Result"
    user.find("Memories").click()
    await see(user, "Nothing yet.")
    assert next(iter(user.find(kind=ui.tab_panels).elements)).value == "Memories"
