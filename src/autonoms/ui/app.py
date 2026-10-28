"""The web page: create a task, run the agent, watch it work, read the result, approve the pull request.

One page, one `Dashboard` object per browser tab. It never decides anything itself: every rule
(what can be published, what needs acknowledging) is enforced by the API, and this page only
shows what the API says and sends back what the person chose.
"""

from nicegui import ui

from autonoms.ui import format as fmt
from autonoms.ui.api import Api, ApiError

POLL_SECONDS = 1.5
DIFF_COLORS = {"add": "text-green-700", "remove": "text-red-700", "header": "text-grey-6", "context": ""}


class Dashboard:
    def __init__(self, api: Api):
        self.api = api
        # what is on screen
        self.tasks: list[dict] = []
        self.task: dict | None = None
        self.status: dict = {"state": "not_started"}
        self.review: dict | None = None
        self.evaluation: dict | None = None
        self.memories: list[dict] = []
        self.publication: dict | None = None
        self.api_up = True
        self.busy = ""  # what the page is waiting for, e.g. "Running the agent", or "" when idle
        self.error = ""
        # form values (bound to inputs)
        self.owner, self.repo, self.number = "Ssr4803", "TestingPythonRepo", 1
        self.max_steps, self.max_repairs = 8, 1
        self.approver, self.acknowledged = "", False
        self.tab = "Result"
        self.memory_kind, self.memory_text = "convention", ""
        self.built = False  # refreshing a part that does not exist yet would be an error

    def field(self, attr: str) -> dict:
        """Keyword arguments that tie an input to `self.<attr>`: starts from its value, and updates it
        the instant the input changes. (`bind_value` also works, but it copies on a timer, so a click
        right after typing could read the old value.)"""
        return {"value": getattr(self, attr), "on_change": lambda e: setattr(self, attr, e.value)}

    # ------------------------------------------------------------------ building the page

    def build(self) -> None:
        with ui.header().classes("items-center justify-between"):
            ui.label("Autonoms").classes("text-h6")
            ui.label("Autonomous GitHub issue resolution: you stay in control").classes("text-caption")
            self.connection()
        self.error_banner()
        self.busy_line()
        # Dialogs live here, not inside a part of the page that redraws: when a refreshable
        # part redraws it destroys everything created inside it, including an open dialog.
        self.dialogs = ui.element("div")
        with ui.row().classes("w-full no-wrap items-start"):
            with ui.column().classes("w-96 shrink-0"):
                self.new_task_form()
                self.task_list()
            with ui.column().classes("grow min-w-0"):
                self.detail()
        ui.timer(POLL_SECONDS, self.poll)
        self.built = True

    @ui.refreshable
    def connection(self) -> None:
        if self.api_up:
            ui.badge("API connected", color="positive")
        else:
            ui.badge("API unreachable", color="negative")

    # ------------------------------------------------------------------ left column

    def new_task_form(self) -> None:
        with ui.card().classes("w-full"):
            ui.label("New task").classes("text-subtitle1")
            ui.input("Owner", **self.field("owner")).props("dense outlined").classes("w-full")
            ui.input("Repository", **self.field("repo")).props("dense outlined").classes("w-full")
            ui.number("Issue number", min=1, precision=0, **self.field("number")).props("dense outlined").classes("w-full")
            ui.button("Create task", icon="add", on_click=self.create_task).props("no-caps")

    @ui.refreshable
    def task_list(self) -> None:
        ui.label("Tasks").classes("text-subtitle1")
        if not self.tasks:
            ui.label("No tasks yet.").classes("text-grey")
        with ui.list().props("bordered separator").classes("w-full"):
            for task in self.tasks:
                selected = self.task is not None and task["id"] == self.task["id"]
                with ui.item(on_click=lambda t=task: self.select(t["id"])).classes("bg-blue-1" if selected else ""):
                    with ui.item_section():
                        ui.item_label(f"#{task['issue_number']} {task['issue_title']}")
                        ui.item_label(f"{task['owner']}/{task['repo']}").props("caption")

    # ------------------------------------------------------------------ right column

    @ui.refreshable
    def detail(self) -> None:
        if self.task is None:
            ui.label("Create a task, or select one on the left.").classes("text-grey")
            return
        issue = self.task["issue"]
        with ui.card().classes("w-full"):
            ui.label(f"#{self.task['issue_number']}  {self.task['issue_title']}").classes("text-h6")
            with ui.row().classes("items-center"):
                ui.link(f"{self.task['owner']}/{self.task['repo']}", issue["url"], new_tab=True)
                for label in issue["labels"]:
                    ui.badge(label, color="grey")
            ui.label(issue["body"] or "(no description)").classes("whitespace-pre-wrap")  # plain text: issue text is untrusted
        self.controls()
        self.progress()
        # value= sets the starting tab explicitly. Binding it instead left no tab selected and an empty panel
        # until the first click (seen in a real browser; the simulated-user tests could not see it).
        with ui.tabs(value=self.tab, on_change=lambda e: setattr(self, "tab", e.value)).classes("w-full") as tabs:
            ui.tab("Result")
            ui.tab("Review & publish")
            ui.tab("Memories")
        with ui.tab_panels(tabs, value=self.tab).classes("w-full"):
            with ui.tab_panel("Result"):
                self.result_panel()
            with ui.tab_panel("Review & publish"):
                self.publish_panel()
            with ui.tab_panel("Memories"):
                self.memories_panel()

    @ui.refreshable
    def controls(self) -> None:
        state = self.status["state"]
        idle = not self.busy
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center"):
                ui.button("1. Get the repository", on_click=self.clone).props("no-caps outline").set_enabled(idle)
                ui.number("Max steps", min=1, max=15, precision=0, **self.field("max_steps")).props("dense outlined").classes("w-28")
                ui.number("Max repairs", min=0, max=3, precision=0, **self.field("max_repairs")).props("dense outlined").classes("w-28")
                if state == "interrupted":
                    ui.button("Resume the interrupted run", icon="play_arrow", on_click=self.resume).props("no-caps color=orange").set_enabled(idle)
                if state == "finished":
                    ui.button("Run again", icon="replay", on_click=lambda: self.run(restart=True)).props("no-caps").set_enabled(idle)
                elif state != "interrupted":
                    ui.button("2. Run the agent", icon="play_arrow", on_click=lambda: self.run(restart=False)).props("no-caps color=primary").set_enabled(idle)
                ui.space()
                ui.button("Delete task", icon="delete", on_click=self.confirm_delete).props("no-caps flat color=negative").set_enabled(idle)

    @ui.refreshable
    def busy_line(self) -> None:
        if self.busy:
            with ui.row().classes("items-center"):
                ui.spinner(size="sm")
                ui.label(self.busy)

    @ui.refreshable
    def error_banner(self) -> None:
        if self.error:
            with ui.card().classes("w-full bg-red-1"):
                ui.label(self.error).classes("text-negative whitespace-pre-wrap")

    @ui.refreshable
    def progress(self) -> None:
        text, color = fmt.run_state(self.status["state"])
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center"):
                ui.label("Run").classes("text-subtitle1")
                ui.badge(text, color=color)
                if self.status.get("model_calls"):
                    ui.label(f"{self.status['model_calls']} model call(s), {self.status.get('repairs', 0)} repair round(s)").classes("text-caption")
            if self.status.get("next"):
                ui.label("Next step in the workflow: " + ", ".join(self.status["next"])).classes("text-caption")
            with ui.column().classes("gap-0"):
                for step in self.status.get("steps", []):
                    ui.label(fmt.step_text(step)).classes("font-mono text-caption")

    # --- result tab

    @ui.refreshable
    def result_panel(self) -> None:
        if self.review is None:
            ui.label("The result appears here once a run has finished.").classes("text-grey")
            return
        if self.status.get("status"):
            ui.label(fmt.run_status_text(self.status["status"])).classes("text-subtitle1")
        if self.evaluation:
            e = self.evaluation
            with ui.row().classes("items-center"):
                ui.badge("completed" if e["completed"] else "not completed", color="positive" if e["completed"] else "negative")
                ui.badge("clean" if e["clean"] else "not clean", color="positive" if e["clean"] else "warning")
                ui.badge("in scope" if e["in_scope"] else "touches other files", color="positive" if e["in_scope"] else "warning")
            ui.label(
                f"Wrong equations fixed: {fmt.yes_no_unknown(e['equations_fixed'])}   newly broken: {fmt.yes_no_unknown(e['equations_broken'])}   "
                f"files changed: {e['files_changed']}"
            ).classes("text-caption")
            ui.label(
                f"{e['seconds']} s in total ({e['model_seconds']} s waiting for the model), {e['model_calls']} model calls, "
                f"{e['prompt_tokens']} tokens read, {e['completion_tokens']} written"
            ).classes("text-caption")
        ui.label("The change").classes("text-subtitle1 q-mt-md")
        self.diff_view(self.review["diff"])

    @staticmethod
    def diff_view(diff: str) -> None:
        lines, cut = fmt.diff_lines(diff)
        with ui.card().classes("w-full"):
            if not lines:
                ui.label("No files were changed.").classes("text-grey")
            # gap-0: lines should touch; overflow-x-auto: a long line scrolls sideways instead of being cut off
            with ui.column().classes("w-full gap-0 overflow-x-auto"):
                for kind, text in lines:
                    ui.label(text or " ").classes(f"font-mono whitespace-pre text-caption leading-snug {DIFF_COLORS[kind]}")
            if cut:
                ui.label(f"(showing the first {fmt.MAX_DIFF_LINES} lines; the whole change is what gets published)").classes("text-caption text-grey")

    # --- review & publish tab

    @ui.refreshable
    def publish_panel(self) -> None:
        review = self.review
        if review is None:
            ui.label("Run the agent first. The review is the report you read before anything is published.").classes("text-grey")
            return
        ui.label(review["title"]).classes("text-h6")
        ui.label(f"Branch {review['branch']}   ·   files: {', '.join(review['files']) or 'none'}   ·   fingerprint {fmt.short_fingerprint(review['fingerprint'])}").classes("text-caption")
        for blocker in review["blockers"]:
            ui.label("Cannot be published: " + blocker).classes("text-negative")
        for risk in review["risks"]:
            warn = risk["level"] == "warn"
            with ui.row().classes("items-start no-wrap"):
                ui.icon("warning" if warn else "info", color="warning" if warn else "grey")
                ui.label(risk["text"]).classes("text-orange-9" if warn else "text-grey-8")
        with ui.expansion("The pull request text that would be posted").classes("w-full"):
            ui.label(review["body"]).classes("whitespace-pre-wrap font-mono text-caption")
        if self.publication or review.get("publication"):
            pub = self.publication or review["publication"]
            with ui.card().classes("w-full bg-green-1"):
                ui.label(f"Published as {'draft ' if pub['draft'] else ''}pull request #{pub['pr_number']} by approval of {pub['approved_by']}.")
                ui.link(pub["pr_url"], pub["pr_url"], new_tab=True)
            return
        with ui.card().classes("w-full"):
            ui.label("Your approval").classes("text-subtitle1")
            ui.label("Read the diff on the Result tab and the text above first. Approving applies to exactly this version, shown by the fingerprint.").classes("text-caption")
            ui.input("Your name", **self.field("approver")).props("dense outlined").classes("w-72")
            if review["needs_acknowledgement"]:
                ui.checkbox("I have read the warnings above and still want to publish", **self.field("acknowledged"))
            ui.button("Approve and publish as draft PR", icon="publish", on_click=self.confirm_publish).props("no-caps color=primary").set_enabled(
                review["can_publish"] and not self.busy
            )
            ui.button("Reload the review", icon="refresh", on_click=self.load_result).props("no-caps flat")

    # --- memories tab

    @ui.refreshable
    def memories_panel(self) -> None:
        repo = f"{self.task['owner']}/{self.task['repo']}"
        ui.label(f"What Autonoms remembers about {repo} (and about every repository)").classes("text-subtitle1")
        if not self.memories:
            ui.label("Nothing yet.").classes("text-grey")
        for memory in self.memories:
            with ui.row().classes("items-center w-full no-wrap"):
                ui.badge(memory["kind"], color="grey")
                ui.label(memory["content"]).classes("grow")
                ui.label(f"used {memory['use_count']}x").classes("text-caption text-grey")
                ui.button(icon="delete", on_click=lambda m=memory: self.delete_memory(m["id"])).props("flat dense color=negative")
        with ui.row().classes("items-end w-full"):
            ui.select(["convention", "preference", "pitfall", "finding"], label="Kind", **self.field("memory_kind")).props("dense outlined").classes("w-40")
            ui.input("A short note to remember (no secrets)", **self.field("memory_text")).props("dense outlined").classes("grow")
            ui.button("Remember", icon="add", on_click=self.add_memory).props("no-caps")

    # ------------------------------------------------------------------ actions

    async def guarded(self, action, busy: str = "") -> None:
        """Run an API action; show any failure instead of crashing the page."""
        self.error = ""
        self.busy = busy
        self.controls.refresh()
        self.busy_line.refresh()
        self.error_banner.refresh()
        try:
            await action()
        except ApiError as exc:
            self.error = exc.message
            # The banner is at the top of the page, out of sight when you are scrolled down at a button.
            # A notification stays where you are looking until you dismiss it.
            ui.notify(exc.message, type="negative", multi_line=True, close_button="Dismiss", timeout=0)
            if exc.status == 0:
                self.api_up = False
                self.connection.refresh()
        finally:
            self.busy = ""
            self.controls.refresh()
            self.busy_line.refresh()
            self.error_banner.refresh()
            self.publish_panel.refresh()  # its buttons depend on "busy": without this a finished run leaves them disabled

    async def refresh_tasks(self) -> None:
        try:
            self.tasks = await self.api.tasks()
            self.api_up = True
        except ApiError:
            self.api_up = False
        if self.built:
            self.connection.refresh()
            self.task_list.refresh()

    async def create_task(self) -> None:
        async def action() -> None:
            task = await self.api.create_task(self.owner.strip(), self.repo.strip(), int(self.number or 0))
            await self.refresh_tasks()
            await self.select(task["id"])
            ui.notify("Task created. Next: get the repository.", type="positive")

        await self.guarded(action, "Fetching the issue from GitHub")

    async def select(self, task_id: str) -> None:
        async def action() -> None:
            self.task = await self.api.get_task(task_id)
            self.review = self.evaluation = self.publication = None
            self.status = await self.api.run_status(task_id)
            self.acknowledged = False
            await self.load_memories()
            if self.status["state"] == "finished":
                await self.load_result()

        await self.guarded(action)
        self.detail.refresh()
        self.task_list.refresh()

    async def clone(self) -> None:
        async def action() -> None:
            summary = await self.api.create_workspace(self.task["id"])
            ui.notify(f"Repository ready: {summary['file_count']} files on branch {summary['branch']}.", type="positive")

        await self.guarded(action, "Cloning the repository")

    async def run(self, restart: bool) -> None:
        async def action() -> None:
            try:
                await self.api.run(self.task["id"], int(self.max_steps or 8), int(self.max_repairs or 0), restart)
            finally:  # even if the run failed: a crash leaves an "interrupted" run, and the page should say so now
                await self.sync_status()
            await self.load_result()
            await self.refresh_tasks()
            ui.notify("The run finished. Read the result before publishing anything.", type="info")

        self.review = self.evaluation = self.publication = None
        self.result_panel.refresh()
        self.publish_panel.refresh()
        await self.guarded(action, "The agent is working (this takes about a minute on a local model)")

    async def resume(self) -> None:
        async def action() -> None:
            try:
                await self.api.resume(self.task["id"])
            finally:
                await self.sync_status()
            await self.load_result()

        await self.guarded(action, "Resuming from the last checkpoint")

    async def sync_status(self) -> None:
        """Fetch the run state now (the timer would do it within a second or two)."""
        try:
            self.status = await self.api.run_status(self.task["id"])
        except ApiError:
            return
        self.progress.refresh()

    async def load_result(self) -> None:
        async def action() -> None:
            self.review = await self.api.review(self.task["id"])
            self.evaluation = await self.api.evaluation(self.task["id"])
            self.publication = self.review.get("publication")

        try:
            await action()
        except ApiError as exc:
            if exc.status != 409:  # 409 = "no finished run yet": nothing to show, not an error
                self.error = exc.message
        self.error_banner.refresh()
        self.progress.refresh()
        self.result_panel.refresh()
        self.publish_panel.refresh()

    async def poll(self) -> None:
        """Every moment: ask the server how the run is going (read from its latest checkpoint)."""
        if self.task is None:
            return
        try:
            status = await self.api.run_status(self.task["id"])
        except ApiError:
            return
        if status != self.status:
            was_finished = self.status.get("state") == "finished"
            self.status = status
            self.progress.refresh()
            self.controls.refresh()
            if status["state"] == "finished" and not was_finished and self.review is None:
                await self.load_result()

    # --- publishing

    async def confirm_publish(self) -> None:
        review = self.review  # what is on screen right now: this exact version is what gets approved
        if not self.approver.strip():
            ui.notify("Please type your name first: it is recorded as the approver.", type="warning")
            return
        if review["needs_acknowledgement"] and not self.acknowledged:
            ui.notify("This review has warnings. Tick the box to say you have read them.", type="warning")
            return
        with self.dialogs, ui.dialog() as dialog, ui.card():
            ui.label("Publish this as a draft pull request?").classes("text-h6")
            ui.label(f"Repository: {self.task['owner']}/{self.task['repo']}")
            ui.label(f"Branch: {review['branch']}")
            ui.label(f"Title: {review['title']}")
            ui.label(f"Files: {', '.join(review['files'])}")
            ui.label(f"Version you reviewed: {fmt.short_fingerprint(review['fingerprint'])}").classes("font-mono")
            ui.label("This pushes the branch to GitHub and opens a DRAFT pull request. Autonoms never merges anything.").classes("text-caption")
            with ui.row():
                ui.button("Cancel", on_click=lambda: dialog.submit(False)).props("flat no-caps").mark("publish-cancel")
                ui.button("Approve and publish", on_click=lambda: dialog.submit(True)).props("no-caps color=primary").mark("publish-confirm")
        confirmed = await dialog
        dialog.delete()  # a closed dialog would otherwise stay on the page for good
        if not confirmed:
            return

        async def action() -> None:
            self.publication = await self.api.publish(
                self.task["id"], review["fingerprint"], self.approver.strip(), bool(self.acknowledged)
            )
            ui.notify(f"Draft pull request #{self.publication['pr_number']} opened.", type="positive")
            self.publish_panel.refresh()

        await self.guarded(action, "Publishing to GitHub")
        self.publish_panel.refresh()

    # --- memories and deleting

    async def load_memories(self) -> None:
        self.memories = await self.api.memories(f"{self.task['owner']}/{self.task['repo']}")

    async def add_memory(self) -> None:
        async def action() -> None:
            await self.api.add_memory(f"{self.task['owner']}/{self.task['repo']}", self.memory_kind, self.memory_text.strip())
            self.memory_text = ""
            await self.load_memories()
            self.memories_panel.refresh()

        await self.guarded(action)

    async def delete_memory(self, memory_id: str) -> None:
        async def action() -> None:
            await self.api.delete_memory(memory_id)
            await self.load_memories()
            self.memories_panel.refresh()

        await self.guarded(action)

    async def confirm_delete(self) -> None:
        with self.dialogs, ui.dialog() as dialog, ui.card():
            ui.label("Delete this task?").classes("text-h6")
            ui.label("Its repository copy, search index and saved run are removed. A pull request that was already published stays on GitHub.").classes("text-caption")
            with ui.row():
                ui.button("Cancel", on_click=lambda: dialog.submit(False)).props("flat no-caps").mark("delete-cancel")
                ui.button("Delete", on_click=lambda: dialog.submit(True)).props("no-caps color=negative").mark("delete-confirm")
        confirmed = await dialog
        dialog.delete()
        if not confirmed:
            return

        async def action() -> None:
            await self.api.delete_task(self.task["id"])
            self.task, self.review, self.evaluation, self.publication = None, None, None, None
            self.status = {"state": "not_started"}
            await self.refresh_tasks()

        await self.guarded(action)
        self.detail.refresh()


def register(api: Api) -> None:
    """Add the page to the NiceGUI app. Called once at startup (and by the tests)."""

    @ui.page("/")
    async def index() -> None:
        ui.query("body").classes("bg-grey-2")
        dashboard = Dashboard(api)
        dashboard.build()  # draw at once: a slow or dead API must not stop the page from appearing
        await ui.context.client.connected()
        await dashboard.refresh_tasks()
