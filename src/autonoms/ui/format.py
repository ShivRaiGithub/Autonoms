"""Small pure helpers that turn API data into things worth showing. No UI code here, so they are easy to test."""

MAX_DIFF_LINES = 400  # a longer diff is cut in the view (the full one is still what gets approved)

# run state -> (words for a person, Quasar colour)
RUN_STATES = {
    "not_started": ("not started", "grey"),
    "running": ("running", "blue"),
    "interrupted": ("interrupted: can be resumed", "orange"),
    "finished": ("finished", "green"),
}
RUN_STATUSES = {
    "finished": "The model finished.",
    "finished_with_regressions": "Finished, but the edits broke something that was correct before.",
    "no_tool_call": "Stopped: the model answered without using a tool.",
    "max_steps": "Stopped: the step limit was reached.",
    "context_limit": "Stopped: the conversation grew too long for the model.",
    "llm_error": "Stopped: the model could not be reached.",
}


def run_state(state: str) -> tuple[str, str]:
    return RUN_STATES.get(state, (state, "grey"))


def run_status_text(status: str) -> str:
    return RUN_STATUSES.get(status, status)


def diff_lines(diff: str) -> tuple[list[tuple[str, str]], bool]:
    """(kind, text) per line, kind in add/remove/header/context, and whether the view was cut."""
    lines = []
    for line in diff.splitlines():
        if line.startswith(("diff --git", "index ", "--- ", "+++ ", "@@")):
            kind = "header"
        elif line.startswith("+"):
            kind = "add"
        elif line.startswith("-"):
            kind = "remove"
        else:
            kind = "context"
        lines.append((kind, line))
    return lines[:MAX_DIFF_LINES], len(lines) > MAX_DIFF_LINES


def short_fingerprint(fingerprint: str) -> str:
    return f"{fingerprint[:8]}...{fingerprint[-4:]}" if len(fingerprint) >= 12 else fingerprint


def yes_no_unknown(value: bool | int | None) -> str:
    """A check that could not run is 'unknown', never 'no' or 0."""
    return "unknown" if value is None else ("yes" if value is True else "no" if value is False else str(value))


def step_text(step: dict) -> str:
    """One tool step as a short line: the tool and its main argument."""
    args = step.get("arguments", {})
    main = args.get("path") or args.get("query") or args.get("summary") or ""
    return f"{step['tool']}({str(main)[:60]})" if main else f"{step['tool']}()"
