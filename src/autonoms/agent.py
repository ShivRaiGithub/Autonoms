"""The agent: a plain loop where the model chooses tools until it calls `finish`.

No framework. The loop is: ask the model -> run the tools it asked for -> show
it the results -> repeat. Everything is bounded: steps, tool-output size and
total conversation size.
"""

import time

from pydantic import BaseModel, Field, ValidationError

from autonoms import workspace
from autonoms.llm import LLM, LLMError
from autonoms.checker import Checker
from autonoms.models import AgentRun, AgentStep, CheckResult, Task
from autonoms.search import SearchIndexes
from autonoms.workspace import Workspaces

MAX_TOOL_OUTPUT_CHARS = 3000  # about 750 tokens; the model's window is only 8k
MAX_CONVERSATION_CHARS = 20000  # about 5k tokens: leaves room for the answer

SYSTEM_PROMPT = """You are Autonoms, a careful assistant that fixes issues in a small repository of notes.

Rules:
- Use search_code and read_file to find the right file. Do not guess file contents.
- Fix the problem with replace_in_file. Change only what the issue asks for.
- old_text must be copied exactly from the file and must appear only once.
- After editing, call run_check to verify the arithmetic. If it reports problems you can fix, fix them.
- When the fix is done, call finish with a one-sentence summary.
- If you cannot fix it, call finish and say why. Never claim a fix you did not make."""


class SearchArgs(BaseModel):
    query: str = Field(min_length=1, max_length=500)


class ReadArgs(BaseModel):
    path: str


class ReplaceArgs(BaseModel):
    path: str
    old_text: str
    new_text: str


class NoArgs(BaseModel):
    pass


class FinishArgs(BaseModel):
    summary: str


def _tool(name: str, description: str, model: type[BaseModel]) -> dict:
    schema = model.model_json_schema()
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    return {"type": "function", "function": {"name": name, "description": description, "parameters": schema}}


TOOLS = [
    _tool("search_code", "Search the repository files by keywords or meaning. Returns file paths with snippets.", SearchArgs),
    _tool("read_file", "Read a whole text file from the repository.", ReadArgs),
    _tool(
        "replace_in_file",
        "Replace one exact piece of text in a file with new text. old_text must appear exactly once.",
        ReplaceArgs,
    ),
    _tool(
        "run_check",
        "Check the arithmetic in all notes (sums like '12 / 4 = 3'). Lists wrong equations. Cannot judge prose.",
        NoArgs,
    ),
    _tool("finish", "Call when the work is done (or impossible). Ends the run.", FinishArgs),
]


def describe(check: CheckResult) -> str:
    """The check result as text for the model. Not-run is never worded like a pass."""
    if check.status == "not_run":
        return f"CHECK NOT RUN: {check.detail}. Nothing was verified."
    if check.status == "passed":
        return f"CHECK PASSED: {check.equations} equations checked, all correct."
    lines = [f"CHECK FAILED: {len(check.problems)} wrong of {check.equations} equations checked."]
    lines += [f"- {p.file} line {p.line}: {p.text} (the result should be {p.actual})" for p in check.problems]
    return "\n".join(lines)


def _clip(text: str) -> str:
    if len(text) <= MAX_TOOL_OUTPUT_CHARS:
        return text
    return text[:MAX_TOOL_OUTPUT_CHARS] + f"\n[... cut: {len(text) - MAX_TOOL_OUTPUT_CHARS} more characters]"


class Agent:
    def __init__(self, llm: LLM, workspaces: Workspaces, indexes: SearchIndexes, checker: Checker):
        self._llm = llm
        self._checker = checker
        self._workspaces = workspaces
        self._indexes = indexes

    # --- tools: each returns text for the model; errors are text too, not crashes ---

    def _run_tool(self, task_id: str, name: str, arguments: dict) -> tuple[str, bool]:
        """Returns (result_text, is_finish)."""
        try:
            if name == "search_code":
                args = SearchArgs(**arguments)
                result = self._indexes.search(task_id, args.query, "hybrid", top_k=3)
                if not result.hits:
                    return "No matches.", False
                return "\n\n".join(f"[{h.path}]\n{h.text}" for h in result.hits), False
            if name == "read_file":
                args = ReadArgs(**arguments)
                return self._workspaces.read_file(task_id, args.path), False
            if name == "replace_in_file":
                args = ReplaceArgs(**arguments)
                self._workspaces.replace_in_file(task_id, args.path, args.old_text, args.new_text)
                self._indexes.build(task_id)  # the files changed, so search must see the new text
                return f"OK: edited {args.path}", False
            if name == "run_check":
                return describe(self._checker.run(task_id)), False
            if name == "finish":
                return FinishArgs(**arguments).summary, True
            return f"Error: unknown tool '{name}'. Available: search_code, read_file, replace_in_file, finish.", False
        except ValidationError:
            return f"Error: wrong arguments for {name}. Check the parameter names and types.", False
        except workspace.InvalidPath:
            return "Error: no such file in the repository (check the path from the file list).", False
        except workspace.BinaryFile:
            return "Error: that is not a text file.", False
        except workspace.FileTooLarge:
            return "Error: that file is too large to read.", False
        except workspace.EditError as exc:
            return f"Error: {exc}", False

    def _opening_message(self, task: Task) -> str:
        files = [f.path for f in self._workspaces.summary(task.id).files]
        hits = self._indexes.search(task.id, f"{task.issue.title}\n{task.issue.body}"[:2000], "hybrid", 3).hits
        likely = ", ".join(dict.fromkeys(h.path for h in hits))
        return (
            f"Issue #{task.issue_number}: {task.issue.title}\n\n{task.issue.body}\n\n"
            f"Files in the repository: {', '.join(files[:50])}\n"
            f"Search suggests these may be relevant: {likely}"
        )

    # --- the loop ------------------------------------------------------------------

    def run(self, task: Task, max_steps: int = 8) -> AgentRun:
        started = time.monotonic()
        check_before = self._checker.run(task.id)  # the baseline: what was wrong before we touched anything
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._opening_message(task)},
        ]
        steps: list[AgentStep] = []
        status, summary = "max_steps", "Stopped: step limit reached."

        for _ in range(max_steps):
            if sum(len(m["content"]) for m in messages) > MAX_CONVERSATION_CHARS:
                status, summary = "context_limit", "Stopped: the conversation grew too long for the model."
                break
            try:
                reply = self._llm.chat(messages, TOOLS)
            except LLMError as exc:
                status, summary = "llm_error", str(exc)
                break
            messages.append(reply)
            if not reply["tool_calls"]:
                status, summary = "no_tool_call", reply["content"] or "The model answered without using a tool."
                break
            finished = False
            for call in reply["tool_calls"]:
                text, is_finish = self._run_tool(task.id, call["name"], call["arguments"])
                text = _clip(text)
                steps.append(AgentStep(tool=call["name"], arguments=call["arguments"], result=text))
                messages.append({"role": "tool", "tool_name": call["name"], "content": text})
                if is_finish:
                    status, summary, finished = "finished", text, True
                    break
            if finished:
                break

        return AgentRun(
            status=status,
            summary=summary,
            steps=steps,
            diff=self._workspaces.diff(task.id),
            check_before=check_before,
            check_after=self._checker.run(task.id),
            seconds=round(time.monotonic() - started, 1),
        )
