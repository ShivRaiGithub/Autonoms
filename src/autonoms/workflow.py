"""The agent as a LangGraph workflow with checkpoints.

Stage 5's loop, split into named steps (nodes) joined by routing rules (edges):

    baseline -> agent -> tools -> verify -> finalize
                  ^        |        |
                  +--------+        +-- check found NEW wrong equations? -> back to agent (repair)

After every node LangGraph saves the whole state (the "checkpoint") in Postgres,
so a run that dies can be resumed from the last saved step instead of starting over.
Everything in the state is plain JSON (dicts, lists, strings, numbers): that is
what gets stored.
"""

import hashlib
import operator
import time
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from autonoms.agent import MAX_CONVERSATION_CHARS, SYSTEM_PROMPT, TOOLS, Agent, clip
from autonoms.checker import Checker
from autonoms.llm import LLM, LLMError
from autonoms.context import select_context, size
from autonoms.models import AgentRun, AgentStep, CheckResult, Memory, RunStatus, Task, Usage
from autonoms.workspace import Workspaces

RECURSION_LIMIT = 150  # LangGraph's own safety net; our limits (max_steps, repairs) bite first


class RunState(TypedDict, total=False):
    # inputs
    task: dict
    max_steps: int
    max_repairs: int
    # the conversation and trace; `operator.add` means "append what a node returns"
    messages: Annotated[list, operator.add]
    steps: Annotated[list, operator.add]
    seconds: Annotated[float, operator.add]
    # progress
    status: str  # running -> finished | finished_with_regressions | no_tool_call | max_steps | ...
    summary: str
    finish_summary: str | None
    model_calls: int
    repairs: int
    prompt_tokens: Annotated[int, operator.add]  # real counts reported by the model server
    completion_tokens: Annotated[int, operator.add]
    peak_prompt_tokens: int
    model_seconds: Annotated[float, operator.add]
    models: Annotated[list, operator.add]  # the model that answered each call
    compactions: Annotated[int, operator.add]  # calls where older steps were collapsed
    memory_ids: list  # long-term memories shown to the model
    check_before: dict
    check_after: dict | None
    diff_digest: str  # fingerprint of the workspace edits at the last checkpoint
    diff: str


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def regressions(before: dict, after: dict) -> list[dict]:
    """Wrong equations present after the run that were not wrong before it."""
    known = {(p["file"], p["text"]) for p in before["problems"]}
    return [p for p in after["problems"] if (p["file"], p["text"]) not in known]


class WorkspaceChanged(Exception):
    """The workspace differs from what the checkpoint recorded; resuming would be unsafe."""


class Workflow:
    def __init__(
        self,
        llm: LLM,
        workspaces: Workspaces,
        agent: Agent,
        checker: Checker,
        checkpointer: BaseCheckpointSaver,
        compact_at: int = 12000,
        memory_lookup: Callable[[Task], list[Memory]] | None = None,
    ):
        self._compact_at = compact_at
        self._memory_lookup = memory_lookup or (lambda task: [])
        self._llm = llm
        self._workspaces = workspaces
        self._agent = agent  # reused for its tools and opening message (Stage 5 code)
        self._checker = checker
        self._checkpointer = checkpointer
        self._graph = self._build()

    # --- nodes: each takes the state and returns only the parts it changes ----------

    def _timed(self, node):
        def wrapper(state: RunState) -> dict:
            started = time.monotonic()
            update = node(state)
            return {**update, "seconds": time.monotonic() - started}

        wrapper.__name__ = node.__name__
        return wrapper

    def _baseline(self, state: RunState) -> dict:
        task = Task.model_validate(state["task"])
        before = self._checker.run(task.id)  # what was wrong before we touched anything
        memories = self._memory_lookup(task)
        return {
            "memory_ids": [m.id for m in memories],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self._agent.opening_message(task, memories)},
            ],
            "check_before": before.model_dump(),
            "diff_digest": digest(self._workspaces.diff(task.id)),
            "status": "running",
            "model_calls": 0,
            "repairs": 0,
        }

    def _agent_node(self, state: RunState) -> dict:
        if state["model_calls"] >= state["max_steps"]:
            return {"status": "max_steps", "summary": "Stopped: step limit reached."}
        # The model sees a view: the full history stays in the state, but older steps
        # are collapsed once the conversation is large (see context.py).
        view, compacted = select_context(state["messages"], self._compact_at)
        if size(view) > MAX_CONVERSATION_CHARS:  # too big even after collapsing
            return {"status": "context_limit", "summary": "Stopped: the conversation grew too long for the model."}
        try:
            reply = dict(self._llm.chat(view, TOOLS))
        except LLMError as exc:
            return {"status": "llm_error", "summary": str(exc)}
        usage = reply.pop("usage", None) or {}
        prompt_tokens = usage.get("prompt_tokens", 0)
        update: dict[str, Any] = {
            "messages": [reply],
            "model_calls": state["model_calls"] + 1,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": usage.get("completion_tokens", 0),
            "peak_prompt_tokens": max(state.get("peak_prompt_tokens", 0), prompt_tokens),
            "compactions": 1 if compacted else 0,
            "model_seconds": usage.get("seconds", 0.0),
            "models": [usage["model"]] if usage.get("model") else [],
        }
        if not reply["tool_calls"]:
            update |= {"status": "no_tool_call", "summary": reply["content"] or "The model answered without using a tool."}
        return update

    def _tools_node(self, state: RunState) -> dict:
        task_id = state["task"]["id"]
        messages, steps, finish_summary = [], [], None
        for call in state["messages"][-1]["tool_calls"]:
            baseline = CheckResult(**state["check_before"])
            text, is_finish = self._agent.run_tool(task_id, call["name"], call["arguments"], baseline)
            text = clip(text)
            steps.append(AgentStep(tool=call["name"], arguments=call["arguments"], result=text).model_dump())
            messages.append({"role": "tool", "tool_name": call["name"], "content": text})
            if is_finish:
                finish_summary = text
                break
        return {
            "messages": messages,
            "steps": steps,
            "finish_summary": finish_summary,
            "diff_digest": digest(self._workspaces.diff(task_id)),
        }

    def _verify_node(self, state: RunState) -> dict:
        """The model says it is done. Check, and send it back if it broke something."""
        after = self._checker.run(state["task"]["id"]).model_dump()
        broke = regressions(state["check_before"], after) if after["status"] != "not_run" else []
        if broke and state["repairs"] < state["max_repairs"]:
            listing = "\n".join(f"- {p['file']} line {p['line']}: {p['text']} (should be {p['actual']})" for p in broke)
            return {
                "check_after": after,
                "repairs": state["repairs"] + 1,
                "finish_summary": None,
                "messages": [
                    {
                        "role": "user",
                        "content": "The check found equations that were correct before your edits and are wrong now:\n"
                        f"{listing}\nFix them with replace_in_file, then call finish again.",
                    }
                ],
            }
        status = "finished_with_regressions" if broke else "finished"
        return {"check_after": after, "status": status, "summary": state["finish_summary"] or ""}

    def _finalize_node(self, state: RunState) -> dict:
        task_id = state["task"]["id"]
        after = state.get("check_after")
        if after is None:  # the run stopped before verification: check what we have
            after = self._checker.run(task_id).model_dump()
        return {"check_after": after, "diff": self._workspaces.diff(task_id)}

    # --- routing rules ---------------------------------------------------------------

    @staticmethod
    def _after_agent(state: RunState) -> str:
        return "finalize" if state["status"] != "running" else "tools"

    @staticmethod
    def _after_tools(state: RunState) -> str:
        return "verify" if state.get("finish_summary") is not None else "agent"

    @staticmethod
    def _after_verify(state: RunState) -> str:
        return "finalize" if state["status"] != "running" else "agent"

    def _build(self):
        graph = StateGraph(RunState)
        for name, node in {
            "baseline": self._baseline,
            "agent": self._agent_node,
            "tools": self._tools_node,
            "verify": self._verify_node,
            "finalize": self._finalize_node,
        }.items():
            graph.add_node(name, self._timed(node))
        graph.add_edge(START, "baseline")
        graph.add_edge("baseline", "agent")
        graph.add_conditional_edges("agent", self._after_agent, ["tools", "finalize"])
        graph.add_conditional_edges("tools", self._after_tools, ["verify", "agent"])
        graph.add_conditional_edges("verify", self._after_verify, ["agent", "finalize"])
        graph.add_edge("finalize", END)
        return graph.compile(checkpointer=self._checkpointer)

    # --- running, inspecting, resuming ---------------------------------------------

    @staticmethod
    def _config(task_id: str) -> dict:
        # thread_id is the stable name of this run's saved history: one per task.
        return {"configurable": {"thread_id": task_id}, "recursion_limit": RECURSION_LIMIT}

    def _snapshot(self, task_id: str):
        return self._graph.get_state(self._config(task_id))

    def exists(self, task_id: str) -> bool:
        return bool(self._snapshot(task_id).values)

    def finished(self, task_id: str) -> bool:
        snapshot = self._snapshot(task_id)
        return bool(snapshot.values) and not snapshot.next

    def reset(self, task_id: str) -> None:
        self._checkpointer.delete_thread(task_id)

    def start(self, task: Task, max_steps: int, max_repairs: int) -> AgentRun:
        self._graph.invoke(
            {"task": task.model_dump(mode="json"), "max_steps": max_steps, "max_repairs": max_repairs, "seconds": 0.0},
            self._config(task.id),
        )
        return self.result(task.id)

    def resume(self, task_id: str) -> AgentRun:
        values = self._snapshot(task_id).values
        # Resuming must not trust that nothing happened while we were down.
        if digest(self._workspaces.diff(task_id)) != values["diff_digest"]:
            raise WorkspaceChanged(task_id)
        self._graph.invoke(None, self._config(task_id))  # None = "continue from the checkpoint"
        return self.result(task_id)

    def status(self, task_id: str, running: bool) -> RunStatus:
        snapshot = self._snapshot(task_id)
        if not snapshot.values:
            return RunStatus(state="not_started")
        v = snapshot.values
        state = "finished" if not snapshot.next else ("running" if running else "interrupted")
        return RunStatus(
            state=state,
            status=v.get("status", ""),
            next=list(snapshot.next),
            model_calls=v.get("model_calls", 0),
            repairs=v.get("repairs", 0),
            steps=[AgentStep(**s) for s in v.get("steps", [])],
        )

    def result(self, task_id: str) -> AgentRun:
        v = self._snapshot(task_id).values
        return AgentRun(
            status=v["status"],
            summary=v.get("summary", ""),
            steps=[AgentStep(**s) for s in v.get("steps", [])],
            diff=v.get("diff", ""),
            check_before=CheckResult(**v["check_before"]),
            check_after=CheckResult(**v["check_after"]),
            usage=Usage(
                model_calls=v.get("model_calls", 0),
                prompt_tokens=v.get("prompt_tokens", 0),
                completion_tokens=v.get("completion_tokens", 0),
                peak_prompt_tokens=v.get("peak_prompt_tokens", 0),
                model_seconds=round(v.get("model_seconds", 0.0), 1),
            ),
            models_used=list(dict.fromkeys(v.get("models", []))),
            compactions=v.get("compactions", 0),
            memories_used=list(v.get("memory_ids", [])),
            seconds=round(v.get("seconds", 0.0), 1),
        )
