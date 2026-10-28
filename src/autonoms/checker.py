"""Runs the arithmetic check for a task, inside the sandbox, and interprets the result."""

import json
from pathlib import Path

from autonoms.models import CheckProblem, CheckResult
from autonoms.sandbox import Sandbox
from autonoms.workspace import NoWorkspace, Workspaces

SCRIPT = Path(__file__).parent / "checks" / "arithmetic_check.py"


class Checker:
    def __init__(self, sandbox: Sandbox, workspaces: Workspaces):
        self._sandbox = sandbox
        self._workspaces = workspaces

    def run(self, task_id: str) -> CheckResult:
        if not self._workspaces.exists(task_id):
            raise NoWorkspace(task_id)
        result = self._sandbox.run(self._workspaces.path(task_id), SCRIPT)
        if result.error:
            return CheckResult(status="not_run", detail=result.error)
        if result.timed_out:
            return CheckResult(status="not_run", detail="The check timed out and was killed", seconds=result.seconds)
        if result.exit_code not in (0, 1):
            # e.g. 137 = killed for using too much memory
            return CheckResult(status="not_run", detail=f"The check crashed (exit code {result.exit_code})", seconds=result.seconds)
        try:
            data = json.loads(result.stdout)
            problems = [CheckProblem(**p) for p in data["problems"]]
            equations = int(data["equations"])
        except (ValueError, KeyError, TypeError):
            return CheckResult(status="not_run", detail="The check printed output we could not read", seconds=result.seconds)
        return CheckResult(
            status="failed" if problems else "passed",
            equations=equations,
            problems=problems,
            seconds=result.seconds,
        )
