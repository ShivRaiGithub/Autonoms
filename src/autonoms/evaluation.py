"""Basic evaluation of a finished run: did it complete, was it correct as far as we can tell, and what did it cost?

Everything here is computed locally from the finished run's result; nothing is sent anywhere.
"""

from pydantic import BaseModel

from autonoms.models import AgentRun, Task
from autonoms.report import files_changed, unmentioned_files


class Evaluation(BaseModel):
    # outcome
    completed: bool  # the model finished and changed at least one file
    clean: bool  # ...and broke nothing that was correct before
    equations_fixed: int | None  # None when the check could not run: unknown, not zero
    equations_broken: int | None
    in_scope: bool  # every changed file is named in the issue (a rough heuristic)
    files_changed: int
    # cost and speed
    seconds: float
    model_seconds: float
    model_calls: int
    tool_steps: int
    prompt_tokens: int
    completion_tokens: int
    compactions: int
    fallbacks: int


def evaluate(task: Task, run: AgentRun) -> Evaluation:
    files = files_changed(run.diff)
    checked = run.check_before.status != "not_run" and run.check_after.status != "not_run"
    before = {(p.file, p.text) for p in run.check_before.problems}
    after = {(p.file, p.text) for p in run.check_after.problems}
    return Evaluation(
        completed=run.status.startswith("finished") and bool(files),
        clean=run.status == "finished",
        equations_fixed=len(before - after) if checked else None,
        equations_broken=len(after - before) if checked else None,
        in_scope=not unmentioned_files(task, files),
        files_changed=len(files),
        seconds=run.seconds,
        model_seconds=run.usage.model_seconds,
        model_calls=run.usage.model_calls,
        tool_steps=len(run.steps),
        prompt_tokens=run.usage.prompt_tokens,
        completion_tokens=run.usage.completion_tokens,
        compactions=run.compactions,
        fallbacks=run.usage.fallbacks,
    )

