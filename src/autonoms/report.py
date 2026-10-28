"""The review a human reads before anything is published.

Pure functions: a finished run goes in, a report with a fingerprint comes out.
The fingerprint is a hash of everything that would be published (title, text,
diff). Approving means quoting it, so approval can only ever apply to the exact
content that was shown: if the run, the diff or the text changes, the old
fingerprint no longer matches and publishing is refused.
"""

import hashlib
import re

from autonoms.models import AgentRun, Review, Risk, Task

MENTION = re.compile(r"@(?=\w)")


def clean(text: str, limit: int = 200) -> str:
    """Text from outside (issue titles, model output) made safe to put in a PR.

    An '@name' in a PR body notifies that person, so a hostile issue title could make
    Autonoms ping strangers. A zero-width space after '@' stops the mention.
    """
    one_line = " ".join(text.split())
    one_line = MENTION.sub("@​", one_line)
    return one_line[:limit] + ("..." if len(one_line) > limit else "")


def branch_name(task: Task) -> str:
    return f"autonoms/issue-{task.issue_number}-{task.id[:8]}"


def files_changed(diff: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"^diff --git a/(\S+)", diff, flags=re.M)))


def diffstat(diff: str) -> dict[str, tuple[int, int]]:
    stat: dict[str, list[int]] = {}
    current = None
    for line in diff.splitlines():
        match = re.match(r"diff --git a/(\S+)", line)
        if match:
            current = stat.setdefault(match.group(1), [0, 0])
        elif current is not None and line.startswith("+") and not line.startswith("+++"):
            current[0] += 1
        elif current is not None and line.startswith("-") and not line.startswith("---"):
            current[1] += 1
    return {name: (added, removed) for name, (added, removed) in stat.items()}


def unmentioned_files(task: Task, files: list[str]) -> list[str]:
    """Changed files whose names the issue never mentions (a rough sign of out-of-scope edits)."""
    issue_text = f"{task.issue.title} {task.issue.body}".lower()
    return [f for f in files if f.lower() not in issue_text]


def _risks(task: Task, run: AgentRun, files: list[str]) -> list[Risk]:
    risks: list[Risk] = []
    before = {(p.file, p.text) for p in run.check_before.problems}
    new = [p for p in run.check_after.problems if (p.file, p.text) not in before]
    if new:
        listing = "; ".join(f"{p.file} line {p.line}: {p.text} (should be {p.actual})" for p in new[:5])
        risks.append(Risk(level="warn", text=f"The edits made {len(new)} equation(s) wrong that were correct before: {listing}"))
    for name, check in (("before", run.check_before), ("after", run.check_after)):
        if check.status == "not_run":
            risks.append(Risk(level="warn", text=f"The automatic check did not run {name} the edits ({check.detail}), so nothing was verified automatically."))
    unmentioned = unmentioned_files(task, files)
    if unmentioned:
        risks.append(Risk(level="warn", text=f"The change touches file(s) the issue does not name: {', '.join(unmentioned)}. Check that this is intended."))
    old = [p for p in run.check_after.problems if (p.file, p.text) in before]
    if old:
        risks.append(Risk(level="info", text=f"{len(old)} equation(s) were already wrong before this change and still are (not part of this PR): " + "; ".join(f"{p.file}: {p.text}" for p in old[:5])))
    risks.append(Risk(level="info", text="The automatic check only verifies arithmetic written like 'a op b = c'. It cannot judge wording or formulas with letters, so the diff needs a human read."))
    return risks


def build_review(task: Task, run: AgentRun) -> Review:
    files = files_changed(run.diff)
    stat = diffstat(run.diff)
    blockers = []
    if not run.status.startswith("finished"):
        blockers.append(f"The run ended as '{run.status}', not as finished: {clean(run.summary, 150)}")
    if not files:
        blockers.append("The run changed no files.")
    risks = _risks(task, run, files)
    title = clean(f"Fix #{task.issue_number}: {task.issue_title}", 100)
    wrong_before, wrong_after = len(run.check_before.problems), len(run.check_after.problems)
    file_lines = "\n".join(f"- `{name}` (+{added} -{removed})" for name, (added, removed) in stat.items())
    risk_lines = "\n".join(f"- {'**Warning:** ' if r.level == 'warn' else ''}{clean(r.text, 400)}" for r in risks)
    models = ", ".join(run.models_used) or "unknown"
    body = (
        f"Closes #{task.issue_number}\n\n"
        f"## Problem\n{clean(task.issue_title, 150)}\n\n"
        f"## Fix\n{clean(run.summary, 400)}\n\n"
        f"Files changed:\n{file_lines}\n\n"
        f"## Verification\n"
        f"Automatic arithmetic check ({run.check_after.equations} equations): "
        f"{wrong_before} wrong before, {wrong_after} wrong after ({run.check_after.status}).\n\n"
        f"## Risks and limits\n{risk_lines}\n\n"
        f"## How this was made\n"
        f"Autonoms ran {run.usage.model_calls or len(run.steps)} model call(s) and {len(run.steps)} tool step(s) "
        f"with {models}, status `{run.status}`."
    )
    fingerprint = hashlib.sha256("\n".join([task.id, title, body, run.diff]).encode()).hexdigest()
    return Review(
        title=title, body=body, diff=run.diff, branch=branch_name(task), files=files, risks=risks, blockers=blockers,
        needs_acknowledgement=any(r.level == "warn" for r in risks), fingerprint=fingerprint,
        can_publish=not blockers,
    )  # fmt: skip


def final_body(review: Review, approved_by: str) -> str:
    """The text actually sent to GitHub: the reviewed body plus who approved it (not part of the fingerprint)."""
    return review.body + f"\n\n---\nPublished only after approval by **{clean(approved_by, 80)}**."
