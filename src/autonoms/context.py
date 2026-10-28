"""Deciding what the model sees on each call.

The full history stays in the workflow state (for audit and resume). The model
gets a *view* of it: once the conversation is large, older steps are collapsed
into short notes and only the most recent rounds are kept word for word.

The notes are written by code, not by the model: a 9B model summarising its own
history costs a whole extra call and can invent or drop facts. What must survive
is decided here, explicitly:
  - what was EDITED (file, old text, new text)        - every ERROR, in full
  - the latest CHECK result and its wrong equations   - search hits (paths only)
What is dropped: file contents (the model can read the file again).
"""

KEEP_ROUNDS = 2  # a round = one model reply plus the tool results that followed it
HEADER = "Progress so far (older steps, shortened to save space; file contents are omitted, read a file again if you need it):"


def size(messages: list[dict]) -> int:
    return sum(len(m["content"]) for m in messages)


def _rounds(messages: list[dict]) -> list[list[dict]]:
    """Group messages into rounds, each starting at an assistant reply."""
    groups: list[list[dict]] = []
    for message in messages:
        if message["role"] == "assistant" or not groups:
            groups.append([message])
        else:
            groups[-1].append(message)
    return groups


def _brief_args(name: str, arguments: dict) -> str:
    if name == "replace_in_file":
        old, new = str(arguments.get("old_text", "")), str(arguments.get("new_text", ""))
        return f"{arguments.get('path')}: {old[:70]!r} -> {new[:70]!r}"
    if name in ("read_file",):
        return str(arguments.get("path"))
    if name == "search_code":
        return repr(str(arguments.get("query", ""))[:60])
    return ""


def _brief_result(name: str, text: str) -> str:
    if text.startswith("Error") or text.startswith("CHECK NOT RUN"):
        return text[:200]  # failures are never shortened away
    if name == "read_file":
        return f"read {len(text)} characters"
    if name == "search_code":
        paths = [line[1:-1] for line in text.splitlines() if line.startswith("[") and line.endswith("]")]
        return "hits: " + ", ".join(dict.fromkeys(paths)) if paths else "no matches"
    if name == "run_check":
        return " | ".join(text.splitlines()[:4])[:350]
    return text[:200]


def _summarise_round(group: list[dict]) -> list[str]:
    lines = []
    assistant, rest = group[0], group[1:]
    if assistant["role"] != "assistant":  # a stray message at the start; keep its text
        return [f"- {assistant['content'][:200]}"] + [f"- {m['content'][:200]}" for m in rest]
    results = [m for m in rest if m["role"] == "tool"]
    for call, result in zip(assistant.get("tool_calls", []), results):
        lines.append(f"- {call['name']}({_brief_args(call['name'], call['arguments'])}) -> {_brief_result(call['name'], result['content'])}")
    for message in rest:
        if message["role"] == "user":  # e.g. a repair request: this is an instruction, keep it
            lines.append(f"- you were told: {message['content'][:300]}")
    return lines


def select_context(messages: list[dict], soft_limit: int, keep_rounds: int = KEEP_ROUNDS) -> tuple[list[dict], bool]:
    """Return (the messages to send, whether older steps were collapsed). Never mutates its input."""
    if size(messages) <= soft_limit:
        return list(messages), False
    system, issue, rest = messages[0], messages[1], messages[2:]
    groups = _rounds(rest)
    if len(groups) <= keep_rounds:
        return list(messages), False  # nothing old enough to collapse
    old, recent = groups[:-keep_rounds], groups[-keep_rounds:]
    notes = [line for group in old for line in _summarise_round(group)]
    # The notes ride on the issue message, so roles keep alternating normally.
    issue_with_notes = {**issue, "content": issue["content"] + "\n\n" + HEADER + "\n" + "\n".join(notes)}
    return [system, issue_with_notes] + [m for group in recent for m in group], True
