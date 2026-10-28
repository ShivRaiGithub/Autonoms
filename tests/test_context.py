from autonoms.context import HEADER, KEEP_ROUNDS, select_context, size

SYSTEM = {"role": "system", "content": "rules"}
ISSUE = {"role": "user", "content": "Issue #5: fix the circle"}


def call(name, **args):
    return {"role": "assistant", "content": "", "tool_calls": [{"name": name, "arguments": args}]}


def tool(name, text):
    return {"role": "tool", "tool_name": name, "content": text}


BIG_FILE = "line of a file\n" * 300  # 4,500 characters


def history(rounds=5):
    messages = [SYSTEM, ISSUE]
    for i in range(rounds):
        messages += [call("read_file", path=f"note{i}.md"), tool("read_file", BIG_FILE)]
    return messages


def test_small_conversations_are_untouched():
    messages = history(1)
    view, compacted = select_context(messages, soft_limit=10_000)
    assert view == messages and not compacted


def test_large_conversations_collapse_older_rounds_and_keep_recent_ones_verbatim():
    messages = history(5)
    view, compacted = select_context(messages, soft_limit=5_000)
    assert compacted and size(view) < size(messages) / 2
    assert view[0] == SYSTEM
    assert view[-4:] == messages[-4:]  # the last 2 rounds (assistant + tool each), word for word
    assert len(view) == 2 + 2 * KEEP_ROUNDS


def test_file_contents_are_dropped_but_which_files_were_read_is_kept():
    view, _ = select_context(history(5), soft_limit=5_000)
    notes = view[1]["content"]
    assert HEADER in notes and "line of a file" not in notes
    assert "read_file(note0.md) -> read 4500 characters" in notes


def test_edits_are_remembered_with_their_old_and_new_text():
    messages = [SYSTEM, ISSUE, call("replace_in_file", path="formulas.md", old_text="radius x 2", new_text="radius x radius"),
                tool("replace_in_file", "OK: edited formulas.md")] + history(4)[2:]  # fmt: skip
    notes = select_context(messages, soft_limit=5_000)[0][1]["content"]
    assert "replace_in_file(formulas.md: 'radius x 2' -> 'radius x radius') -> OK: edited formulas.md" in notes


def test_errors_and_failed_checks_are_never_shortened_away():
    error = "Error: old_text appears 2 times; include more surrounding text"
    check = "CHECK FAILED: 1 wrong of 11 equations checked.\n- bodmas.md line 11: 2 + 12 = 20 (the result should be 14)"
    messages = [SYSTEM, ISSUE, call("replace_in_file", path="a.md", old_text="x", new_text="y"), tool("replace_in_file", error),
                call("run_check"), tool("run_check", check)] + history(4)[2:]  # fmt: skip
    notes = select_context(messages, soft_limit=5_000)[0][1]["content"]
    assert error in notes
    assert "bodmas.md line 11: 2 + 12 = 20" in notes


def test_a_repair_request_is_kept_because_it_is_an_instruction():
    repair = {"role": "user", "content": "The check found equations that were correct before your edits: bodmas.md line 11"}
    messages = [SYSTEM, ISSUE, call("finish", summary="done"), tool("finish", "done"), repair] + history(4)[2:]
    notes = select_context(messages, soft_limit=5_000)[0][1]["content"]
    assert "you were told: The check found equations" in notes


def test_the_input_is_never_modified():
    messages = history(5)
    snapshot = [dict(m) for m in messages]
    select_context(messages, soft_limit=5_000)
    assert messages == snapshot


def test_roles_stay_valid_for_the_model():
    view, _ = select_context(history(6), soft_limit=5_000)
    roles = [m["role"] for m in view]
    assert roles[:2] == ["system", "user"]
    assert all(r != "tool" or roles[i - 1] in ("assistant", "tool") for i, r in enumerate(roles))
    assert "user" not in roles[2:]  # the notes ride on the issue message: no second user message


def test_nothing_to_collapse_when_there_are_only_recent_rounds():
    messages = history(KEEP_ROUNDS)
    view, compacted = select_context(messages, soft_limit=100)
    assert view == messages and not compacted
