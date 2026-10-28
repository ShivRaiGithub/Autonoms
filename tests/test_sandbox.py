"""Tests that use real Docker. Skipped when Docker or the image is missing."""

import subprocess
import time
from pathlib import Path

import pytest

from autonoms.checker import Checker
from autonoms.config import Settings
from autonoms.sandbox import Sandbox

IMAGE = Settings().sandbox_image


def docker_ready() -> bool:
    try:
        return subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


pytestmark = pytest.mark.skipif(not docker_ready(), reason=f"Docker or image {IMAGE} not available")


@pytest.fixture
def box():
    return Sandbox(IMAGE, timeout_seconds=15, memory="256m")


@pytest.fixture
def workdir(tmp_path):
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / "a.md").write_text("12 / 4 = 4\n")
    return tmp_path / "work"


def script(tmp_path, code):
    path = tmp_path / "probe.py"
    path.write_text(code)
    return path


def test_network_is_blocked(box, workdir, tmp_path):
    result = box.run(workdir, script(tmp_path, """
import socket
try:
    socket.create_connection(("1.1.1.1", 80), timeout=3)
    print("REACHED")
except OSError as e:
    print("BLOCKED")
"""))  # fmt: skip
    assert result.stdout.strip() == "BLOCKED"


def test_filesystem_is_read_only_except_tmp(box, workdir, tmp_path):
    result = box.run(workdir, script(tmp_path, """
for path in ("/work/new.txt", "/work/a.md", "/etc/new.txt", "/new.txt", "/tmp/ok.txt"):
    try:
        open(path, "w").write("x")
        print(path, "WRITTEN")
    except OSError:
        print(path, "BLOCKED")
"""))  # fmt: skip
    lines = dict(line.split() for line in result.stdout.splitlines())
    assert lines == {
        "/work/new.txt": "BLOCKED", "/work/a.md": "BLOCKED", "/etc/new.txt": "BLOCKED",
        "/new.txt": "BLOCKED", "/tmp/ok.txt": "WRITTEN",
    }  # fmt: skip
    assert (workdir / "a.md").read_text() == "12 / 4 = 4\n"  # and the host file really is untouched
    assert not (workdir / "new.txt").exists()


def test_runs_as_nobody_and_gets_no_host_secrets(box, workdir, tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_should_never_appear")
    result = box.run(workdir, script(tmp_path, "import os; print(os.getuid()); print(dict(os.environ))"))
    assert result.stdout.splitlines()[0] == "65534"
    assert "ghp_should_never_appear" not in result.stdout


def test_memory_limit_kills_the_script(box, workdir, tmp_path):
    result = box.run(workdir, script(tmp_path, "x = bytearray(900 * 1024 * 1024); print('survived')"))
    assert result.exit_code == 137 and "survived" not in result.stdout  # 137 = killed (out of memory)


def test_timeout_kills_the_container_too(workdir, tmp_path):
    short = Sandbox(IMAGE, timeout_seconds=2)
    started = time.monotonic()
    result = short.run(workdir, script(tmp_path, "import time; time.sleep(60)"))
    assert result.timed_out and time.monotonic() - started < 15
    leftovers = subprocess.run(["docker", "ps", "-q", "--filter", "name=autonoms-"], capture_output=True, text=True)
    assert leftovers.stdout.strip() == ""  # no container left running


def test_missing_image_is_an_error_not_a_crash(workdir, tmp_path):
    result = Sandbox("autonoms-missing.invalid/none:0").run(workdir, script(tmp_path, "print(1)"))
    assert result.error and result.exit_code is None


# --- the real check, through the sandbox ------------------------------------------


class Workspaces:
    """Just enough of the real class for Checker."""

    def __init__(self, path: Path):
        self._path = path

    def exists(self, task_id):
        return True

    def path(self, task_id):
        return self._path


def test_check_fails_then_passes_after_the_fix(box, workdir):
    checker = Checker(box, Workspaces(workdir))
    before = checker.run("t")
    assert before.status == "failed" and before.equations == 1
    assert before.problems[0].text == "12 / 4 = 4" and before.problems[0].actual == "3"
    (workdir / "a.md").write_text("12 / 4 = 3\n")
    after = checker.run("t")
    assert after.status == "passed" and after.problems == []


def test_check_with_nothing_to_check_says_zero_equations(box, tmp_path):
    (tmp_path / "x.md").write_text("just words\n")
    result = Checker(box, Workspaces(tmp_path)).run("t")
    assert result.status == "passed" and result.equations == 0


def test_check_that_cannot_run_is_not_run_never_passed(workdir):
    checker = Checker(Sandbox("autonoms-missing.invalid/none:0"), Workspaces(workdir))
    result = checker.run("t")
    assert result.status == "not_run" and result.detail
