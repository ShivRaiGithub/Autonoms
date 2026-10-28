"""Run a script in a locked-down Docker container against a task's workspace.

The container gets: no network, a read-only root filesystem, the workspace
mounted read-only, no Linux capabilities, a non-root user, and limits on
memory, CPU, processes and time. Nothing from your environment (tokens!) is
passed in. The script can read the files and print; it cannot change anything.
"""

import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SandboxResult:
    exit_code: int | None  # None if it never ran or was killed on timeout
    stdout: str
    stderr: str
    timed_out: bool
    seconds: float
    error: str = ""  # set when Docker itself was unavailable


class Sandbox:
    def __init__(self, image: str, timeout_seconds: int = 20, memory: str = "256m"):
        self.image = image
        self.timeout_seconds = timeout_seconds
        self.memory = memory

    def command(self, name: str, workspace: Path, script: Path) -> list[str]:
        return [
            "docker", "run", "--rm", "--name", name,
            "--network", "none",
            "--read-only", "--tmpfs", "/tmp:size=16m,noexec",
            "--memory", self.memory, "--memory-swap", self.memory,  # swap = memory: no swapping
            "--cpus", "1", "--pids-limit", "64",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", "65534:65534",  # "nobody"
            "-v", f"{workspace.resolve()}:/work:ro",
            "-v", f"{script.resolve()}:/checker/check.py:ro",
            "-w", "/work",
            self.image, "python", "/checker/check.py", "/work",
        ]  # fmt: skip

    def run(self, workspace: Path, script: Path) -> SandboxResult:
        name = f"autonoms-{uuid.uuid4().hex[:12]}"
        started = time.monotonic()
        try:
            done = subprocess.run(
                self.command(name, workspace, script),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            # Killing the `docker` client does NOT stop the container; do it explicitly.
            subprocess.run(["docker", "kill", name], capture_output=True, timeout=15)
            return SandboxResult(None, "", "", True, round(time.monotonic() - started, 1))
        except (FileNotFoundError, OSError) as exc:
            return SandboxResult(None, "", "", False, 0, error=f"Docker is not available ({exc.__class__.__name__})")
        # "docker run" itself failing (daemon down, image missing) is exit 125-127
        if done.returncode in (125, 126, 127):
            return SandboxResult(None, "", "", False, 0, error="Docker could not start the container")
        return SandboxResult(
            done.returncode, done.stdout[-20000:], done.stderr[-4000:], False, round(time.monotonic() - started, 1)
        )
