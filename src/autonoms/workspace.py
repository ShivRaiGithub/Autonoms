"""Per-task workspaces: a cloned repository in its own directory, inspected with git.

Everything here shells out to the `git` command line. The agent never gets a
shell; it gets these narrow functions.
"""

import base64
import os
import shutil
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

from autonoms.models import Commit, RepoFile, RepoSummary

MAX_FILES_LISTED = 500
MAX_FILE_BYTES = 200_000
GIT_TIMEOUT_SECONDS = 60
MANIFESTS = {
    "pyproject.toml", "requirements.txt", "setup.py", "Pipfile",
    "package.json", "go.mod", "Cargo.toml", "pom.xml",
}  # fmt: skip


class GitError(Exception):
    def __init__(self, message: str, not_found: bool = False):
        super().__init__(message)
        self.not_found = not_found


class NoWorkspace(Exception):
    pass


class InvalidPath(Exception):
    pass


class FileTooLarge(Exception):
    pass


class BinaryFile(Exception):
    pass


class Workspaces:
    def __init__(self, root: Path, clone_base: str, token: str | None = None):
        self.root = Path(root).resolve()
        self.clone_base = clone_base.rstrip("/")
        self._token = token

    # --- paths ---------------------------------------------------------------

    def path(self, task_id: str) -> Path:
        # uuid.UUID() raises on anything that is not a UUID, so a task id can
        # never smuggle "../" into the directory name.
        return self.root / str(uuid.UUID(task_id))

    def exists(self, task_id: str) -> bool:
        return self.path(task_id).is_dir()

    def _require(self, task_id: str) -> Path:
        if not self.exists(task_id):
            raise NoWorkspace(task_id)
        return self.path(task_id)

    # --- running git -----------------------------------------------------------

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"  # fail instead of asking for a password
        env["GIT_CONFIG_NOSYSTEM"] = "1"
        env["GIT_CONFIG_GLOBAL"] = os.devnull  # ignore your personal git config
        # The token goes in an environment variable, not on the command line, so
        # it is not visible in `ps` and is not stored in the clone's .git/config.
        # It is only sent to github.com, never to another host.
        if self._token and self.clone_base.startswith("https://github.com"):
            basic = base64.b64encode(f"x-access-token:{self._token}".encode()).decode()
            env["GIT_CONFIG_COUNT"] = "1"
            env["GIT_CONFIG_KEY_0"] = "http.extraHeader"
            env["GIT_CONFIG_VALUE_0"] = f"Authorization: Basic {basic}"
        return env

    def _git(self, args: list[str], cwd: Path | None = None) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=cwd,
                env=self._env(),
                capture_output=True,
                text=True,
                timeout=GIT_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise GitError("git timed out") from exc
        except FileNotFoundError as exc:
            raise GitError("git is not installed") from exc
        if result.returncode != 0:
            stderr = result.stderr.lower()
            # GitHub says "Repository not found"; a local path says "does not
            # appear to be a git repository". Both mean: no such repo.
            not_found = any(
                phrase in stderr
                for phrase in ("not found", "does not exist", "does not appear to be a git repository")
            )
            # Don't pass stderr on: keep error text fixed and predictable.
            raise GitError("git command failed", not_found=not_found)
        return result.stdout

    # --- lifecycle --------------------------------------------------------------

    def create(self, task_id: str, owner: str, repo: str) -> bool:
        """Clone the repo for this task. Returns False if the workspace already existed."""
        dest = self.path(task_id)
        if dest.exists():
            return False
        self.root.mkdir(parents=True, exist_ok=True)
        url = f"{self.clone_base}/{owner}/{repo}.git"
        try:
            # --depth 50: only recent history; plenty for "what changed lately".
            self._git(["clone", "--quiet", "--depth", "50", url, str(dest)])
        except GitError:
            shutil.rmtree(dest, ignore_errors=True)  # remove a half-made clone
            raise
        return True

    def remove(self, task_id: str) -> bool:
        dest = self.path(task_id)
        if not dest.is_dir() or dest.parent != self.root:
            return False
        shutil.rmtree(dest)
        return True

    # --- inspection -------------------------------------------------------------

    def summary(self, task_id: str, commit_limit: int = 10) -> RepoSummary:
        ws = self._require(task_id)
        names = [n for n in self._git(["ls-files", "-z"], ws).split("\0") if n]
        files = []
        for name in names[:MAX_FILES_LISTED]:
            target = ws / name
            # A symlink's own size is meaningless and may point outside; report 0.
            size = 0 if target.is_symlink() else target.stat().st_size
            files.append(RepoFile(path=name, size=size))
        log = self._git(["log", f"-{commit_limit}", "--format=%H%x1f%an%x1f%aI%x1f%s"], ws)
        commits = []
        for line in log.splitlines():
            sha, author, date, subject = line.split("\x1f", 3)
            commits.append(Commit(sha=sha, author=author, date=datetime.fromisoformat(date), subject=subject))
        return RepoSummary(
            branch=self._git(["rev-parse", "--abbrev-ref", "HEAD"], ws).strip(),
            head=self._git(["rev-parse", "HEAD"], ws).strip(),
            file_count=len(names),
            files=files,
            files_truncated=len(names) > MAX_FILES_LISTED,
            manifests=sorted(n for n in names if Path(n).name in MANIFESTS),
            commits=commits,
        )

    def text_files(self, task_id: str) -> list[tuple[str, str]]:
        """(path, content) for every tracked file that is readable text.

        Binary, oversized and symlinked files are skipped. This reuses
        read_file, so it gets the same safety checks.
        """
        ws = self._require(task_id)
        names = [n for n in self._git(["ls-files", "-z"], ws).split("\0") if n]
        result = []
        for name in names:
            try:
                result.append((name, self.read_file(task_id, name)))
            except (BinaryFile, FileTooLarge, InvalidPath):
                continue
        return result

    def read_file(self, task_id: str, relative_path: str) -> str:
        ws = self._require(task_id)
        if not relative_path or "\0" in relative_path or Path(relative_path).is_absolute():
            raise InvalidPath(relative_path)
        # resolve() follows symlinks and removes "..", so the check below sees
        # where the file really is, even if the repo contains a link to /etc.
        target = (ws / relative_path).resolve()
        inside = target.is_relative_to(ws.resolve())
        if not inside or target.relative_to(ws.resolve()).parts[:1] == (".git",):
            raise InvalidPath(relative_path)
        if not target.is_file():
            raise InvalidPath(relative_path)
        if target.stat().st_size > MAX_FILE_BYTES:
            raise FileTooLarge(relative_path)
        try:
            return target.read_bytes().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise BinaryFile(relative_path) from exc
