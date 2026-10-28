"""Per-task workspaces: a cloned repository in its own directory, inspected with git.

Everything here shells out to the `git` command line. The agent never gets a
shell; it gets these narrow functions.
"""

import base64
import os
import re
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
    def __init__(self, message: str, not_found: bool = False, denied: bool = False):
        super().__init__(message)
        self.not_found = not_found
        self.denied = denied  # the remote refused our credentials or permissions


BRANCH = re.compile(r"^autonoms/issue-\d+-[0-9a-f]{8}$")  # the only branch names we ever create or push


class NoWorkspace(Exception):
    pass


class InvalidPath(Exception):
    pass


class FileTooLarge(Exception):
    pass


class BinaryFile(Exception):
    pass


class EditError(Exception):
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
            denied = "permission to" in stderr or "denied" in stderr or "returned error: 403" in stderr
            # Don't pass stderr on: keep error text fixed and predictable.
            raise GitError("git command failed", not_found=not_found, denied=denied)
        return result.stdout

    # --- lifecycle --------------------------------------------------------------

    def create(self, task_id: str, owner: str, repo: str) -> bool:
        """Clone the repo for this task. Returns False if the workspace already existed.

        The clone is made in a temporary folder and renamed into place when complete, so
        `exists()` can never be true for a half-finished clone (a request arriving mid-clone
        would otherwise see a workspace with no files in it).
        """
        dest = self.path(task_id)
        if dest.exists():
            return False
        self.root.mkdir(parents=True, exist_ok=True)
        partial = self.root / f".{dest.name}.partial"
        shutil.rmtree(partial, ignore_errors=True)  # left over from a crash
        url = f"{self.clone_base}/{owner}/{repo}.git"
        try:
            # --depth 50: only recent history; plenty for "what changed lately".
            self._git(["clone", "--quiet", "--depth", "50", url, str(partial)])
            partial.rename(dest)
        except GitError:
            shutil.rmtree(partial, ignore_errors=True)  # remove a half-made clone
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

    def _safe_target(self, task_id: str, relative_path: str) -> Path:
        """Turn a path from outside into a real file inside the workspace, or raise."""
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
        return target

    def read_file(self, task_id: str, relative_path: str) -> str:
        target = self._safe_target(task_id, relative_path)
        if target.stat().st_size > MAX_FILE_BYTES:
            raise FileTooLarge(relative_path)
        try:
            return target.read_bytes().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise BinaryFile(relative_path) from exc

    def replace_in_file(self, task_id: str, relative_path: str, old: str, new: str) -> None:
        """Replace one exact piece of text. Refuses if it is missing or ambiguous."""
        text = self.read_file(task_id, relative_path)  # same checks as reading
        if not old:
            raise EditError("old_text must not be empty")
        count = text.count(old)
        if count == 0:
            raise EditError("old_text was not found in the file")
        if count > 1:
            raise EditError(f"old_text appears {count} times; include more surrounding text")
        self._safe_target(task_id, relative_path).write_text(text.replace(old, new, 1), encoding="utf-8")

    def diff(self, task_id: str) -> str:
        """The changes made in the workspace so far, as a unified diff."""
        return self._git(["diff", "--no-color"], self._require(task_id))

    # --- publishing --------------------------------------------------------------

    def current_branch(self, task_id: str) -> str:
        return self._git(["rev-parse", "--abbrev-ref", "HEAD"], self._require(task_id)).strip()

    def base_branch(self, task_id: str) -> str:
        """The branch the clone was made from (the repository's default branch)."""
        ref = self._git(["symbolic-ref", "--short", "refs/remotes/origin/HEAD"], self._require(task_id)).strip()
        return ref.removeprefix("origin/")

    def commit_to_branch(self, task_id: str, branch: str, message: str, author: tuple[str, str]) -> str:
        """Commit the workspace's edits to a new branch; returns the commit id.

        Only files git already tracks are included (`add -u`), so a stray new file can never be
        published by accident. Safe to call again: if we are already on the branch, nothing is redone.
        """
        if not BRANCH.match(branch):
            raise InvalidPath(branch)  # the name goes to git: only our own pattern is allowed
        ws = self._require(task_id)
        if self.current_branch(task_id) != branch:
            self._git(["checkout", "-b", branch], ws)
            self._git(["add", "-u"], ws)
            name, email = author
            self._git(["-c", f"user.name={name}", "-c", f"user.email={email}", "commit", "-m", message], ws)
        return self._git(["rev-parse", "HEAD"], ws).strip()

    def push_branch(self, task_id: str, branch: str) -> None:
        """Push the branch. Never forced: if the remote branch exists with other history, this fails."""
        if not BRANCH.match(branch):
            raise InvalidPath(branch)
        self._git(["push", "origin", f"HEAD:refs/heads/{branch}"], self._require(task_id))
