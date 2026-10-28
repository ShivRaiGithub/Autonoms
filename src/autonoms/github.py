"""A small GitHub REST API client that turns issues into our own models."""

import httpx

from autonoms.models import Comment, Issue


class GitHubError(Exception):
    """Base class: anything that went wrong talking to GitHub."""


class IssueNotFound(GitHubError):
    pass


class NotAnIssue(GitHubError):
    """GitHub's issues API also returns pull requests; we only want real issues."""


class RateLimited(GitHubError):
    pass


class PullRequestExists(GitHubError):
    pass


class GitHubClient:
    def __init__(self, client: httpx.Client):
        # The httpx.Client is passed in so tests can give it a fake transport.
        self._client = client

    @classmethod
    def create(cls, base_url: str, token: str | None) -> "GitHubClient":
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return cls(httpx.Client(base_url=base_url, headers=headers, timeout=15))

    def close(self) -> None:
        self._client.close()

    def _get(self, path: str, **params) -> httpx.Response:
        try:
            response = self._client.get(path, params=params)
        except httpx.HTTPError as exc:
            raise GitHubError(f"Could not reach GitHub: {exc.__class__.__name__}") from exc
        if response.status_code == 404:
            raise IssueNotFound(path)
        if response.status_code in (403, 429) and response.headers.get("x-ratelimit-remaining") == "0":
            raise RateLimited("GitHub rate limit exceeded")
        if response.status_code >= 400:
            raise GitHubError(f"GitHub returned {response.status_code}")
        return response

    def get_issue(self, owner: str, repo: str, number: int) -> Issue:
        base = f"/repos/{owner}/{repo}/issues/{number}"
        data = self._get(base).json()
        if "pull_request" in data:
            raise NotAnIssue(f"{owner}/{repo}#{number} is a pull request")
        # Only the first 100 comments: enough for a first version (see stage doc).
        raw_comments = self._get(f"{base}/comments", per_page=100).json()
        return Issue(
            owner=owner,
            repo=repo,
            number=data["number"],
            title=data["title"],
            body=data["body"] or "",
            state=data["state"],
            author=data["user"]["login"],
            labels=[label["name"] for label in data["labels"]],
            url=data["html_url"],
            created_at=data["created_at"],
            comments=[
                Comment(author=c["user"]["login"], body=c["body"], created_at=c["created_at"])
                for c in raw_comments
            ],
        )

    def create_pull_request(self, owner: str, repo: str, head: str, base: str, title: str, body: str, draft: bool = True) -> tuple[int, str]:
        """Open a pull request; returns (number, url)."""
        try:
            response = self._client.post(
                f"/repos/{owner}/{repo}/pulls",
                json={"title": title, "head": head, "base": base, "body": body, "draft": draft},
            )
        except httpx.HTTPError as exc:
            raise GitHubError(f"Could not reach GitHub: {exc.__class__.__name__}") from exc
        if response.status_code in (401, 403, 404):
            # 404 is what GitHub answers when a token may not see or write the repository
            raise GitHubError("GitHub refused the request: the token may lack 'Pull requests: write' permission")
        if response.status_code == 422:
            if "already exists" in response.text:
                raise PullRequestExists(head)
            raise GitHubError("GitHub rejected the pull request (is there a difference between the branches?)")
        if response.status_code >= 400:
            raise GitHubError(f"GitHub returned {response.status_code}")
        data = response.json()
        return data["number"], data["html_url"]

    def find_pull_request(self, owner: str, repo: str, head: str) -> tuple[int, str] | None:
        found = self._get(f"/repos/{owner}/{repo}/pulls", head=f"{owner}:{head}", state="all").json()
        return (found[0]["number"], found[0]["html_url"]) if found else None
