"""FastAPI application: the HTTP interface to Autonoms."""

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Path

from autonoms.config import get_settings
from autonoms.github import GitHubClient, GitHubError, IssueNotFound, NotAnIssue, RateLimited
from autonoms.models import Issue

app = FastAPI(title="Autonoms")

# GitHub names only allow these characters. Validating here also blocks
# path tricks like "..", because owner/repo go straight into a URL path.
NAME = r"^[A-Za-z0-9_.-]+$"


def get_github() -> Iterator[GitHubClient]:
    settings = get_settings()
    token = settings.github_token.get_secret_value() if settings.github_token else None
    client = GitHubClient.create(settings.github_api_url, token)
    try:
        yield client
    finally:
        client.close()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/issues/{owner}/{repo}/{number}", response_model=Issue)
def read_issue(
    owner: Annotated[str, Path(pattern=NAME, max_length=100)],
    repo: Annotated[str, Path(pattern=NAME, max_length=100)],
    number: Annotated[int, Path(ge=1)],
    github: Annotated[GitHubClient, Depends(get_github)],
) -> Issue:
    try:
        return github.get_issue(owner, repo, number)
    except IssueNotFound:
        raise HTTPException(404, "Issue not found")
    except NotAnIssue:
        raise HTTPException(422, "That number is a pull request, not an issue")
    except RateLimited:
        raise HTTPException(429, "GitHub rate limit exceeded")
    except GitHubError as exc:
        raise HTTPException(502, str(exc))
