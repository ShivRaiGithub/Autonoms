"""The UI's only way to reach Autonoms: the REST API, over HTTP.

The UI imports nothing from the backend. It is an ordinary client, so anything you can
click you could also do with curl, and the safety rules (approval by fingerprint, warnings,
blockers) live in one place: the server.
"""

from typing import Any

import httpx


class ApiError(Exception):
    """A request failed. `status` is the HTTP status, or 0 if the server could not be reached."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def explain(detail: Any) -> str:
    """The server's error body as one readable sentence."""
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):  # FastAPI validation errors: [{"loc": [...], "msg": "..."}]
        return "; ".join(f"{'.'.join(str(p) for p in e.get('loc', [])[1:])}: {e.get('msg', '')}".strip(": ") for e in detail)
    return str(detail)


class Api:
    def __init__(self, base_url: str, client: httpx.AsyncClient | None = None):
        self.base_url = base_url.rstrip("/")
        # A run takes minutes on a local model, so the default is "no timeout"; quick calls pass their own.
        self._client = client or httpx.AsyncClient(base_url=self.base_url, timeout=None)

    async def _call(self, method: str, path: str, *, timeout: float | None = 30, **kwargs) -> Any:
        try:
            response = await self._client.request(method, path, timeout=timeout, **kwargs)
        except httpx.HTTPError as exc:
            raise ApiError(0, f"Cannot reach the Autonoms API at {self.base_url} ({exc.__class__.__name__}). Is the server running?") from exc
        if response.status_code >= 400:
            try:
                detail = explain(response.json().get("detail", response.text))
            except ValueError:
                detail = response.text[:200]
            raise ApiError(response.status_code, detail)
        return None if response.status_code == 204 or not response.content else response.json()

    # --- tasks ---------------------------------------------------------------------------
    async def health(self) -> bool:
        try:
            return (await self._call("GET", "/health", timeout=3))["status"] == "ok"
        except ApiError:
            return False

    async def tasks(self) -> list[dict]:
        return await self._call("GET", "/tasks", timeout=8)

    async def get_task(self, task_id: str) -> dict:
        return await self._call("GET", f"/tasks/{task_id}")

    async def create_task(self, owner: str, repo: str, number: int) -> dict:
        return await self._call("POST", "/tasks", json={"owner": owner, "repo": repo, "number": number})

    async def delete_task(self, task_id: str) -> None:
        await self._call("DELETE", f"/tasks/{task_id}")

    async def create_workspace(self, task_id: str) -> dict:
        return await self._call("POST", f"/tasks/{task_id}/workspace", timeout=120)

    # --- running -------------------------------------------------------------------------
    async def run(self, task_id: str, max_steps: int, max_repairs: int, restart: bool) -> dict:
        body = {"max_steps": max_steps, "max_repairs": max_repairs, "restart": restart}
        return await self._call("POST", f"/tasks/{task_id}/run", timeout=None, json=body)

    async def resume(self, task_id: str) -> dict:
        return await self._call("POST", f"/tasks/{task_id}/resume", timeout=None)

    async def run_status(self, task_id: str) -> dict:
        return await self._call("GET", f"/tasks/{task_id}/run", timeout=10)

    # --- reviewing and publishing --------------------------------------------------------
    async def review(self, task_id: str) -> dict:
        return await self._call("GET", f"/tasks/{task_id}/review")

    async def evaluation(self, task_id: str) -> dict:
        return await self._call("GET", f"/tasks/{task_id}/evaluation")

    async def publish(self, task_id: str, fingerprint: str, approved_by: str, acknowledge_warnings: bool, draft: bool = True) -> dict:
        body = {"fingerprint": fingerprint, "approved_by": approved_by, "acknowledge_warnings": acknowledge_warnings, "draft": draft}
        return await self._call("POST", f"/tasks/{task_id}/publish", timeout=120, json=body)

    # --- memories ------------------------------------------------------------------------
    async def memories(self, repo: str) -> list[dict]:
        return await self._call("GET", "/memories", params={"repo": repo})

    async def add_memory(self, repo: str, kind: str, content: str) -> dict:
        return await self._call("POST", "/memories", json={"repo": repo, "kind": kind, "content": content})

    async def delete_memory(self, memory_id: str) -> None:
        await self._call("DELETE", f"/memories/{memory_id}")
