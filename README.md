# Autonoms

**An autonomous GitHub issue-resolution agent that you stay in control of.** It reads an issue, finds the relevant code, edits it, checks its own work in a sandbox, and prepares a pull request, but only publishes after a human has read the report and approved that exact version.

It runs on a **local LLM** (Ollama), uses **LangGraph** for a resumable workflow, and was built **one technology at a time** as a learning project: every stage has a write-up in [`docs/`](docs/00-roadmap.md) explaining what was added, why, and what went wrong.

> **Status:** all 12 stages are built and tested (281 tests). One step is not yet verified against the real service: **publishing the pull request to GitHub** (the access token used during development could read the repository but not write to it). The push/PR logic is tested against a local git repository and a mocked GitHub API. See [Honest status](#honest-status).

## How it works

```
GitHub issue ──► task (Postgres) ──► clone into an isolated workspace
                                          │
                       search (LlamaIndex: keyword + semantic)
                                          ▼
          ┌──────────── LangGraph workflow, checkpointed in Postgres ────────────┐
          │  baseline check ► agent (model picks tools) ► tools ► verify ► finish │
          │                         ▲                              │              │
          │                         └──── repair if edits broke something ◄───────┘
          └───────────────────────────────────────────────────────────────────────┘
                                          ▼
              review report + diff + fingerprint ──► you read it ──► you approve
                                          ▼
                                 branch + DRAFT pull request   (never merged by Autonoms)
```

- **Tool calling, not free-form edits.** The model can only search, read a file, replace an exact piece of text, run the check, or finish. It never gets a shell.
- **Checks run in a sandbox.** A Docker container with no network, a read-only filesystem, no secrets, and memory/time limits runs an arithmetic checker over the workspace.
- **Before and after.** The check runs before the agent starts and after it finishes, so "3 wrong → 2 wrong" is measured, and a fix that breaks something else is sent back for repair.
- **Approval is bound to content.** The review contains a fingerprint (a hash of the diff and text). Publishing requires quoting it, so approval can't apply to something you didn't read.
- **Resumable.** The workflow state is saved in Postgres after every step; kill the server mid-run and resume from the last checkpoint.

## Features

| Area | What it does |
|---|---|
| Issue intake | Fetches an issue and comments through the GitHub API, validated and rate-limit aware |
| Retrieval | LlamaIndex index per task: BM25 keyword, `bge-small` embeddings (CPU), and fused hybrid search |
| Agent | A small LLM loop with four tools, bounded steps, bounded repair rounds |
| Workflow | LangGraph graph with conditional repair routing and Postgres checkpoints |
| Memory | Scoped long-term notes (conventions, pitfalls, run history), secret filtering, relevance-only retrieval |
| Context | Older steps are collapsed into short notes when the conversation gets large |
| Gateway (optional) | LiteLLM proxy in Docker: model routing, fallback, rate limits, latency tracking |
| Review and PR | Report, risks, blockers and warnings, fingerprint-bound approval, draft PRs |
| Evaluation | Local per-run summary: completed, clean, in scope, cost and latency |
| Web UI | NiceGUI page: create a task, run with live progress, read the diff, approve |

## Technology

Python 3.13 · FastAPI · Pydantic · SQLAlchemy + PostgreSQL · Git · LlamaIndex · Ollama (tool calling) · Docker · LangGraph · LiteLLM · NiceGUI · pytest

## Quick start

**You need:** Python 3.13 with [`uv`](https://docs.astral.sh/uv/), a PostgreSQL server, [Ollama](https://ollama.com) with a tool-calling model, Docker, and a GitHub token.

```bash
git clone <this repository> && cd Autonoms
uv sync                               # installs everything; PyTorch is the CPU-only build

cp .env.example .env                  # then edit .env (see below). It is git-ignored: never commit it.
docker pull python:3.13.7-slim        # the sandbox image (about 180 MB)
ollama pull <a-tool-calling-model>    # the default name is qwen3.5:9b-8k; change it with OLLAMA_MODEL

uv run uvicorn autonoms.main:app      # terminal 1: the API (creates its own tables)
uv run python -m autonoms.ui          # terminal 2: the web page at http://127.0.0.1:8765
```

Create an empty database for it first (for example `CREATE DATABASE autonoms;`). The first search downloads the embedding model (about 130 MB) once.

### Configuration (`.env`)

| Variable | Needed for | Notes |
|---|---|---|
| `POSTGRES_CONNECTION_STRING` | everything | `postgresql://user:password@host:5432/autonoms` |
| `GITHUB_TOKEN` | reading issues (public repos work without it, with a low rate limit) and **publishing** | To publish, a fine-grained token needs *Contents: Read and write* and *Pull requests: Read and write* on the target repository |
| `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | the model | defaults: `http://localhost:11434`, `qwen3.5:9b-8k` |
| `LLM_BACKEND`, `LITELLM_MASTER_KEY` | optional gateway | `LLM_BACKEND=gateway`, then `./gateway/run.sh` (see [Stage 9](docs/stage-09-litellm-gateway.md)) |
| `API_URL`, `UI_PORT` | the web page | defaults: `http://127.0.0.1:8000`, `8765` |

Secrets are held as `SecretStr` (never printed in logs or `repr`), and nothing is sent to any tracing service.

### Using it

In the web page: create a task from an issue number → **Get the repository** → **Run the agent** → read the result and diff → on **Review & publish**, type your name and approve. Or use the API directly:

```bash
curl -X POST localhost:8000/tasks -H 'content-type: application/json' -d '{"owner":"you","repo":"repo","number":2}'
curl -X POST localhost:8000/tasks/$ID/workspace          # clone
curl -X POST localhost:8000/tasks/$ID/run                # run (a minute or so on a laptop GPU)
curl        localhost:8000/tasks/$ID/review              # the report + fingerprint; publishes nothing
curl -X POST localhost:8000/tasks/$ID/publish -H 'content-type: application/json' \
     -d '{"fingerprint":"<from /review>","approved_by":"Your Name"}'
```
Interactive API docs: <http://localhost:8000/docs>.

## The practice repository

[`seed-repo/`](seed-repo/) holds five short Markdown math notes (arithmetic, BODMAS, fractions, Pythagoras, area formulas) with one planted mistake each, and the text of five matching issues. A small model handles tasks like this far better than real code, which keeps the learning loop fast. Copy it to a repository of your own to try Autonoms without risking anything real.

## Tests

```bash
uv run pytest        # 281 tests
```
Most tests need no services: they use fakes and an in-memory or file-backed SQLite database. The sandbox tests use real Docker and skip themselves if the image is missing; the search tests use the real embedding model; the UI tests use NiceGUI's simulated user against the real API. Nothing in the suite touches your Postgres or GitHub.

## Project layout

```
src/autonoms/
  main.py          FastAPI app and endpoints          agent.py       tools + the agent loop
  workflow.py      LangGraph graph, checkpoints       context.py     what the model sees each call
  workspace.py     git clone/commit/push, safe paths  memory.py      long-term memory
  search.py        LlamaIndex retrieval               report.py      review text, risks, fingerprint
  sandbox.py       locked-down Docker runner          evaluation.py  per-run summary
  checker.py       runs checks/arithmetic_check.py    llm.py         Ollama and gateway clients
  ui/              the NiceGUI page                   github.py      GitHub API client
gateway/           LiteLLM config and start script    tests/         281 tests
docs/              one write-up per stage             seed-repo/     practice notes and issues
```

## Learning notes

Each stage added one technology, was tested, and was run against the real service before being written up. The write-ups include real numbers and the mistakes found along the way.

| | Stage | | Stage |
|---|---|---|---|
| 1 | [Issue intake: FastAPI, Pydantic, GitHub API](docs/stage-01-issue-intake.md) | 7 | [LangGraph with Postgres checkpoints](docs/stage-07-langgraph-checkpoints.md) |
| 2 | [PostgreSQL task records](docs/stage-02-postgres-tasks.md) | 8 | [Memory and context management](docs/stage-08-memory-and-context.md) |
| 3 | [Git workspaces](docs/stage-03-git-workspace.md) | 9 | [LiteLLM gateway and prompt reuse](docs/stage-09-litellm-gateway.md) |
| 4 | [LlamaIndex search](docs/stage-04-llamaindex-search.md) | 10 | [Human-approved pull requests](docs/stage-10-human-approved-pr.md) |
| 5 | [Tool-calling agent on a local model](docs/stage-05-tool-calling-agent.md) | 11 | [Evaluation (and why LangSmith was dropped)](docs/stage-11-evaluation.md) |
| 6 | [Docker sandbox and checks](docs/stage-06-docker-sandbox.md) | 12 | [Web interface](docs/stage-12-web-ui.md) |

The plan and its decisions are in the [roadmap](docs/00-roadmap.md); the original specification is [`docs/autonoms.md`](docs/autonoms.md). Two things in the spec were deliberately dropped: the Neo4j knowledge graph (the practice notes are independent files, so a graph had nothing to hold) and LangSmith tracing (usage is visible through the gateway and in each run result).

## Honest status

- **Real GitHub publish is unverified.** Everything up to the review and approval has been run against real services; the final push and pull request have only been tested locally and with a mocked API.
- **Small sample.** About eight real fixes, on tiny Markdown notes with obvious mistakes, with a 9B model. It shows the loop works, not that the model is reliable on real code.
- **The check is narrow.** It verifies arithmetic written like `12 / 4 = 3`. It cannot judge wording or formulas with letters, so a human must read the diff; the review says so.
- **Local, single user.** The web page has no login and binds to `127.0.0.1` only, because it can publish pull requests.
- **The sandbox is not a security boundary against a determined attacker.** Docker shares the host kernel; it is right for code we control, not for running untrusted third-party code.

## License

No license has been chosen yet. Until one is added, all rights are reserved by the author.
