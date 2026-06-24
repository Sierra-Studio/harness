# Harness

A pluggable, multi-tenant LLM agent harness implementing the design in
`plano-implementacao.html`: **Memory, Skills, MCP, Tools, Loop**, an **OpenRouter**
provider, and **token-level observability** — all persisted in **Postgres**.

Every component sits behind an interface, so the provider, the sandbox, and the
persistence backend are all swappable.

## What it does

| Component | Behaviour |
|---|---|
| **Loop** | `perceive → build context → call model → run tools → repeat`, with a per-session **token-budget guard** that stops and returns the partial reply. |
| **Memory** | Window = `system prompt + chained summary + active turns`. Budget = `context_window − system_prompt − response_reserve`. On overflow: keep the last 10% verbatim, fold the rest (plus the previous summary) into a new **chained** summary. Folded turns stay in Postgres (`in_window=false`). |
| **Checkpoints** | Every 20 **user** turns, the subject is classified in a few words. |
| **Tools** | Built-ins always in the prompt — `SearchTools`, `GetTools`, `GetSkills`, `Bash`. **Index Tools** (from MCP) live in `tool_index` and are reached on demand via **keyword (full-text) search in Postgres** — **O(1)** prompt cost regardless of how many MCP tools exist. |
| **Skills** | Owned per user. Every 10 closed sessions, an induction pass mines recurring requests into new skills (deduped by embedding). |
| **Provider** | OpenRouter: model context window from `/models`, chat completions with tool calling, real `usage` recorded per turn. |
| **Observability** | One `step_logs` row per loop step; `tokens_in/out` on every model turn; live totals on the session. |
| **Bash sandbox** | Runs behind a pluggable `SandboxBackend`. Ships with a local-subprocess impl (one workdir per session). Swap for gVisor/Firecracker/K8s for real isolation. |

## Quickstart (uv)

Requires [uv](https://docs.astral.sh/uv/). One-time setup:
```bash
uv sync --extra dev    # creates .venv and installs deps from pyproject.toml
```

### 1. Offline demo (no DB, no API key)
Runs the whole harness with an in-memory repo and a scripted fake provider:
```bash
uv run python demo.py
```

### 2. Tests
```bash
uv run pytest -q
```

### 3. Real run (OpenRouter + Postgres)
```bash
cp .env.example .env          # set OPENROUTER_API_KEY (needs credits)
docker compose up -d          # Postgres + pgvector
uv run harness init-db        # apply schema.sql
uv run harness chat           # interactive session
```
With `DATABASE_URL` unset the harness uses the in-memory repo; with
`OPENROUTER_API_KEY` unset it uses the offline `FakeProvider`. Mix and match, e.g.
force offline + in-memory:
```bash
DATABASE_URL="" OPENROUTER_API_KEY="" uv run harness chat
```

> Not using uv? The core still runs with plain `python` (e.g. `python -m harness.cli chat`);
> install deps however you like — they're declared in `pyproject.toml`.

## Configuration

All via env (see `.env.example`): model, budgets, `RESPONSE_RESERVE_TOKENS`,
`MAX_STEPS`, `CHECKPOINT_EVERY_USER_TURNS`, `SKILL_INDUCTION_EVERY_SESSIONS`,
`SUMMARY_KEEP_RATIO`, `EMBEDDING_*`.

## Project layout

```
harness/
  config.py        # env-driven settings (+ tiny .env loader)
  tokenizer.py     # tiktoken with heuristic fallback
  embeddings.py    # OpenAI-compatible embeddings + local fallback
  models.py        # dataclasses mirroring the schema
  repository.py    # Repository contract + InMemory + Postgres(pgvector)
  provider.py      # Provider contract + OpenRouter + FakeProvider
  sandbox.py       # SandboxBackend contract + local-subprocess impl
  mcp_client.py    # MCP clients: stdio + Streamable-HTTP, + tool ingestion
  memory.py        # budget, build_window, chained summarize, checkpoints
  tools.py         # built-ins + Index Tool dispatch
  skills.py        # induction (every N sessions, deduped)
  observer.py      # step logging + latency
  loop.py          # the agent loop + token-budget guard
  app.py           # Harness facade — wire/swap components here
  cli.py           # init-db, chat
schema.sql         # Postgres + pgvector DDL
docker-compose.yml # pgvector/pgvector:pg16
pyproject.toml     # project metadata + deps (uv); `harness` console script
uv.lock            # pinned dependency lockfile
demo.py            # offline end-to-end demo
tests/test_core.py
```

## Swapping pieces

`app.py` is the single wiring point. Examples:
```python
Harness(cfg, provider=MyProvider(...))          # different LLM gateway
Harness(cfg, sandbox=FirecrackerSandbox(...))   # kernel-isolated Bash
Harness(cfg, repo=PostgresRepository(dsn))       # forced backend
```

## Connecting MCP servers

Two transports, one-line helpers on `Harness` — both index the server's tools
into `tool_index` and enable dispatch:

```python
h = Harness()

# local stdio server (subprocess)
h.add_mcp_stdio(["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                name="fs")

# remote Streamable-HTTP server (pass headers for auth)
h.add_mcp_http("https://fellow.app/mcp", name="fellow",
               headers={"Authorization": f"Bearer {token}"})
```
The model finds these tools through `SearchTools` (keyword/full-text search over
`tool_index`), never via the prompt. See `examples/add_fellow_mcp.py`.

**From the CLI**, remote HTTP servers are auto-connected from the env — no code:
```bash
# .env
MCP_HTTP_SERVERS=fellow=https://fellow.app/mcp   # comma-separated name=url
MCP_FELLOW_TOKEN=...                              # MCP_<NAME>_TOKEN -> bearer auth
```
`uv run harness chat` connects each one at startup (failures are reported and
skipped) and disconnects them on exit.

## Status / notes

- Core logic verified by `tests/test_core.py` (12/12) and `demo.py`.
- Validated end-to-end against **real Postgres + pgvector** (repository, vector
  `SearchTools`, summarization, checkpoints, induction, token accounting).
- OpenRouter integration verified up to billing: the harness authenticates,
  pulls the model context window, and forms valid chat requests. A live
  completion needs account credits (a `402 Payment Required` means no credit).
- The bundled `LocalSubprocessSandbox` is **not** isolated — replace it with a
  kernel-isolated backend before exposing untrusted multi-tenant Bash. See the
  companion architecture guide (`guia-harness.html`).
