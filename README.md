# Harness

A pluggable, multi-tenant LLM agent harness: **Memory, Skills, MCP, Tools, Loop**,
an **OpenRouter** provider, and **token-level observability** — all persisted in
**Postgres**.

Every component sits behind an interface, so the provider, the sandbox, and the
persistence backend are all swappable.

## What it does

| Component | Behaviour |
|---|---|
| **Persona** | Layered system prompt: a customizable persona (`PERSONA.md`, inspired by Hermes' SOUL) + built-in tool guidance. Falls back to a default identity when no `PERSONA.md` is set. |
| **Loop** | `perceive → build context → call model → run tools → repeat`, with a per-session **token-budget guard** that stops and returns the partial reply. |
| **Memory** | Window = `system prompt + chained summary + active turns`. Budget = `context_window − system_prompt − response_reserve`. On overflow: keep the last 10% verbatim, fold the rest (plus the previous summary) into a new **chained** summary. Folded turns stay in Postgres (`in_window=false`). |
| **Checkpoints** | Every 20 **user** turns, the subject is classified in a few words. |
| **Tools** | Built-ins always in the prompt — `SearchTools`, `GetTools`, `GetSkills`, `Bash`. **Index Tools** (from MCP) live in `tool_index` and are reached on demand via **keyword (full-text) search in Postgres** — **O(1)** prompt cost regardless of how many MCP tools exist. |
| **Skills** | Owned per user. Every 10 closed sessions, an induction pass mines recurring requests into new skills (deduped by embedding). |
| **Provider** | OpenRouter: model context window from `/models`, chat completions with tool calling, real `usage` recorded per turn. |
| **Observability** | One `step_logs` row per loop step; `tokens_in/out` on every model turn; live totals on the session. |
| **Bash tool** | The agent's **universal fallback**: used whenever no specialized tool fits but the OS can do the job. Working directory **persists across calls** within a session; structured output (exit code / cwd / stdout / stderr); large output is head/tail-elided. Configurable timeout. |
| **Bash sandbox** | Runs behind a pluggable `SandboxBackend`. Ships with a local-subprocess impl (one workdir per session). Swap for gVisor/Firecracker/K8s for real isolation. |

## Quickstart (uv)

Requires [uv](https://docs.astral.sh/uv/). One-time setup:
```bash
uv sync --extra dev    # creates .venv and installs deps from pyproject.toml
```

### 1. Offline demo (no DB, no API key)
Runs the whole harness with an in-memory repo and a scripted fake provider:
```bash
uv run python -c "
from harness.testing import offline_harness
h = offline_harness()
session = h.start_session('u1')
for ev in h.run_turn_stream(session, 'hello'):
    print(ev)
"
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

> Not using uv? The core still runs with plain `python` (e.g. `python -m harness.interfaces.cli chat`);
> install deps however you like — they're declared in `pyproject.toml`.

## Persona

The system prompt is assembled in layers: a **persona** first, then the harness's
tool guidance (which tells the agent to treat **Bash as its universal fallback** —
use it whenever no specialized tool fits but the OS can solve the task).

Set the persona by creating a `PERSONA.md`, or point `HARNESS_PERSONA_PATH` at
one, or pass it in code:
```python
Harness(persona="You are Atlas, a terse senior SRE. You think in shell commands.")
```
An empty or comment-only `PERSONA.md` falls back to a built-in default identity.
Pass `system_prompt=...` to bypass the layered assembly entirely.

> `SOUL.md` is a deprecated fallback filename (one release, with a warning) —
> use `PERSONA.md`.

## Configuration

All via env (see `.env.example`): model, budgets, `RESPONSE_RESERVE_TOKENS`,
`MAX_STEPS`, `CHECKPOINT_EVERY_USER_TURNS`, `SKILL_INDUCTION_EVERY_SESSIONS`,
`SUMMARY_KEEP_RATIO`, `EMBEDDING_*`, `HARNESS_PERSONA_PATH`, `BASH_TIMEOUT`,
`BASH_MAX_OUTPUT`.

## Project layout

Modules are grouped into subpackages by concern; each subpackage's `__init__.py`
re-exports its public API, so e.g. `from harness.tools import Bash` still works
without knowing which submodule `Bash` actually lives in.

```
harness/
  settings.py         # env-driven Config (+ tiny .env loader)
  testing.py          # offline_harness() convenience factory for demos/tests
  core/
    app.py            # Harness facade — wire/swap components here
    loop.py           # the agent loop + token-budget guard
    models.py         # dataclasses mirroring the schema
  memory/
    window.py         # Memory: budget, build_window, chained summarize, checkpoints
    persona.py        # Persona: layered system prompt (PERSONA.md + tool guidance)
    skills.py         # Skills: induction (every N sessions, deduped)
  llm/
    provider.py       # LLM Provider contract + OpenRouter/Azure + FakeProvider
    registry.py       # ProviderRegistry (explicit, name-keyed provider selection)
    tokenizer.py      # tiktoken with heuristic fallback
  tools/
    builtin.py        # built-in Tools + ToolRegistry + Index Tool dispatch
    capabilities.py   # ToolProvider capability composition (MCP servers, bundles)
    sandbox.py        # SandboxBackend contract + local-subprocess impl
  mcp/
    client.py         # MCP clients: stdio + Streamable-HTTP, + tool ingestion
    oauth.py          # OAuth 2.1 flow for remote MCP (discovery, DCR, PKCE, cache)
  persistence/
    repository.py     # Repository contract + InMemory + Postgres(pgvector)
  observability/
    observer.py       # step logging + latency + pluggable Tracer
  interfaces/
    cli.py            # init-db, chat, serve
    server.py         # SSE HTTP server
schema.sql            # Postgres + pgvector DDL
docker-compose.yml    # pgvector/pgvector:pg16
pyproject.toml        # project metadata + deps (uv); `harness` console script
uv.lock               # pinned dependency lockfile
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

# remote Streamable-HTTP server — static bearer token...
h.add_mcp_http("https://fellow.app/mcp", name="fellow",
               headers={"Authorization": f"Bearer {token}"})
# ...or the interactive OAuth 2.1 flow (browser login + PKCE, token cached)
h.add_mcp_http("https://fellow.app/mcp", name="fellow", oauth=True)
```
The model finds these tools through `SearchTools` (keyword/full-text search over
`tool_index`), never via the prompt. See `examples/add_fellow_mcp.py`.

**From the CLI**, remote HTTP servers are auto-connected from the env — no code:
```bash
# .env
MCP_HTTP_SERVERS=fellow=https://fellow.app/mcp   # comma-separated name=url
MCP_FELLOW_OAUTH=1                                # browser OAuth (PKCE, token cached)
# MCP_FELLOW_TOKEN=...                            # or a static bearer instead
```
`uv run harness chat` connects each one at startup (failures are reported and
skipped) and disconnects them on exit. OAuth tokens are cached under
`~/.harness/mcp-auth/`, so the browser login happens only once.

## Documentation

Full docs (concepts, configuration reference, deployment, API reference) are
built with [MkDocs](https://www.mkdocs.org/) + Material. Serve them locally:

```bash
uv sync --extra docs
uv run mkdocs serve   # http://127.0.0.1:8000
```

## Status / notes

- Core logic verified by `tests/test_core.py`.
- Validated end-to-end against **real Postgres + pgvector** (repository, full-text
  `SearchTools`, summarization, checkpoints, induction, token accounting).
- OpenRouter integration verified up to billing: the harness authenticates,
  pulls the model context window, and forms valid chat requests. A live
  completion needs account credits (a `402 Payment Required` means no credit).
- The bundled `LocalSubprocessSandbox` is **not** isolated — replace it with a
  kernel-isolated backend before exposing untrusted multi-tenant Bash.
