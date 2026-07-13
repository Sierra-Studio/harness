# Harness

A pluggable, multi-tenant LLM agent harness: **Memory, Skills, MCP, Tools, Loop**,
a full-screen **chat TUI**, four **LLM providers** (OpenRouter, Azure AI Foundry,
Google Vertex AI, AWS Bedrock), and **token-level observability** — all persisted
in **Postgres** (or in-memory for dev/tests).

Every component sits behind an interface, so the provider, the sandbox, and the
persistence backend are all swappable. `Harness` (`harness/core/app.py`) is the
single wiring point.

## What it does

| Component | Behaviour |
|---|---|
| **Persona** | Layered system prompt: a customizable persona (`PERSONA.md`, inspired by Hermes' SOUL) + built-in tool guidance. Falls back to a default identity when no `PERSONA.md` is set. |
| **Loop** | `perceive → build context → call model → run tools → repeat`, with a per-session **token-budget guard** and per-step/per-turn **tool-call limits** that stop and return a partial reply. |
| **Memory** | Window = `system prompt + chained summary + active turns`. Budget = `context_window − system_prompt − response_reserve`. On overflow: keep the last 10% verbatim, fold the rest (plus the previous summary) into a new **chained** summary. Folded turns stay in Postgres (`in_window=false`). |
| **Checkpoints** | Every 20 **user** turns, the subject is classified in a few words. |
| **Tools** | Built-ins always in the prompt — `SearchTools`, `GetTools`, `CallTool`, `SearchSkills`, `GetSkill`, `Bash`. **Index Tools** (from MCP) live in `tool_index` and are reached on demand via **keyword (full-text) search in Postgres** — **O(1)** prompt cost regardless of how many MCP tools exist. |
| **Skills** | Owned per user. Every 10 closed sessions, an induction pass mines recurring requests into new skills (deduped by embedding). |
| **Providers (LLM)** | `OpenRouterProvider`, `AzureFoundryProvider`, `VertexProvider`, `BedrockProvider` — each report a model's context window, do chat completions with tool calling, and record real `usage` per turn — plus an offline `FakeProvider` for tests/demos. |
| **Providers (Tools)** | `ToolProvider` — the uniform way to compose capabilities (MCP servers, tool bundles) into a harness, each with its own lifecycle. |
| **Observability** | One `step_logs` row per loop step; `tokens_in/out` on every model turn; live totals on the session; pluggable `Tracer` seam for OpenTelemetry/Datadog/`logging`. |
| **Bash tool** | The agent's **universal fallback**: used whenever no specialized tool fits but the OS can do the job. Working directory **persists across calls** within a session; structured output (exit code / cwd / stdout / stderr); large output is head/tail-elided. Configurable timeout. |
| **Bash sandbox** | Runs behind a pluggable `SandboxBackend`. Ships with a local-subprocess impl (one workdir per session). Swap for gVisor/Firecracker/K8s for real isolation. |

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Docker.

### Fastest path (`make`)

```bash
make setup   # uv sync --extra dev, create .env, start Postgres+pgvector, apply schema.sql
make chat    # interactive session
```
Run `make help` for the full list (`test`, `lint`, `typecheck`, `check`, `demo`, `serve`, `docs`, `db-reset`, ...).

> The bundled `docker-compose.yml` publishes Postgres on **host port 5433** (not
> 5432), so `DATABASE_URL` must point at `localhost:5433` — `make env` sets this
> up correctly; if you hand-roll `.env` from `.env.example`, double-check the port.

### Manual path (uv)

One-time setup:
```bash
uv sync --extra dev    # creates .venv and installs deps from pyproject.toml
```

**1. Offline demo** (no DB, no API key) — runs the whole harness with an in-memory
repo and a scripted fake provider:
```bash
uv run python -c "
from harness.testing import offline_harness
h = offline_harness()
session = h.start_session('u1')
for ev in h.run_turn_stream(session, 'hello'):
    print(ev)
"
```

**2. Tests**
```bash
uv run pytest -q
```

**3. Real run** (an LLM provider + Postgres):
```bash
cp .env.example .env          # set OPENROUTER_API_KEY (needs credits) or an Azure/Vertex/Bedrock block
docker compose up -d          # Postgres + pgvector, host port 5433
uv run harness init-db        # apply schema.sql
uv run harness chat           # interactive session
```
With `DATABASE_URL` unset the harness uses the in-memory repo; with no LLM
provider configured it uses the offline `FakeProvider`. Mix and match, e.g.
force offline + in-memory:
```bash
DATABASE_URL="" OPENROUTER_API_KEY="" uv run harness chat
```

> Not using uv? The core still runs with plain `python` (e.g. `python -m harness.interfaces.cli chat`);
> install deps however you like — they're declared in `pyproject.toml`.

## Install as a global command (run `harness` in any folder)

To use `harness` from anywhere on your machine — like `claude` — install the
console script onto your `PATH`. The `harness` command is declared in
`pyproject.toml` (`[project.scripts]`), so any of these work:

```bash
uv tool install --editable .   # recommended while developing: reflects your working tree
# uv tool install .            # or a pinned snapshot (rerun with --force to update)
# pipx install .               # non-uv equivalent
```

`uv tool` installs into `~/.local/bin` (run `uv tool update-shell` once if that
isn't on your `PATH`). Then, from any directory:

```bash
harness              # start a chat in the current folder (Bash/Write/Edit act on it)
harness --version    # print the version   (also: -V)
harness --help       # list commands        (also: -h)
```

**Config from any folder.** On startup `harness` loads environment config with
this precedence (highest first):

1. real shell environment variables
2. a `.env` in the **current** folder (per-project override)
3. `~/.harness/.env` (**global** fallback)

So drop your provider keys once into `~/.harness/.env` and the command works
everywhere; a project can still override per-folder with its own `.env`:

```bash
mkdir -p ~/.harness && cp .env ~/.harness/.env && chmod 600 ~/.harness/.env
```

> With `DATABASE_URL` set in the global file, sessions persist via Postgres (only
> while it's running); omit it to fall back to the in-memory repo so `harness`
> runs even with no database.

To update or remove the global command: `uv tool install . --force` /
`uv tool uninstall harness`.

## Interactive session (`harness chat`)

In a real terminal, `chat` launches a full-screen [Textual](https://textual.textualize.io/)
TUI (`harness/interfaces/tui.py`): scrollable message history, a docked input,
and a status bar with an animated "thinking…"/"running `<tool>`…" spinner plus a
live token-budget readout. Each turn runs in a worker thread, so the UI never
freezes while the model works. When stdout/stdin aren't a TTY (piped, CI,
`--plain`), or Textual can't import, it falls back to a line-based Rich REPL that
renders the same widgets inline.

`↑`/`↓` recall previous messages, `/` triggers ghost-text command autocomplete, a
trailing `\` continues a line, `Ctrl+C`/`Esc` stops a running turn (quits when
idle), `Ctrl+L` clears. Type `/help` for the full list — everything else is sent
to the model as a message.

| Command | Does |
|---|---|
| `/session` | session id, model, context window, token spend |
| `/sessions` | list your recent sessions |
| `/resume [n\|id]` | resume a past session and reload its history |
| `/retry` | re-run your last message |
| `/copy` | copy the last answer to the clipboard |
| `/save [file]` | save the transcript as markdown |
| `/skills`, `/skills add <name> <summary> [body...]` | list / author skills |
| `/tools` | list active tools, grouped by source |
| `/mcp`, `/mcp http <url> [name] [--direct]`, `/mcp stdio <name> <cmd...> [--direct]`, `/mcp remove <name>` | manage MCP servers live |
| `/persona [text\|clear]`, `/system-prompt [text\|clear]` | show / set / reset the system prompt (mutually exclusive) |
| `/model [name]`, `/budget [n\|unlimited]`, `/theme [name]` | change and persist session settings |
| `/mode`, `/auto`, `/manual` (or `Shift+Tab`) | show/switch tool-permission mode — `auto` runs every tool, `manual` asks before each side-effecting one (shown in the bar below the input) |
| `/new`, `/clear`, `/exit`/`/quit` | new session / clear screen / end session |

`/persona`, `/system-prompt`, `/model`, `/budget`, `/theme` persist to
`~/.harness/preferences.json`; MCP servers added via `/mcp http|stdio` persist to
`~/.harness/mcp_servers.json` — both reload automatically on the next `harness chat`.
Precedence: a real env var > a saved preference > a `.env`-file default > the
library default. Full reference: [`docs/deployment/cli.md`](docs/deployment/cli.md).

## CLI commands

```bash
uv run harness init-db                                   # create schema in DATABASE_URL
uv run harness chat [user]                                # interactive session
uv run harness serve [host] [port]                        # HTTP server, turns as Server-Sent Events
uv run harness add-skill <user_id> <name> <summary> [body] # author a skill (stdin if body omitted)
uv run harness list-skills <user_id>
```
(or without uv: `python -m harness.interfaces.cli init-db | chat | serve | ...`)

The CLI is an **application**, not library-internal code — it's the one place
that opts into `Config.from_env()`, picks a provider via `detect_provider(cfg)`,
and picks a repository from `DATABASE_URL` (`build_repository`). `Harness` never
does any of that on its own.

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

## Configuration

Everything is env-driven and read explicitly via `Config.from_env()` — `Config()`
with no arguments is always env-free (see `harness/settings.py`). `.env.example`
is a working template; `docs/configuration.md` is the full reference. Highlights:

- **Provider selection**: set exactly one of `AZURE_AI_ENDPOINT`,
  `GCP_VERTEX_PROJECT`, `AWS_BEDROCK_REGION`, or `OPENROUTER_API_KEY` —
  `detect_provider()` (used by the CLI) picks in that precedence order, falling
  back to the offline `FakeProvider` if none are set. `HARNESS_MODEL` doubles as
  the Azure *deployment* name once Azure wins. Azure/Vertex/Bedrock each need an
  optional extra: `pip install 'harness[azure|vertex|bedrock]'`.
- **Budgets/loop**: `TOKEN_BUDGET_PER_SESSION`, `RESPONSE_RESERVE_TOKENS`,
  `MAX_STEPS`, `MAX_TOOL_CALLS_PER_STEP`, `MAX_TOOL_CALLS_PER_TURN`.
- **Memory/skills**: `CHECKPOINT_EVERY_USER_TURNS`, `SKILL_INDUCTION_EVERY_SESSIONS`,
  `SKILLS_IN_PROMPT_LIMIT`, `SUMMARY_KEEP_RATIO`, `HARNESS_PERSONA_PATH`.
- **Bash tool**: `BASH_TIMEOUT`, `BASH_MAX_OUTPUT`.
- **Database**: `DATABASE_URL` — unset ⇒ in-memory repository (dev/demo only,
  loses all state on restart).
- **Remote MCP**: `MCP_HTTP_SERVERS` (comma-separated `name=url`), per-server
  `MCP_<NAME>_TOKEN` or `MCP_<NAME>_OAUTH=1`.

## Project layout

Modules are grouped into subpackages by concern; each subpackage's `__init__.py`
re-exports its public API, so e.g. `from harness.tools import Bash` still works
without knowing which submodule `Bash` actually lives in.

```
harness/
  __init__.py          # top-level re-exports (Harness, Config, tools, ...)
  settings.py           # env-driven Config (+ tiny .env loader)
  models.py              # dataclasses mirroring the schema
  testing.py             # offline_harness() convenience factory for demos/tests
  core/
    app.py               # Harness facade — wire/swap components here
    loop.py               # the agent loop + token-budget + tool-call-limit guards
  memory/
    window.py             # Memory: budget, build_window, chained summarize, checkpoints
    persona.py             # Persona: layered system prompt (PERSONA.md + tool guidance)
    skills.py              # Skills: seam + RepositorySkills induction (every N sessions)
  llm/
    provider.py            # Provider contract + OpenRouter/Azure/Vertex/Bedrock/Fake
    registry.py             # ProviderRegistry (explicit, name-keyed provider selection)
    tokenizer.py            # tiktoken with heuristic fallback
  tools/
    builtin.py              # built-in Tools + ToolRegistry + Index Tool dispatch
    capabilities.py          # ToolProvider capability composition (MCP servers, bundles)
    sandbox.py                # SandboxBackend contract + local-subprocess impl
  mcp/
    client.py                # MCP clients: stdio + Streamable-HTTP, + tool ingestion
    oauth.py                  # OAuth 2.1 flow for remote MCP (discovery, DCR, PKCE, cache)
  persistence/
    repository.py             # Repository contract + InMemory + Postgres(pgvector)
  observability/
    observer.py                # step logging + latency + pluggable Tracer
  interfaces/
    cli.py                     # entry point: init-db, chat, serve, add-skill, list-skills
    tui.py                      # full-screen Textual chat app
    ui.py                        # shared Rich renderables (TUI + plain REPL)
    prefs.py                      # ~/.harness/preferences.json
    mcpstore.py                    # ~/.harness/mcp_servers.json
    server.py                       # SSE HTTP server
    assets/logo.png                 # welcome-card logo
docs/                    # MkDocs site: concepts, deployment, reference, ADRs
examples/                # add_fellow_mcp.py, custom_tools_and_hooks.py
tests/                   # test_core.py + provider/CLI/prefs/skills/tracer suites
schema.sql               # Postgres + pgvector DDL
docker-compose.yml       # pgvector/pgvector:pg16, host port 5433
Makefile                 # setup/db/chat/serve/lint/typecheck/test/docs shortcuts
mkdocs.yml               # docs site config
pyproject.toml           # project metadata + deps (uv); `harness` console script
uv.lock                  # pinned dependency lockfile
```

## Swapping pieces

`app.py` is the single wiring point. Examples:
```python
Harness(cfg, provider=MyProvider(...))          # different LLM gateway
Harness(cfg, sandbox=FirecrackerSandbox(...))   # kernel-isolated Bash
Harness(cfg, repo=PostgresRepository(dsn))       # forced backend
Harness(cfg, skills=NullSkills())                # disable skills entirely
Harness(cfg, tracer=LoggingTracer())             # bridge spans into logging/OTel/Datadog
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
`~/.harness/mcp-auth/`, so the browser login happens only once. Servers can also
be added live with `/mcp http`/`/mcp stdio` (see above) — those persist to
`~/.harness/mcp_servers.json`.

## Documentation

Full docs (concepts, configuration reference, deployment, API reference, ADRs)
are built with [MkDocs](https://www.mkdocs.org/) + Material, sourced from `docs/`:

```bash
uv sync --extra docs
uv run mkdocs serve   # http://127.0.0.1:8000
```

## Status / notes

- Core logic verified by `tests/` (`test_core.py`, plus dedicated suites for
  providers, CLI/TUI rendering, prefs/runtime controls, skills, and the
  redesigned DI seams).
- Validated end-to-end against **real Postgres + pgvector** (repository, full-text
  `SearchTools`, summarization, checkpoints, induction, token accounting).
- OpenRouter and Azure AI Foundry integration verified up to billing: the harness
  authenticates, resolves the model's context window, and forms valid chat
  requests. A live completion needs account credits/quota (OpenRouter:
  `402 Payment Required` means no credit).
- Vertex AI and Bedrock providers are implemented against the same
  `OpenAICompatibleProvider`/`Provider` contracts but are newer additions —
  exercise them against your own project/account before relying on them in
  production.
- The bundled `LocalSubprocessSandbox` is **not** isolated — replace it with a
  kernel-isolated backend before exposing untrusted multi-tenant Bash.
