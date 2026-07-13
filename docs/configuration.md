# Configuration

Every field in `harness.settings` is a static literal default — nothing reads
`os.environ` at import time or as a field default. `Config()` with zero
arguments is always env-free and side-effect-free.

Reading the environment is an explicit, opt-in action the **calling
application** takes via `Config.from_env()` (or a sub-config's own
`from_env()`) — never something `Harness` or `Config`'s own defaults do for
you.

```python
from harness.settings import Config

cfg = Config()               # pure defaults, no env read
cfg = Config.from_env()      # reads os.environ (or an injected mapping)
```

`Config` is grouped by concern, each with its own `from_env()`:

| Sub-config | Covers |
|---|---|
| `ProviderConfig` | LLM provider selection, model, OpenRouter/Azure credentials |
| `LoopConfig` | steps, tool-call limits, token budget, response reserve |
| `MemoryConfig` | checkpoint cadence, skill induction cadence, summary keep-ratio, persona path |
| `BashConfig` | Bash tool timeout and output cap |

## Environment variables

See `.env.example` for a working template. Full list:

### Provider

| Variable | Default | Notes |
|---|---|---|
| `HARNESS_PROVIDER_NAME` | `""` | explicit registry key (e.g. `azure`, `openrouter`); never auto-detected by `build_provider` |
| `HARNESS_MODEL` | `deepseek/deepseek-v4-pro` | model id (Azure: the *deployment* name) |
| `OPENROUTER_API_KEY` | `""` | |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | |
| `AZURE_AI_ENDPOINT` | `""` | setting this selects Azure via `detect_provider` (takes precedence over OpenRouter) |
| `AZURE_AI_API_KEY` | `""` | empty ⇒ Entra ID / managed identity |
| `AZURE_AI_API_VERSION` | `preview` | |
| `AZURE_CLIENT_ID` | `""` | user-assigned managed identity client id |
| `DEFAULT_CONTEXT_WINDOW` | `128000` | fallback when the provider can't report a context window |

Managed identity (no stored secret) needs the `azure` extra: `pip install 'harness[azure]'`.

### Database

| Variable | Default | Notes |
|---|---|---|
| `DATABASE_URL` | `""` | empty ⇒ in-memory repository (dev/demo only — no persistence across restarts) |

### Loop

| Variable | Default |
|---|---|
| `MAX_STEPS` | `25` |
| `MAX_TOOL_CALLS_PER_STEP` | `8` (0 = unlimited) |
| `MAX_TOOL_CALLS_PER_TURN` | `40` (0 = unlimited) |
| `TOKEN_BUDGET_PER_SESSION` | `500000` (0/`none`/`unlimited`/`inf`/`-1` = unlimited) |
| `RESPONSE_RESERVE_TOKENS` | `4000` |

### Memory / Skills / Persona

| Variable | Default |
|---|---|
| `CHECKPOINT_EVERY_USER_TURNS` | `20` |
| `SKILL_INDUCTION_EVERY_SESSIONS` | `10` |
| `SKILLS_IN_PROMPT_LIMIT` | `30` |
| `SUMMARY_KEEP_RATIO` | `0.10` |
| `HARNESS_PERSONA_PATH` | `""` — empty ⇒ look for `./PERSONA.md`, else default identity |

### Bash tool

| Variable | Default |
|---|---|
| `BASH_TIMEOUT` | `60` seconds |
| `BASH_MAX_OUTPUT` | `10000` chars before head/tail elision |

### Embeddings (skills semantic search)

| Variable | Default | Notes |
|---|---|---|
| `EMBEDDING_API_KEY` | `""` | empty ⇒ deterministic local fallback (non-semantic, dev only) |
| `EMBEDDING_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | |
| `EMBEDDING_DIM` | `1536` | |

### Remote MCP servers (auto-connected by `harness chat`)

| Variable | Notes |
|---|---|
| `MCP_HTTP_SERVERS` | comma-separated `name=url` list |
| `MCP_<NAME>_TOKEN` | static bearer token for that server |
| `MCP_<NAME>_OAUTH` | `1`/`true`/`yes` ⇒ interactive browser OAuth 2.1 (PKCE, token cached under `~/.harness/mcp-auth/`) |

Read via `harness.settings.mcp_http_servers()`, an opt-in helper — nothing reads
this implicitly.

!!! warning "Ephemeral / container deployments"
    An unset `DATABASE_URL` silently falls back to the in-memory repository and
    loses **all** state on every restart or scale event. Set it to a managed
    Postgres and run `uv run harness init-db` once to apply `schema.sql`.
