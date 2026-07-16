# Harness

A pluggable, multi-tenant LLM agent harness: **Memory, Skills, MCP, Tools, Loop**,
an **OpenRouter**/**Azure AI Foundry** provider, and **token-level observability** —
all persisted in **Postgres** (or in-memory for dev/tests).

Every component sits behind an interface, so the provider, the sandbox, and the
persistence backend are all swappable. `Harness` (in `harness/core/app.py`) is the
single wiring point.

## What it does

| Component | Behaviour |
|---|---|
| **Persona** | Layered system prompt: a customizable persona (`PERSONA.md`) + built-in tool guidance. Falls back to a default identity when no `PERSONA.md` is set. |
| **Loop** | `perceive → build context → call model → run tools → repeat`, with a per-session **token-budget guard** that stops and returns the partial reply. |
| **Memory** | Window = `system prompt + chained summary + active turns`. Budget = `context_window − system_prompt − response_reserve`. On overflow: keep the last 10% verbatim, fold the rest (plus the previous summary) into a new **chained** summary. Folded turns stay in Postgres (`in_window=false`). |
| **Checkpoints** | Every 20 **user** turns, the subject is classified in a few words. |
| **Tools** | Custom tools with the **`@tool` decorator** (schema inferred from type hints + docstring). Built-ins always in the prompt — `SearchTools`, `GetTools`, `CallTool`, `SearchSkills`, `GetSkill`, `Bash`. **Index Tools** (from MCP) live in `tool_index` and are reached on demand via **keyword (full-text) search in Postgres** — O(1) prompt cost regardless of how many MCP tools exist. |
| **Skills** | Owned per user. Every 10 closed sessions, an induction pass mines recurring requests into new skills (deduped by name). |
| **Providers (LLM)** | OpenRouter and Azure AI Foundry: model context window discovery, chat completions with tool calling, real `usage` recorded per turn, plus an offline `FakeProvider` for tests. |
| **Providers (Tools)** | `ToolProvider` — the uniform way to compose capabilities (MCP servers, tool bundles) into a harness, each with its own lifecycle. |
| **Observability** | One `step_logs` row per loop step; `tokens_in/out` on every model turn; live totals on the session; pluggable `Tracer` for external systems (OpenTelemetry, Datadog, ...). |
| **Bash tool** | The agent's **universal fallback**: used whenever no specialized tool fits but the OS can do the job. Working directory **persists across calls** within a session; structured output (exit code / cwd / stdout / stderr); large output is head/tail-elided. Configurable timeout. |
| **Bash sandbox** | Runs behind a pluggable `SandboxBackend`. Ships with a local-subprocess impl (one workdir per session). Swap for gVisor/Firecracker/K8s for real isolation. |

## Where to start

- New to the project? Start with the [Quickstart](quickstart.md).
- Wiring your own config/provider/repo? See [Configuration](configuration.md).
- Want to understand a specific piece? See [Concepts](concepts/loop-and-memory.md).
- Running this as a service? See [Deployment](deployment/cli.md).
- Looking for a specific class or function? See the [API Reference](reference/core.md).
