# Tools & Providers (capabilities)

## Tool

`Tool` (`harness/tools/builtin.py`) is a uniform abstraction covering both built-in
tools and developer-supplied ones. Each carries its own model-facing spec
(name, description, JSON-schema parameters) plus an optional `guidance`
snippet the system prompt assembles on demand — so tool guidance only appears
when the tool is actually active.

Build a custom tool with `make_tool(...)` rather than subclassing `Tool`
directly, unless you need custom `run` logic beyond a plain callable.

### Built-in tools

Always present in the prompt:

| Tool | Purpose |
|---|---|
| `SearchTools` | keyword/full-text search over `tool_index` (discovers MCP tools) |
| `GetTools` | fetch full specs for named tools found via search |
| `CallTool` | dispatch a call to a discovered (indexed) tool |
| `SearchSkills` | keyword search over a user's saved skills |
| `GetSkill` | fetch a skill's full body |
| `Bash` | the agent's universal fallback — see below |
| `RenderUI` | render a structured UI payload back to the caller |

`default_tools()` returns the standard set; `ToolRegistry` composes a list of
tools (built-ins + app-supplied + MCP-discovered) into one dispatchable
registry.

### The Bash tool

The agent's **universal fallback**: used whenever no specialized tool fits but
the OS can do the job. Its working directory **persists across calls** within
a session (a `cd` in one call carries to the next); output is structured (exit
code / cwd / stdout / stderr) and large output is head/tail-elided. Runs
behind a pluggable `SandboxBackend` — see [Sandbox](sandbox.md).

## ToolProvider (capability composition)

`harness/tools/capabilities.py` is the uniform way to compose **capabilities** into a
harness — not just a single tool. A `ToolProvider` is a self-contained
capability module that owns its own resources (connections, clients),
contributes one or more tools, and tears itself down:

```python
Harness(cfg, providers=[
    DomainTools(...),                           # app-supplied bundle
    McpServer(sandbox_url, "sandbox", expose="direct"),
    McpServer(jira_url, "jira", expose="index"),
])
```

Contract:

- `register(host: ProviderHost)` — contribute tools via the `ProviderHost`
  handed in. Called once during harness construction, **before** the system
  prompt is assembled, so a provider's tools (and their guidance) are
  advertised to the model from the start.
- `stop()` — release resources. Called by `Harness.close()`. Default: no-op.

`ProviderHost` is the small, stable surface a provider uses to wire itself in
— `add_tool()` for a direct tool, `add_mcp()` for an MCP client — so providers
never poke registry/repo internals directly.

Built-in providers:

- **`ToolBundle`** — the simplest provider: wraps a fixed list of already-built
  tools so stateless tools compose uniformly alongside connection-bearing
  providers.
- **`McpServer`** / **`McpStdioServer`** — see [MCP](mcp.md).

Set `optional = True` on a provider whose failure to register (e.g. an
unreachable server) should degrade gracefully — the harness logs and skips it
instead of aborting construction. Default is fail-fast.

!!! note "Not to be confused with LLM providers"
    `harness.tools` (`ToolProvider`) composes *capabilities* into a
    harness. `harness.llm` (`Provider`) is the contract for talking
    to an *LLM*. See [Providers (LLM)](llm-providers.md).
