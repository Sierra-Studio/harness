# Tools & Providers (capabilities)

## What a tool is

A **tool** is a capability the model can invoke: it has a model-facing spec
(name, description, JSON-Schema parameters) and a handler that runs when the
model calls it. The life of a tool call:

```
system prompt ── tool specs advertised to the model
      │
      ▼
model emits a tool call ──► ToolRegistry.dispatch()
      │                          │
      │                          ▼
      │                    Tool.run(ctx, session, args) ──► result string
      │                          │
      ◄──────────────────────────┘
model reads the result and continues the turn
```

`Tool` (`harness/tools/builtin.py`) is the uniform abstraction covering both
built-in tools and developer-supplied ones. Every tool carries five things:

| Field | What it is | Who sees it |
|---|---|---|
| `name` | unique identifier the model calls | model |
| `description` | what the tool does | model |
| `parameters` | JSON-Schema of the arguments | model |
| `guidance` | *when/how* to use it — composed into the **system prompt** only while the tool is active | model (system prompt) |
| `run()` | the handler that does the work | harness only |

`description` and `guidance` answer different questions: the description says
*what the tool does* ("Current weather for a city"); guidance says *when to
prefer it or how to combine it with siblings* ("Prefer this over Bash+curl
when the user asks about weather"). Guidance is optional and only enters the
prompt when the tool is registered — so unused tools cost nothing.

## Writing a tool with `@tool` (recommended)

Decorate a plain function; everything is inferred from its signature and
docstring:

```python
from harness import tool

@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Current weather for a city.

    Args:
        city: City name to look up.
        unit: "celsius" or "fahrenheit".

    Guidance:
        Prefer this over Bash+curl when the user asks about weather.
    """
    return lookup(city, unit)
```

This is what the model receives:

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Current weather for a city.",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {"type": "string", "description": "City name to look up."},
        "unit": {"type": "string", "description": "\"celsius\" or \"fahrenheit\"."}
      },
      "required": ["city"]
    }
  }
}
```

The decorated object **is a `Tool`** — drop it in `tools=[...]` or a
`ToolBundle` — and **stays callable** as the original function, so tests and
other code can keep invoking it directly:

```python
get_weather("São Paulo")            # plain call still works
get_weather.spec()                  # the JSON above
```

### What inference covers

| Python annotation | JSON-Schema |
|---|---|
| `str` | `{"type": "string"}` |
| `int` | `{"type": "integer"}` |
| `float` | `{"type": "number"}` |
| `bool` | `{"type": "boolean"}` |
| `dict` | `{"type": "object"}` |
| `list[str]` | `{"type": "array", "items": {"type": "string"}}` |
| `Literal["a", "b"]` | `{"enum": ["a", "b"], "type": "string"}` |
| `int \| None` / `Optional[int]` | `{"type": "integer"}` |
| no annotation / `Any` / wide unions | `{}` (any) |

Inference is deliberately shallow. For schemas it can't express (nested
objects, per-field constraints), pass an explicit `parameters=` — see
[Overriding inference](#overriding-inference).

### Required vs optional arguments

One rule, taken from the signature: **no default → required**.

```python
@tool
def search(
    query: str,                  # no default            → required
    limit: int | None,           # Optional, no default  → still required
    lang: str = "pt",            # has a default         → optional
    filters: str | None = None,  # default None          → optional
):
    """Search the catalog. ..."""
```

produces `"required": ["query", "limit"]`. Note that `Optional[X]` / `X | None`
describes the *type* ("accepts X or None"), not optionality — same semantics
as Python itself. Defaults are applied by Python when the model omits an
argument; if the model should *know* the default, say so in the `Args:`
description (`lang: Language code. Default: "pt".`).

### Accessing harness internals: `ctx` and `session`

Parameters named `ctx` and/or `session` are **reserved**: they never appear in
the model-facing schema and are injected at dispatch time. Both are optional —
most tools need neither.

```python
@tool
def list_orders(status: str, ctx=None, session=None) -> list:
    """List a user's orders by status.

    Args:
        status: "open", "shipped" or "delivered".
    """
    rows = ctx.repo.query(...)           # ctx: repo, sandbox, mcp_clients, config, skills
    return [r for r in rows if r.user == session.user_id]
```

The model only ever sees `status`. `ctx` is a `ToolContext`; `session` is the
current `Session`.

### Return values

The handler may return any of these — no serialization boilerplate needed:

| Return | Sent to the model |
|---|---|
| `str` | verbatim |
| `dict` / `list` / numbers | JSON-encoded (`ensure_ascii=False`) |
| `None` | `""` |

### Overriding inference

Precedence is always **explicit decorator option > inference**:

| Field | Explicit (wins) | Inferred fallback |
|---|---|---|
| `name` | `@tool(name=...)` | function name |
| `description` | `@tool(description=...)` | docstring before the first section |
| `parameters` | `@tool(parameters=...)` | type hints + defaults + `Args:` |
| `guidance` | `@tool(guidance=...)` | `Guidance:` docstring section |

```python
@tool(
    name="OrderSearch",                       # model-facing name ≠ function name
    parameters={                              # hand-written schema, inference skipped
        "type": "object",
        "properties": {
            "filter": {
                "type": "object",
                "properties": {"status": {"type": "string"}, "days": {"type": "integer"}},
            },
        },
        "required": ["filter"],
    },
)
def search_orders(filter: dict) -> list:
    """Search orders with a structured filter."""
    ...
```

Errors are raised **at decoration time**, not at dispatch: a function with no
docstring and no `description=` raises `TypeError`, and so does `*args` /
`**kwargs` (pass `parameters=` instead).

## Other ways to build a tool

### `make_tool` — assembly from parts

For tools built at runtime (from config, in a loop) where there is no function
to decorate:

```python
from harness import make_tool

for entity in ("orders", "invoices", "customers"):
    tools.append(make_tool(
        name=f"count_{entity}",
        description=f"Count {entity} records.",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=lambda session, args, e=entity: str(counts[e]),
    ))
```

The handler signature is `(session, args) -> str`: arguments arrive as a raw
dict and the result must already be a string.

### Subclassing `Tool` — full control

When `run` needs logic beyond a plain callable (state, retries, streaming into
the sandbox), subclass and set the class attributes; `spec()` is assembled for
you:

```python
from harness import Tool

class Deploy(Tool):
    name = "Deploy"
    description = "Deploy a service to an environment."
    parameters = {
        "type": "object",
        "properties": {"service": {"type": "string"}, "env": {"type": "string"}},
        "required": ["service", "env"],
    }
    guidance = "Always run Deploy only after the user confirms the target env."

    def __init__(self, client):
        self._client = client

    def run(self, ctx, session, args) -> str:
        result = self._client.deploy(args["service"], args["env"])
        return result.summary
```

Rule of thumb: `@tool` for functions, `make_tool` for runtime assembly,
subclass for stateful/custom dispatch.

## Composing tools into a harness

Pass tools (and providers — next section) through the constructor:

```python
from harness import Harness, default_tools, ToolBundle

Harness(cfg, provider=llm, tools=[*default_tools(), get_weather, list_orders])
# or, equivalently, grouped as a provider:
Harness(cfg, provider=llm, tools=[*default_tools(), ToolBundle([get_weather, list_orders])])
```

`tools=` takes a single mixed list: bare `Tool`s and `ToolProvider`s compose
side by side (omit it entirely to get just the built-ins).

`default_tools()` returns the built-in set; `ToolRegistry` composes everything
into one dispatchable registry. Duplicate names **raise** at registration
(`register(tool, replace=True)` opts into replacement) — a tool registered
after construction is advertised to the model on the next turn.

## ToolProvider (capability composition)

`harness/tools/capabilities.py` is the uniform way to compose **capabilities** into a
harness — not just a single tool. A `ToolProvider` is a self-contained
capability module that owns its own resources (connections, clients),
contributes one or more tools, and tears itself down:

```python
Harness(cfg, provider=llm, tools=[
    *default_tools(),
    ToolBundle([get_weather]),                  # stateless tools, no lifecycle
    ErpTools(erp_url, token),                   # app-supplied, owns a client
    McpServer(sandbox_url, "sandbox", expose="direct"),
    McpServer(jira_url, "jira", expose="index", optional=True),
])
```

Contract:

- `register(host: ProviderHost)` — contribute tools via the `ProviderHost`
  handed in. Called once during harness construction, **before** the system
  prompt is assembled, so a provider's tools (and their guidance) are
  advertised to the model from the start.
- `stop()` — release resources. Called by `Harness.close()`. Default: no-op.

A complete custom provider with a connection lifecycle:

```python
from harness import tool, ToolProvider

class ErpTools(ToolProvider):
    """Owns one ERP client shared by all its tools."""

    def __init__(self, url: str, token: str):
        self._url, self._token = url, token
        self._client = None

    def register(self, host) -> None:
        self._client = ErpClient(self._url, self._token)   # connect once
        client = self._client

        @tool
        def stock(sku: str) -> dict:
            """Stock level for a SKU."""
            return client.stock(sku)

        @tool
        def price(sku: str) -> float:
            """Current price for a SKU."""
            return client.price(sku)

        host.add_tool(stock)
        host.add_tool(price)

    def stop(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
```

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

## Built-in tools

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

### The Bash tool

The agent's **universal fallback**: used whenever no specialized tool fits but
the OS can do the job. Its working directory **persists across calls** within
a session (a `cd` in one call carries to the next); output is structured (exit
code / cwd / stdout / stderr) and large output is head/tail-elided. Runs
behind a pluggable `SandboxBackend` — see [Sandbox](sandbox.md).

!!! note "Not to be confused with LLM providers"
    `harness.tools` (`ToolProvider`) composes *capabilities* into a
    harness. `harness.llm` (`Provider`) is the contract for talking
    to an *LLM*. See [Providers (LLM)](llm-providers.md).
