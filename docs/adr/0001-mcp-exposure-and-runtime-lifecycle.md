# ADR 0001 — MCP tool exposure & runtime/turn lifecycle separation

Status: **partially accepted** — Decisions 1 & 2 implemented; Decisions 3 & 4 proposed.

## Context

Applications embedding the harness alongside an MCP-backed **sandbox** (or any
small, always-relevant MCP server) hit three recurring problems. They were first
observed integrating a gVisor sandbox control plane (exposed via a FastMCP
Streamable-HTTP server) into a FastAPI "chat with your data" backend, but none of
them are app-specific — any harness + MCP deployment meets them.

1. **MCP tools are second-class.** They are indexed in the repo and reached only
   through `SearchTools`/`GetTools`/`CallTool`. That discovery hop is the right
   default for a large SaaS catalog (you cannot dump 500 schemas into context),
   but for a 6-tool sandbox it is pure friction: the model frequently never takes
   the hop and instead refuses ("I can't run code"). Apps worked around this by
   hand-rolling proxy `Tool`s and **mutating the registry's private `tools`
   dict** — a layering violation forced by a missing public API.

2. **No public tool-registration API.** `ToolRegistry` exposed no supported way
   to add a tool after construction, so the workaround above was the *only* way.

3. **Runtime and turn lifetimes are conflated.** The `Harness` bundles
   long-lived, expensive, shareable state (provider, tool registry, **MCP
   connections**) with per-conversation state (memory, budget, session). A common
   stateless-worker deployment (source of truth in a DB, no session affinity,
   horizontal scale) wants a *fresh* per-request conversation but a *shared*
   runtime. With only `Harness` available, apps rebuild the whole thing per
   request — which drags MCP connect + `tools/list` (three network round-trips)
   and a connection **leak** into every turn's hot path.

## Decision 1 — public `ToolRegistry.register(tool, *, replace=False)` — DONE

The supported way to extend the live registry after construction. Because
`tool_specs()` reads the map at call time, a registered tool reaches the model on
the next turn. Raises on name clash unless `replace=True`.

## Decision 2 — `expose=` policy on `add_mcp_*` — DONE

`add_mcp_http`/`add_mcp_stdio` gain a keyword-only `expose: "index" | "direct"`
(default `"index"`, behaviour-preserving).

- `"index"` — unchanged discovery model.
- `"direct"` — each of the server's tools is also registered (via Decision 1) as
  a first-class `McpProxyTool`, so its schema is sent to the model directly and
  its summary is composed into the prompt's tool-guidance layer automatically.

`McpProxyTool.run` calls the owning client directly, so it carries **no**
dependency on the repo index — a direct-exposed server needs no discovery
machinery at all.

### Consequence for apps

The app drops its custom proxy classes, its registry-poking, and the
tool-name-specific lines in its system prompt. It writes only:

```python
harness.add_mcp_http(sandbox_url, "sandbox", expose="direct")
```

## Decision 3 — separate `HarnessRuntime` (shared) from the conversation — PROPOSED

Introduce a long-lived runtime owning config, provider, tool registry, and MCP
connections; make turns ephemeral over caller-supplied context:

```python
rt = HarnessRuntime(cfg, provider=..., tools=domain_tools)   # once, at app startup
rt.add_mcp_http(sandbox_url, "sandbox", expose="direct")     # one connection, app lifetime
# per request, on a worker thread — no repo writes, no reconnect:
rt.run_turn_stream(system_prompt=..., history=[...], user=query, model=..., hooks=[...])
```

This serves the stateless-worker deployment directly and makes the MCP connection
app-scoped *by construction* rather than by each app remembering to hoist it.
Incremental fallback if the full split is too large: a `Harness.run_stateless_turn(...)`
that takes caller-supplied history and persists nothing — same benefit, smaller
surface.

**Open question:** does connecting MCP *before* prompt assembly (so
`expose="direct"` guidance lands in the prompt without an app-side capability
line) become the norm under the runtime? Under Decision 2 alone, apps that still
build `Harness` per request connect MCP *after* construction, so the proxy
guidance misses the prompt and a generic capability line in the system prompt is
still needed. The runtime removes that ordering constraint.

## Decision 4 — declare the MCP client concurrency contract — PROPOSED

A shared runtime is hit by concurrent turns on worker threads. `HttpMcpClient`
carries a per-session `Mcp-Session-Id`; concurrent `call_tool`s on one session
are unsafe. Pick and document one:

- **Per-call HTTP session** — stateless request/response. Simplest, safe under
  concurrency, but loses server-side session state.
- **Bounded client pool per server** — health-checked, lazy-reconnect. Safe,
  preserves state, and yields the timeout/degradation story for free.

**Recommendation:** the pool. The sandbox exposes `run_session` (stateful), and
the pool is the only option that both stays safe and keeps that capability. This
belongs in the library so no app re-solves it.

## Rollout

Additive, behaviour-preserving defaults throughout. Ship order:
**1 → 2** (apps drop the poke, the wrappers, and the prompt hack immediately)
**→ 3** (apps drop per-request reconnect) **→ 4** (concurrency guarantee). Each
step is independently shippable and independently valuable.
