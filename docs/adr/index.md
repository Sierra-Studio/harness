# Architecture Decision Records

ADRs capture the *why* behind non-obvious design decisions in this project —
context, alternatives, and the rollout plan — so they don't have to be
re-derived from the code later.

| ADR | Status | Summary |
|---|---|---|
| [0001](0001-mcp-exposure-and-runtime-lifecycle.md) | partially accepted | Public `ToolRegistry.register`, `expose=` policy on MCP providers ("done"); separate `HarnessRuntime` from per-conversation state, and an MCP client concurrency contract (proposed) |
| [0002](0002-decorator-extension-api.md) | accepted | `@tool` (schema inferred from signature + docstring, explicit options win) and `@before_turn`/`@after_turn`/`@before_tool`/`@after_tool` as the low-boilerplate extension path; classes remain for state and lifecycle |
