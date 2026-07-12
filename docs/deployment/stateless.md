# Stateless mode

`Harness.run_stateless` / `run_stateless_stream` are a first-class,
documented mode for callers who own conversation history themselves — e.g. a
request-scoped Postgres-backed app that keeps its own message table — and want
a single turn against caller-supplied history, persisting nothing on the
harness side.

```python
result = harness.run_stateless(
    history=[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
    message="what's next?",
)
```

Requires the harness to have been constructed with `repo=InMemoryRepository()`
(the default) — it raises otherwise, to avoid silently creating orphan rows
against a real Postgres-backed harness. `history` items are
`{"role": "user"|"assistant", "content": ...}`; anything else is ignored.

`run_stateless_stream` is the streaming form (same `LoopEvent` stream as
`run_turn_stream`); `run_stateless` collects it down to the final `TurnResult`.

## Resuming an existing session

For an app that *does* want the harness's own persistence, `start_session`
supports resuming:

```python
session = harness.start_session(external_id, session_id=existing_id)
```

If `session_id` is given and still exists in this harness's repo, that session
is resumed; otherwise a new one is created.

!!! danger "No ownership check"
    `start_session` performs **no** authorization/ownership check — it does
    not verify that `session_id` belongs to `external_id`, and it never will
    (`Harness` has no concept of "the current authenticated caller"). Never
    pass a `session_id` sourced directly from unauthenticated client input (a
    raw query param, an unsigned cookie). Resolve it first through your own
    access-controlled lookup — e.g. a table keyed by your app's own
    conversation id, scoped to the already-authenticated user — and only pass
    the resulting trusted `session_id` here.

## Related: separating runtime from conversation

For stateless-worker deployments (source of truth in a DB, no session
affinity, horizontal scale) that also want a *shared* runtime (provider, tool
registry, MCP connections) rather than rebuilding `Harness` per request, see
the proposed `HarnessRuntime` split in
[ADR 0001](../adr/0001-mcp-exposure-and-runtime-lifecycle.md#decision-3-separate-harnessruntime-shared-from-the-conversation-proposed).
