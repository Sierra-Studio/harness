# CallTool: bridge MCP tool discovery → execution

Reference for an implementation pass on `harness/`. Not yet applied.

## Problem

Discovered MCP tools can't be invoked — discovery is wired, execution isn't.

- `builtin_specs()` (`harness/tools.py:30`) returns only 4 built-ins:
  `SearchTools`, `GetTools`, `GetSkills`, `Bash`.
- `loop.py:70` sends only `builtin_specs()` to the provider, so the model's
  callable set never includes indexed MCP tools.
- Under standard OpenAI/OpenRouter function-calling, a model can only emit a
  call for a function in the provided list — so it can never call an MCP tool.

The execution path already exists but is unreachable: `dispatch()` has an
`else: _index_tool(name, args)` branch (`tools.py:77`), and `_index_tool`
(`:144`) looks up the spec and calls `client.call_tool`. The model just never
gets to trigger it.

Pre-existing, not a regression (present in old and new harness).

Symptom: with an MCP server connected (e.g. Fellow), the agent finds
`search_meetings` via SearchTools/GetTools, reports "no executable path to it,"
and answers instead of acting.

## Fix: add a generic `CallTool` built-in

Close the loop: `SearchTools` → `GetTools` (schema) → `CallTool("search_meetings", {...})`.

1. Add `CallTool` to `BUILTINS` (`tools.py:17`).
2. Register its spec in `builtin_specs()` (`tools.py:36`):
   - `name` (string, required) — exact tool name from SearchTools.
   - `arguments` (object, required) — the tool's own input per its schema.
3. Route it in `dispatch()` (`tools.py:65`):
   `elif name == "CallTool": content = self._index_tool(args["name"], args["arguments"])`
4. **Drop the catch-all `else` branch** (`tools.py:77`). Today any unknown name
   falls through to `_index_tool`, so a hallucinated/typo'd tool name is
   silently routed to an MCP lookup. With `CallTool` explicit, route only it and
   let unknown names return a clean error.
5. Update `prompt.py` to teach the flow:
   SearchTools (find) → GetTools (schema) → CallTool (execute).

### Why CallTool, not dynamic spec injection

The alternative — inject a tool's spec into the model's function list after
GetTools — is more stateful and fights the design: specs are rebuilt fresh
every turn from `builtin_specs()` (`loop.py:70`), so injection would require
threading per-session "enabled tools" state through the loop. `CallTool` is
stateless and matches the existing dispatch. Clear winner.

## Auth (separate concern)

Once invocation is wired, servers like Fellow still need OAuth before a call
succeeds (401 otherwise). That's the MCP client's responsibility
(`_index_tool` → `client.call_tool`), independent of this bridge.

## Related: skills body gap

The companion issue — `GetSkills` returns `{name, summary}` only, so the agent
can list skills but not load their steps — is the same gap tracked in
[`skills-keyword-search.md`](./skills-keyword-search.md). Fix there via a
separate `GetSkill(name)` tool, **not** by folding `body` into `GetSkills`
(which would redump up to 5 full procedures and defeat progressive disclosure).
Converge on `GetSkill`; do not double-implement.

## Net result

`SearchTools → GetTools → CallTool` closes the discovery-to-execution loop with
one stateless built-in, and the dead `else` branch is removed.
