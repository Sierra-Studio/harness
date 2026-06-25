# CallTool: bridge MCP tool discovery → execution

Reference for an implementation pass on `harness/`. Not yet applied.

> **Status correction (verified against a live run).** The original premise of
> this doc — that MCP tools are *uninvocable* — is **false in practice for
> lenient models**. See "How invocation actually works today" below. The
> `CallTool` fix is still worth doing for robustness, but it is an *improvement*,
> not a missing bridge, and the catch-all `else` branch must **not** be removed.

## How invocation actually works today

Empirically, with OpenRouter + `deepseek-v4-pro`, an MCP tool call **succeeds**.
Observed trace: `SearchTools` → `GetTools` → Bash×3 → `search_meetings` (the raw
MCP tool name) → answer. So the discovery→execution loop already closes.

Why, despite the tool never being in `builtin_specs()`:

1. **The "only call listed functions" rule is a convention, not a runtime
   constraint.** Nothing stops a model from emitting a `tool_call` whose name is
   not in the provided list. Lenient models (deepseek) do exactly that; stricter
   models (e.g. some OpenAI models) refuse and answer in text instead.
2. **The loop doesn't validate tool names.** `loop.py:86-87` passes every
   `res.tool_calls` entry straight to `dispatch()` — no membership check.
3. **`dispatch()`'s catch-all `else` executes any unknown name.** `tools.py:77`
   routes it to `_index_tool(name, args)` (`:144`), which looks up the spec and
   calls `client.call_tool`.

So **the `else` branch IS the working bridge.** The report's "no executable
path" conclusion holds only for strict models that won't emit an unlisted name —
which is the likely source of the original symptom. This is model-dependent
behavior, not a hard gap.

### Empirically confirmed

Scripting a model to emit a tool call named `echo` (not in the built-in list)
made the harness execute it via `else → _index_tool` and return the MCP result:

```
TOOL RESULT: {'tool_call_id': 'c1',
  'content': '{"content": [{"type": "text", "text": "hello-via-else-branch"}]}'}
```

### Why it's model-dependent (not transport-dependent)

OpenRouter is the same transport for every model: the harness sends identical
requests — `tools = [the 4 builtins]`, `tool_choice: "auto"` — to
`/chat/completions`, and OpenRouter relays whatever the backend returns without
validating emitted tool names. The variable is the **backend model's decoding
policy**:

- **Unconstrained (deepseek-v4-pro, others):** happily emits a tool call named
  `search_meetings` → relayed verbatim → `else → _index_tool` runs it. ✓
- **Constrained (OpenAI family, e.g. GPT-mini):** strongly biased to only the
  provided function names, so it won't emit `search_meetings`; lacking a
  callable function it explains it can't reach the tool. ✗ (the original
  screenshot). This is a training/decoding bias — "overwhelmingly won't," not a
  hard grammar in standard chat-completions.

**App impact:** the default model is `deepseek-v4-pro`, so MCP tools work today
(given the connector is authorized). Selecting an OpenAI-family model in the
picker is where invocation would silently not fire — and the reason to add
`CallTool`.

## Problem (the real, narrower one)

Invocation works only by relying on the model going "off-script" and emitting a
raw MCP tool name. That's fragile:

- Strict models won't do it → they fall back to answering in text (the reported
  symptom).
- It's implicit: there's no declared, schema'd way to invoke, so the model has
  to infer it's allowed to emit the bare name.

`builtin_specs()` (`harness/tools.py:30`) advertises only `SearchTools`,
`GetTools`, `GetSkills`, `Bash` — so a well-behaved model has no *declared* path
to execution even though `_index_tool` (`:144`) is ready to run it.

## Fix: add a generic `CallTool` built-in (robustness, not a missing bridge)

Give strict/well-behaved models a *declared* execution path so invocation no
longer depends on emitting a raw, unlisted tool name. Close the loop:
`SearchTools` → `GetTools` (schema) → `CallTool("search_meetings", {...})`.

1. Add `CallTool` to `BUILTINS` (`tools.py:17`).
2. Register its spec in `builtin_specs()` (`tools.py:36`):
   - `name` (string, required) — exact tool name from SearchTools.
   - `arguments` (object, required) — the tool's own input per its schema.
3. Route it in `dispatch()` (`tools.py:65`):
   `elif name == "CallTool": content = self._index_tool(args["name"], args["arguments"])`
4. **Keep the catch-all `else` branch** (`tools.py:77`). It is the working
   bridge for lenient models that emit the raw MCP name (verified with
   deepseek-v4-pro). Removing it would break invocation for those models.
   `CallTool` and the `else` are belt-and-suspenders: declared path for strict
   models, raw-name path for lenient ones — both land in `_index_tool`.
5. Update `prompt.py` to teach the flow:
   SearchTools (find) → GetTools (schema) → CallTool (execute). This nudges
   strict models onto the declared path and makes lenient models more reliable.

### Optional hardening

If hallucinated/typo'd tool names routing into `_index_tool` becomes noisy,
don't remove the `else` — instead have `_index_tool` already returns a clean
`"Unknown tool '<name>'. Use SearchTools first."` when the lookup misses
(`tools.py:147`), which is the desired behavior. No change needed.

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

Invocation already works for lenient models via the `else` branch. Adding
`CallTool` gives strict models a declared `SearchTools → GetTools → CallTool`
path too, so execution is reliable across providers — without removing the
existing raw-name bridge.
