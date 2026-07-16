# Loop & Memory

## The agent loop

`AgentLoop` (`harness/core/loop.py`) runs: **perceive → build context → call model →
run tools → repeat**, under a per-session token-budget guard that stops and
returns the partial response when the ceiling is reached.

Key entry points:

```python
loop.start_session(external_id, model="") -> Session
loop.run_turn(session, user_message) -> TurnResult
loop.run_turn_stream(session, user_message) -> Iterator[LoopEvent]
```

`run_turn_stream` yields a `LoopEvent` per assistant text delta and per tool
call/result, then a final event carrying the `TurnResult`. `run_turn` is the
non-streaming convenience wrapper.

### Hooks

`Hook` gives you four extension points without subclassing the loop:

```python
class Hook:
    def before_turn(self, session, message) -> None: ...
    def after_turn(self, session, result) -> None: ...
    def before_tool(self, session, name, args) -> dict | None: ...
    def after_tool(self, session, name, result) -> str | None: ...
```

`before_tool`/`after_tool` can rewrite arguments/results by returning a
non-`None` value. Pass hooks in via `Harness(hooks=[...])`.

For the common case — one function for one point — decorate a plain function
instead of subclassing; the decorated object IS a `Hook` and stays callable:

```python
from harness import before_tool, after_tool

@before_tool
def block_rm(session, name, args):
    if name == "Bash" and "rm -rf" in args.get("command", ""):
        return {"command": "echo 'blocked by policy'"}   # rewrite the call

@after_tool
def redact(session, name, result):
    return result.replace(SECRET, "***")                 # rewrite the result

Harness(cfg, provider=llm, hooks=[block_rm, redact])
```

`@before_turn` and `@after_turn` work the same way (returns ignored). A
function whose signature doesn't match the hook point fails at decoration
time, not mid-turn. Subclass `Hook` when one object needs state shared across
several points.

### Token-budget guard

`LoopConfig.token_budget_per_session` (env `TOKEN_BUDGET_PER_SESSION`, default
500,000; `0`/`none`/`unlimited`/`inf`/`-1` = unlimited) caps total spend for a
session. When exceeded mid-turn, the loop stops and returns whatever partial
`TurnResult` it has — it never raises.

`LoopConfig.max_steps`, `max_tool_calls_per_step`, and
`max_tool_calls_per_turn` bound loop iterations and tool-call fan-out per turn.

## Memory

`Memory` (`harness/memory/window.py`) manages the token-budgeted context window:

```
window = system prompt + accumulated (chained) summary + active turns
budget = context_window - system_prompt_tokens - response_reserve
```

On overflow: keep the last `SUMMARY_KEEP_RATIO` (default 10%) of turns
verbatim, fold the rest — plus the previous summary — into a new **chained**
summary via the provider's `summarize()`. Folded turns are never deleted; they
stay in the repository with `in_window=false`, so nothing is lost, only moved
out of the active prompt.

### Checkpoints

Every `CHECKPOINT_EVERY_USER_TURNS` (default 20) **user** turns, the subject of
the conversation is classified in a few words via the provider's
`classify_subject()` and recorded — useful for session lists/search in an
application UI.
