# ADR 0002 — Decorator-based extension API (`@tool` and hook decorators)

Status: **accepted** — implemented.

## Context

The two extension points a developer touches most — custom tools and hooks —
required disproportionate boilerplate relative to the size of the task:

1. **Tools.** `make_tool(...)` demanded a hand-written JSON-Schema and a
   `(session, args) -> str` handler that unpacks a raw dict, even for a
   three-line function. The schema duplicated information already present in
   the function's signature (types, defaults) and docstring (descriptions) —
   two sources of truth that drift.

2. **Hooks.** Intercepting a single point (block a command, redact a result)
   required a full `Hook` subclass to override one method. The class carried
   no state; it existed only to satisfy the registration shape.

3. **Precedent.** The codebase already used the pattern for providers
   (`@register_provider`), and the surrounding ecosystem (FastMCP's
   `@mcp.tool`, LangChain's `@tool`) has converged on decorators as the
   idiom for "one function + registration metadata".

## Decision 1 — `@tool`: a plain function becomes a `Tool` — DONE

`harness/tools/decorator.py`. The decorated object IS a `Tool` instance
(composes into `tools=[...]` / `ToolBundle`) and stays callable with the
original signature.

- **Inference, signature-first**: `name` from the function name, `description`
  from the docstring summary, `parameters` from type hints + defaults
  (`str`/`int`/`float`/`bool`/`dict`/`list[T]`/`Literal`/`Optional`),
  per-argument descriptions from a Google-style `Args:` section, `guidance`
  from a `Guidance:` section. Required = no default.
- **Precedence**: explicit decorator option (`name=`, `description=`,
  `parameters=`, `guidance=`) beats its inferred counterpart, wholesale.
- **Injection**: parameters named `ctx` / `session` are reserved — excluded
  from the model-facing schema, injected at dispatch.
- **Fail at decoration, not dispatch**: missing description and
  `*args`/`**kwargs` raise `TypeError` at import time.
- **Deliberately shallow inference**: nested/constrained schemas take an
  explicit `parameters=`; stateful tools still subclass `Tool`; runtime
  assembly keeps `make_tool`.

## Decision 2 — hook-point decorators — DONE

`harness/core/hooks.py`: `@before_turn`, `@after_turn`, `@before_tool`,
`@after_tool`. The decorated object IS a `Hook` (`FunctionHook`) with exactly
one point filled, composing into `hooks=[...]` beside class-based hooks.

- The function is stored as an **instance attribute** under the hook-point
  name, shadowing the class no-op — the loop's existing fan-out
  (`h.before_tool(...)`) finds it unbound, matching the plain function's
  signature. Zero changes to the loop.
- Return semantics are the `Hook` method's own (dict replaces args, string
  replaces result, turn-hook returns ignored).
- The signature is validated against the hook point at decoration time.
- Subclassing `Hook` remains the shape for state shared across points.

## Where decorators were considered and rejected

- **`ToolProvider`** — owns resources and a lifecycle (`register`/`stop`);
  a class is the honest shape. Factories *returning* decorated tools cover
  per-harness configuration (tenant credentials, role-scoped schemas).
- **`Tracer` / `Repository` / `SandboxBackend`** — backend adapter contracts;
  subclassing is the idiom.

The dividing rule: a decorator fits when the developer writes **one function**
and the rest is registration metadata; a class stays when there is **state and
lifecycle**.

## Consequences

- The common tool/hook goes from a class (or schema dict + adapter) to a
  decorated function; signature and docstring become the single source of
  truth the model sees.
- Decorated module-level objects are singletons — fine for stateless tools,
  and per-harness variation composes via factories without new API.
- `make_tool`, `Tool` subclassing, and `Hook` subclassing remain supported and
  documented; the decorators are additive, no behaviour change for existing
  code.

## Related follow-ups (not part of this ADR)

Candidates surfaced by the same analysis, in rough value order: a
`readonly=True` flag on `@tool` feeding the permission gate (today a
hard-coded frozenset); a decorated slash-command registry deduplicating the
CLI/TUI `elif` chains; retry/backoff as a decorator over provider calls.
