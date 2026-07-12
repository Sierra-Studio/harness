# Observability

`harness/observability/observer.py` treats token spend and step logging as first-class.

Persistence (`repo.add_step_log`) and the `echo` stdout printer are built in
and always on — that's the audit trail. Tracing/spans for an external system
(OpenTelemetry, Datadog, plain `logging`, ...) are pluggable: inject a `Tracer`
via `Harness(tracer=...)` (or `Observer(repo, tracer=...)` directly). The
static default is `NullTracer` — zero overhead when you don't need external
tracing.

```python
class Tracer:
    def span(self, name: str, **attributes) -> Iterator[dict]: ...  # timed
    def event(self, name: str, **attributes) -> None: ...            # instantaneous
```

- `NullTracer` — the default; does nothing.
- `LoggingTracer` — emits spans/events through the stdlib `logging` module.
- Write your own for OpenTelemetry, Datadog, etc.

`Observer.log(...)` records a one-off event (zero duration); `Observer.timed(...)`
is a context manager that wraps a span and calls `log()` with the measured
latency on exit — use it for anything you want to time.

## What gets recorded

- One `step_logs` row per loop step.
- `tokens_in`/`tokens_out` on every model turn.
- Live running totals on the session.
- `model_call` and `tool_call` spans, each with attributes (token counts, tool
  name, etc.) — visible to any injected `Tracer`.
