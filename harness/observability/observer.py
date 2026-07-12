"""Observability: every loop step is logged; token spend is first-class.

Persistence (`repo.add_step_log`) and the `echo` stdout printer are built in
and always on — that's the audit trail. Tracing/spans for an external system
(OpenTelemetry, Datadog, plain `logging`, ...) are pluggable: inject a
`Tracer` via `Harness(tracer=...)` (or `Observer(repo, tracer=...)` directly).
The static default is `NullTracer` — same pattern as `sandbox=`/`repo=`: a
fixed, predictable no-op unless you supply your own, never something
`Harness` selects by inspecting `Config`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ..persistence import Repository


class Tracer:
    """Pluggable tracing backend. Subclass and override `span` (the rest
    follow from it) to bridge into OpenTelemetry, Datadog, structlog, or
    anything else with a start/attributes/duration shape. Mirrors `Hook`:
    a plain, overridable base — not an ABC — because the no-op default
    (`NullTracer`, an alias for this class) is itself a valid, useful value.
    """

    def span(self, name: str, **attributes: Any) -> Iterator[dict]:
        """Context manager for one unit of work (a model call, a tool call,
        memory summarization, skill induction, ...). Must be a context
        manager yielding a mutable mapping the caller may add attributes to
        before the span closes (e.g. tokens_in/out) and must propagate any
        exception raised inside the `with` block after recording it. Default:
        a no-op span."""
        return _null_span()

    def event(self, name: str, **attributes: Any) -> None:
        """A zero-duration, point-in-time event (e.g. `hook_error`,
        `provider_unavailable`, `skill_induced`). Default: an empty span."""
        with self.span(name, **attributes):
            pass


@contextmanager
def _null_span(*_a: Any, **_k: Any) -> Iterator[dict]:
    yield {}


class NullTracer(Tracer):
    """Explicit no-op tracer — identical to `Tracer()` itself; use whichever
    reads better at the call site."""


class LoggingTracer(Tracer):
    """A minimal, dependency-free `Tracer` that writes each span to Python's
    `logging` module. Useful as-is for simple deployments, and as a template
    for adapting to a real tracing backend: implement `Tracer.span` against
    your own client (e.g. `opentelemetry.trace.Tracer.start_as_current_span`)
    instead of subclassing this."""

    def __init__(self, logger: logging.Logger | None = None, level: int = logging.INFO):
        self._logger = logger or logging.getLogger("harness.trace")
        self._level = level

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[dict]:
        attrs: dict = dict(attributes)
        t0 = time.perf_counter()
        try:
            yield attrs
        except Exception:
            attrs["error"] = True
            raise
        finally:
            attrs["duration_ms"] = int((time.perf_counter() - t0) * 1000)
            self._logger.log(self._level, "%s %s", name, attrs)


class Observer:
    def __init__(self, repo: Repository, echo: bool = False, tracer: Tracer | None = None):
        self.repo = repo
        self.echo = echo
        self.tracer = tracer or NullTracer()

    def log(
        self,
        session_id: str | None,
        turn_id: str | None,
        step_type: str,
        detail: dict,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        latency_ms: int | None = None,
    ) -> None:
        self.repo.add_step_log(
            session_id, turn_id, step_type, detail, tokens_in, tokens_out, latency_ms
        )
        if self.echo:
            cost = ""
            if tokens_in is not None or tokens_out is not None:
                cost = f" [in={tokens_in or 0} out={tokens_out or 0}]"
            print(f"  · {step_type}{cost} {detail}")
        # `timed()` below already opens/closes its own span around calls that
        # set latency_ms; only forward point-in-time log() calls here so a
        # timed step isn't traced twice. `detail` is nested (not spread) since
        # its keys are caller-controlled and could otherwise collide with the
        # fixed attributes below (e.g. a detail dict containing "name").
        if latency_ms is None:
            self.tracer.event(
                step_type,
                session_id=session_id,
                turn_id=turn_id,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                detail=detail,
            )

    @contextmanager
    def timed(self, session_id, turn_id, step_type, detail):
        """Context manager that records latency for a step (and, via the
        injected `Tracer`, a span for it). Yields a dict the caller can fill
        with tokens_in/tokens_out before exit."""
        slot = {"tokens_in": None, "tokens_out": None}
        t0 = time.perf_counter()
        with self.tracer.span(
            step_type, session_id=session_id, turn_id=turn_id, detail=detail
        ) as span_attrs:
            try:
                yield slot
            finally:
                dt = int((time.perf_counter() - t0) * 1000)
                span_attrs["tokens_in"] = slot["tokens_in"]
                span_attrs["tokens_out"] = slot["tokens_out"]
                self.log(
                    session_id, turn_id, step_type, detail, slot["tokens_in"], slot["tokens_out"], dt
                )
