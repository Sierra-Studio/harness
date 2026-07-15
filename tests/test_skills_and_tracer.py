"""Tests for the Skills and Tracer DI seams — run with: python3 -m pytest -q."""

from __future__ import annotations

from helpers import make_cfg as _cfg

from harness.core import Harness
from harness.llm.provider import FakeProvider
from harness.memory.skills import NullSkills, RepositorySkills, Skills
from harness.models import Skill
from harness.observability.observer import LoggingTracer, NullTracer, Observer, Tracer
from harness.persistence.repository import InMemoryRepository
from harness.settings import MemoryConfig

# --------------------------------------------------------------------------
# Skills DI
# --------------------------------------------------------------------------


def test_default_skills_is_repository_skills():
    h = Harness(_cfg(), provider=FakeProvider(context_window=4000))
    assert isinstance(h.skills, RepositorySkills)


def test_null_skills_disables_catalog_and_induction():
    p = FakeProvider(context_window=4000)

    class P(FakeProvider):
        def induce_skills(self, model, signals):
            raise AssertionError("induction must never run behind NullSkills")

    p = P(context_window=4000)
    h = Harness(_cfg(), system_prompt="sys.", provider=p, skills=NullSkills())
    uid = h.repo.get_or_create_user("u1").id
    # NullSkills.add() refuses to persist — the feature is off, not silently lossy.
    try:
        h.skills.add(uid, "n", "s", "b", "authored")
    except NotImplementedError:
        pass
    else:
        raise AssertionError("expected NotImplementedError from NullSkills.add")

    s = h.start_session("u1")
    p.queue(content="ok")
    h.run_turn(s, "hi")
    # no catalog block injected into the prompt sent to the model
    assert "# Your saved skills" not in p.calls[-1][0]["content"]
    # closing the session never triggers induction (NullSkills.on_session_closed -> [])
    for _ in range(50):
        s2 = h.start_session("u1")
        created = h.close_session(s2)
        assert created == []


def test_search_skills_and_get_skill_use_injected_skills_backend():
    """SearchSkills/GetSkill go through ctx.skills, not ctx.repo directly —
    swap in a custom Skills implementation and the tools follow it."""

    class InMemorySkills(Skills):
        def __init__(self):
            self._skills: dict[str, list[Skill]] = {}

        def list(self, user_id):
            return self._skills.get(user_id, [])

        def search(self, user_id, query, k):
            return [s for s in self.list(user_id) if query.lower() in s.name.lower()][:k]

        def get(self, user_id, name):
            return next((s for s in self.list(user_id) if s.name == name), None)

        def add(self, user_id, name, summary, body, origin):
            sk = Skill(id=name, user_id=user_id, name=name, summary=summary, body=body, origin=origin)
            self._skills.setdefault(user_id, []).append(sk)
            return sk

        def on_session_closed(self, session):
            return []

    custom = InMemorySkills()
    h = Harness(_cfg(), system_prompt="sys.", provider=FakeProvider(context_window=4000), skills=custom)
    assert h.skills is custom
    s = h.start_session("u1")  # session.user_id is the repo's internal id, not "u1"
    custom.add(s.user_id, "deploy_web", "Ship the web app", "1. test\n2. deploy", "authored")

    found = h.tools.dispatch(
        s, {"id": "1", "function": {"name": "SearchSkills", "arguments": '{"query": "deploy"}'}}
    )["content"]
    assert "deploy_web" in found

    body = h.tools.dispatch(
        s, {"id": "2", "function": {"name": "GetSkill", "arguments": '{"name": "deploy_web"}'}}
    )["content"]
    assert "1. test" in body


def test_repository_skills_matches_prior_builtin_behavior():
    """RepositorySkills reproduces the induction cadence + dedup that used to
    be hardcoded into Harness (regression guard for the DI extraction)."""

    class P(FakeProvider):
        def induce_skills(self, model, signals):
            return [{"name": "n", "summary": "s", "body": "do x"}]

    p = P(context_window=4000)
    h = Harness(_cfg(memory=MemoryConfig(skill_induction_every_sessions=2)), system_prompt="sys.", provider=p)
    uid = h.repo.get_or_create_user("alice").id
    for _ in range(2):
        s = h.start_session("alice")
        h.close_session(s)
    assert [sk.name for sk in h.skills.list(uid)] == ["n"]
    for _ in range(2):
        s = h.start_session("alice")
        h.close_session(s)
    assert len(h.skills.list(uid)) == 1  # deduped, not doubled


# --------------------------------------------------------------------------
# Tracer / observability DI
# --------------------------------------------------------------------------


def test_default_tracer_is_null_and_harmless():
    h = Harness(_cfg(), system_prompt="sys.", provider=FakeProvider(context_window=4000))
    assert isinstance(h.observer.tracer, NullTracer)
    s = h.start_session("u1")
    h.observer.log(s.id, None, "custom_event", {"k": "v"})  # must not raise


def test_custom_tracer_receives_spans_for_model_and_tool_calls():
    class RecordingTracer(Tracer):
        def __init__(self):
            self.spans: list[tuple[str, dict]] = []

        def span(self, name, **attributes):
            from contextlib import contextmanager

            @contextmanager
            def _cm():
                attrs = dict(attributes)
                try:
                    yield attrs
                finally:
                    self.spans.append((name, attrs))

            return _cm()

    tracer = RecordingTracer()
    p = FakeProvider(context_window=4000)
    p.queue(
        tool_calls=[
            {"id": "c1", "function": {"name": "Bash", "arguments": '{"command": "echo hi"}'}}
        ]
    )
    p.queue(content="done")
    h = Harness(_cfg(), system_prompt="sys.", provider=p, tracer=tracer)
    s = h.start_session("u1")
    h.run_turn(s, "go")

    names = [name for name, _ in tracer.spans]
    assert "model_call" in names
    assert "tool_call" in names
    model_span = next(attrs for name, attrs in tracer.spans if name == "model_call")
    assert model_span["tokens_in"] is not None and model_span["tokens_out"] is not None
    tool_span = next(attrs for name, attrs in tracer.spans if name == "tool_call")
    assert tool_span["detail"] == {"name": "Bash"}  # nested, not spread (no key collision)


def test_tracer_event_is_zero_duration_and_distinct_from_timed_span():
    class RecordingTracer(Tracer):
        def __init__(self):
            self.events = []
            self.spans = []

        def event(self, name, **attributes):
            self.events.append((name, attributes))

        def span(self, name, **attributes):
            from contextlib import contextmanager

            @contextmanager
            def _cm():
                self.spans.append(name)
                yield {}

            return _cm()

    tracer = RecordingTracer()
    observer = Observer(InMemoryRepository(), tracer=tracer)
    observer.log(None, None, "one_off", {"x": 1})
    assert tracer.events == [("one_off", {"session_id": None, "turn_id": None, "tokens_in": None, "tokens_out": None, "detail": {"x": 1}})]
    assert tracer.spans == []  # log() without latency_ms uses event(), not span()

    with observer.timed(None, None, "timed_step", {}):
        pass
    assert tracer.spans == ["timed_step"]  # timed() opens exactly one span
    assert len(tracer.events) == 1  # log() inside timed()'s finally sets latency_ms, so no extra event


def test_logging_tracer_smoke():
    """LoggingTracer runs end-to-end without needing a real logging backend."""
    h = Harness(
        _cfg(), system_prompt="sys.", provider=FakeProvider(context_window=4000), tracer=LoggingTracer()
    )
    s = h.start_session("u1")
    r = h.run_turn(s, "hi")
    assert r.status == "ok"
