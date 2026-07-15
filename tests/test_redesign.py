"""Tests for the explicit-DI redesign: required provider, static repo default,
tool-call limits, stateless mode, and session resume — run with:
python3 -m pytest -q.
"""

from __future__ import annotations

from helpers import make_cfg

from harness.core import Harness
from harness.llm.provider import FakeProvider
from harness.persistence.repository import InMemoryRepository, PostgresRepository
from harness.settings import LoopConfig


def _cfg(**loop_overrides):
    return make_cfg(loop=LoopConfig(**loop_overrides) if loop_overrides else None)


def test_harness_requires_provider():
    try:
        Harness(_cfg())
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError: provider is a required kwarg")


def test_harness_defaults_to_in_memory_repo():
    h = Harness(_cfg(), provider=FakeProvider(context_window=4000))
    assert isinstance(h.repo, InMemoryRepository)


def test_max_tool_calls_per_step_truncates():
    p = FakeProvider(context_window=4000)
    p.queue(
        tool_calls=[
            {"id": f"c{i}", "function": {"name": "Bash", "arguments": '{"command": "echo k"}'}}
            for i in range(5)
        ]
    )
    p.queue(content="done")
    h = Harness(
        _cfg(max_tool_calls_per_step=2), system_prompt="sys.", provider=p
    )
    s = h.start_session("u1")
    r = h.run_turn(s, "go")
    tool_turns = [t for t in h.repo.turns if t.role == "tool"]
    assert len(tool_turns) == 2  # truncated from 5 to the configured cap
    assert r.status == "ok"


def test_max_tool_calls_per_turn_stops_turn():
    p = FakeProvider(context_window=4000)
    for _ in range(5):
        p.queue(
            tool_calls=[
                {"id": "c", "function": {"name": "Bash", "arguments": '{"command": "echo k"}'}}
            ]
        )
    h = Harness(
        _cfg(max_tool_calls_per_turn=2), system_prompt="sys.", provider=p
    )
    s = h.start_session("u1")
    r = h.run_turn(s, "go")
    assert r.status == "tool_limit_exhausted"


def test_run_stateless_requires_in_memory_repo():
    class _FakePostgres(PostgresRepository):
        def __init__(self):  # bypass real psycopg connection
            pass

    p = FakeProvider(context_window=4000)
    h = Harness(_cfg(), provider=p, repo=_FakePostgres())
    try:
        h.run_stateless([], "hi")
    except RuntimeError as e:
        assert "InMemoryRepository" in str(e)
    else:
        raise AssertionError("expected RuntimeError for non-in-memory repo")


def test_run_stateless_seeds_history_each_call():
    p = FakeProvider(context_window=4000)
    p.queue(content="ok")
    h = Harness(_cfg(), system_prompt="sys.", provider=p)
    history = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]
    h.run_stateless(history, "follow-up")
    sent = p.calls[-1]
    joined = " ".join(m.get("content", "") for m in sent if isinstance(m.get("content"), str))
    assert "earlier question" in joined and "earlier answer" in joined


def test_start_session_resume_and_unknown_id_fallback():
    p = FakeProvider(context_window=4000)
    h = Harness(_cfg(), system_prompt="sys.", provider=p)
    s1 = h.start_session("u1")
    # resuming a known id returns the SAME session
    s2 = h.start_session("u1", session_id=s1.id)
    assert s2.id == s1.id
    # an unknown id falls back to creating a new session
    s3 = h.start_session("u1", session_id="does-not-exist")
    assert s3.id != s1.id


def test_multi_round_continuity_with_durable_session():
    p = FakeProvider(context_window=100_000)
    p.queue(content="first reply")
    p.queue(content="second reply")
    h = Harness(_cfg(), system_prompt="sys.", provider=p)
    s = h.start_session("u1")
    h.run_turn(s, "remember the number 42")
    h.run_turn(s, "what number did I mention?")
    # the second model call's window includes the first turn's content
    second_call_messages = p.calls[-1]
    joined = " ".join(
        m.get("content", "") if isinstance(m.get("content"), str) else ""
        for m in second_call_messages
    )
    assert "remember the number 42" in joined


def test_find_session_returns_none_for_unknown_id():
    repo = InMemoryRepository()
    assert repo.find_session("nope") is None
    user = repo.get_or_create_user("u1")
    s = repo.create_session(user.id, "m", 4000, 0)
    assert repo.find_session(s.id) is not None
    assert repo.find_session(s.id).id == s.id
