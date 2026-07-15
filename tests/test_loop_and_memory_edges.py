"""Pins the core's documented failure-mode promises.

Each test here guards a behavior the code promises (mostly in docstrings and
comments) but that no test previously exercised: hooks can never kill a turn,
every terminal status is reachable, malformed model output degrades gracefully,
and memory's summarize/checkpoint guards hold.
"""

from __future__ import annotations

import pytest
from helpers import make_harness

from harness.core.loop import Hook
from harness.llm.provider import FakeProvider
from harness.memory import Memory
from harness.models import Turn
from harness.settings import LoopConfig, MemoryConfig
from harness.tools.builtin import make_tool


def _bash_call(cmd="echo xyz", call_id="c1"):
    return [{"id": call_id, "function": {"name": "Bash", "arguments": f'{{"command": "{cmd}"}}'}}]


# --------------------------------------------------------------------------
# Loop: hooks, terminal statuses, malformed model output
# --------------------------------------------------------------------------


def test_raising_hook_never_kills_turn_and_is_logged():
    """A hook that raises at EVERY lifecycle point must not break the turn;
    each failure is recorded as a hook_error step log instead."""

    class Boom(Hook):
        def before_turn(self, session, message):
            raise RuntimeError("bt")

        def after_turn(self, session, result):
            raise RuntimeError("at")

        def before_tool(self, session, name, args):
            raise RuntimeError("btool")

        def after_tool(self, session, name, result):
            raise RuntimeError("atool")

    p = FakeProvider(context_window=4000)
    p.queue(tool_calls=_bash_call())
    p.queue(content="done")
    h = make_harness(p, hooks=[Boom()])
    s = h.start_session("u1")
    r = h.run_turn(s, "go")

    assert r.status == "ok" and r.text == "done"
    # the tool still ran despite before_tool/after_tool raising
    tool_turns = [t for t in h.repo.turns if t.role == "tool"]
    assert tool_turns and "xyz" in tool_turns[0].content["content"]
    # every failing hook method was logged as hook_error
    hooks_logged = {
        log["detail"]["hook"] for log in h.repo.step_logs if log["step_type"] == "hook_error"
    }
    assert {"before_turn", "before_tool", "after_tool", "after_turn"} <= hooks_logged


def test_max_steps_status_preserves_last_text():
    """The fourth terminal status: the loop gives up after max_steps, keeping
    the model's last non-empty text as the partial answer."""
    p = FakeProvider(context_window=4000)
    p.queue(content="", tool_calls=_bash_call(call_id="c1"))
    p.queue(content="almost there", tool_calls=_bash_call(call_id="c2"))
    h = make_harness(p, loop=LoopConfig(max_steps=2))
    s = h.start_session("u1")
    r = h.run_turn(s, "go")
    assert r.status == "max_steps"
    assert r.steps == 2
    assert r.text == "almost there"


def test_max_steps_placeholder_when_model_said_nothing():
    p = FakeProvider(context_window=4000)
    p.queue(content="", tool_calls=_bash_call())
    h = make_harness(p, loop=LoopConfig(max_steps=1))
    s = h.start_session("u1")
    r = h.run_turn(s, "go")
    assert r.status == "max_steps"
    assert r.text == "(max steps reached)"


def test_malformed_tool_arguments_fall_back_to_empty_dict():
    """A model emitting broken JSON in `arguments` must not crash the turn:
    _parse degrades to {} and the tool still runs."""
    seen: dict = {}

    def handler(session, args):
        seen["args"] = args
        return "probed"

    probe = make_tool("Probe", "records its args", {"type": "object", "properties": {}}, handler)
    p = FakeProvider(context_window=4000)
    p.queue(tool_calls=[{"id": "c1", "function": {"name": "Probe", "arguments": '{"broken'}}])
    p.queue(content="done")
    h = make_harness(p, tools=[probe])
    s = h.start_session("u1")
    r = h.run_turn(s, "go")
    assert r.status == "ok"
    assert seen["args"] == {}  # invalid JSON degraded to empty args, tool still ran


def test_tool_exception_becomes_error_result_not_a_crash():
    """'tools must never crash the loop': an exception inside a tool comes back
    to the model as an ERROR tool result and the turn completes normally."""

    def handler(session, args):
        raise RuntimeError("boom")

    exploding = make_tool("Explode", "always raises", {"type": "object", "properties": {}}, handler)
    p = FakeProvider(context_window=4000)
    p.queue(tool_calls=[{"id": "c1", "function": {"name": "Explode", "arguments": "{}"}}])
    p.queue(content="recovered")
    h = make_harness(p, tools=[exploding])
    s = h.start_session("u1")
    r = h.run_turn(s, "go")
    assert r.status == "ok" and r.text == "recovered"
    tool_turns = [t for t in h.repo.turns if t.role == "tool"]
    assert tool_turns and tool_turns[0].content["content"] == "ERROR running Explode: boom"


def test_tool_calls_truncated_is_observable():
    """Truncation above max_tool_calls_per_step is already tested; this pins
    that it is also LOGGED (requested vs kept) for observability."""
    p = FakeProvider(context_window=4000)
    p.queue(
        tool_calls=[
            {"id": f"c{i}", "function": {"name": "Bash", "arguments": '{"command": "echo k"}'}}
            for i in range(5)
        ]
    )
    p.queue(content="done")
    h = make_harness(p, loop=LoopConfig(max_tool_calls_per_step=2))
    s = h.start_session("u1")
    h.run_turn(s, "go")
    logs = [log for log in h.repo.step_logs if log["step_type"] == "tool_calls_truncated"]
    assert logs and logs[0]["detail"] == {"requested": 5, "kept": 2}


# --------------------------------------------------------------------------
# Memory: checkpoint guard, summarize guards, message translation
# --------------------------------------------------------------------------


def test_checkpoint_guard_prevents_duplicates():
    """maybe_checkpoint must be idempotent between user turns: once a
    checkpoint exists at user-turn N, calling it again without new user turns
    is a no-op (the `last >= n` guard)."""
    p = FakeProvider(context_window=4000)
    h = make_harness(p, memory=MemoryConfig(checkpoint_every_user_turns=1))
    s = h.start_session("u1")
    p.queue(content="ok")
    h.run_turn(s, "q0")
    assert len(h.repo.checkpoints) == 1

    session = h.repo.get_session(s.id)
    assert h.memory.maybe_checkpoint(session) is False  # same N, guard holds
    assert len(h.repo.checkpoints) == 1


def test_as_message_translates_each_role_shape():
    """_as_message: the multimodal user envelope forwards only `content` (UI
    metadata dropped); tool turns keep tool_call_id; assistant/tool strings
    pass through; non-envelope dicts are JSON-serialized."""

    def turn(role, content):
        return Turn(
            id="t", session_id="s", user_id="u", idx=0, role=role, content=content, token_count=1
        )

    blocks = [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
    ]
    envelope = {
        "kind": "user_message",
        "content": blocks,
        "text": "what is this?",  # display metadata: must NOT reach the model
        "attachments": ["shot.png"],
    }
    msg = Memory._as_message(turn("user", envelope))
    assert msg == {"role": "user", "content": blocks}

    msg = Memory._as_message(turn("tool", {"tool_call_id": "c9", "content": "out"}))
    assert msg == {"role": "tool", "tool_call_id": "c9", "content": "out"}

    assert Memory._as_message(turn("tool", "plain")) == {"role": "tool", "content": "plain"}
    assert Memory._as_message(turn("assistant", "hello")) == {
        "role": "assistant",
        "content": "hello",
    }
    # a plain (non-envelope) dict user turn is serialized, not forwarded raw
    msg = Memory._as_message(turn("user", {"free": "form"}))
    assert msg["role"] == "user" and msg["content"] == '{"free": "form"}'


def test_summarize_failure_aborts_the_turn():
    """Pins CURRENT behavior: a provider failure during summarization
    propagates and kills the turn (there is no retry/fallback). If that policy
    ever changes — e.g. skip summarization on failure — update this test."""

    class SummarizeBoom(FakeProvider):
        def summarize(self, model, prev_summary, messages):
            raise RuntimeError("summarizer down")

    p = SummarizeBoom(context_window=140)
    h = make_harness(
        p,
        loop=LoopConfig(response_reserve_tokens=10),
        memory=MemoryConfig(summary_keep_ratio=0.2),
    )
    s = h.start_session("u1")
    with pytest.raises(RuntimeError, match="summarizer down"):
        for _ in range(6):
            p.queue(content="answer " + "z" * 80)
            h.run_turn(s, "question " + "w" * 60)


def test_maybe_summarize_skips_when_everything_is_in_the_kept_slice():
    """Even over budget, if keep_ratio covers all active turns there is
    nothing to fold — maybe_summarize must return False, not loop or raise."""
    p = FakeProvider(context_window=50)
    h = make_harness(
        p,
        loop=LoopConfig(response_reserve_tokens=10),
        memory=MemoryConfig(summary_keep_ratio=1.0),
    )
    s = h.start_session("u1")
    h.memory.append(s, "user", "x" * 400)  # way over the tiny budget
    assert h.memory.maybe_summarize(s, "sys.") is False
    assert h.repo.summaries == []


def test_chained_summaries_link_parent_ids_deterministically():
    """Two forced folds: the second summary must point at the first
    (parent_id) and cover a later turn (covers_until). Deterministic — no
    'if there happened to be two folds' like the loop-driven test."""
    p = FakeProvider(context_window=140)
    h = make_harness(
        p,
        loop=LoopConfig(response_reserve_tokens=10),
        memory=MemoryConfig(summary_keep_ratio=0.5),
    )
    s = h.start_session("u1")
    for i in range(2):
        h.memory.append(s, "user", f"m{i} " + "z" * 200)
    assert h.memory.maybe_summarize(s, "sys.") is True
    first = h.repo.current_summary(s.id)
    assert first is not None and first.parent_id is None

    for i in range(2):
        h.memory.append(s, "user", f"n{i} " + "z" * 200)
    assert h.memory.maybe_summarize(s, "sys.") is True
    second = h.repo.current_summary(s.id)
    assert second.id != first.id
    assert second.parent_id == first.id  # the chain links back
    assert second.covers_until > first.covers_until
    # folded turns stay in the repo, just out of the window
    assert len(h.repo.active_turns(s.id)) < len(h.repo.turns)
