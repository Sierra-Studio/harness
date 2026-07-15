"""Resume (human-in-the-loop) — deterministic continuation of a suspended turn.

A turn that stops on the permission gate is resumed by executing the exact
captured tool call and continuing the step loop, WITHOUT asking the model to
re-emit the call (no replay). These tests pin that behaviour.
"""

from __future__ import annotations

import pytest
from helpers import make_harness as _harness

from harness.llm.provider import FakeProvider
from harness.settings import PermissionConfig


def _suspended_seed(cmd="echo xyz"):
    """The window as captured at suspension: the user turn plus the assistant
    message whose (single) tool call is pending — no tool result yet."""
    return [
        {"role": "user", "content": "run it"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "Bash", "arguments": f'{{"command": "{cmd}"}}'}}
            ],
        },
    ]


def test_resume_executes_approved_call_then_continues():
    p = FakeProvider(context_window=4000)
    p.queue(content="done")  # the continuation completion, after the tool result
    h = _harness(p)

    approved = {"name": "Bash", "args": {"command": "echo xyz"}, "call_id": "c1"}
    events = list(h.resume_stateless_stream(_suspended_seed(), approved, external_id="chat"))

    kinds = [e.kind for e in events]
    # No redundant tool_start — the call was already announced pre-suspension;
    # resume emits only the new tool_result, then continues.
    assert "tool_start" not in kinds
    tool_res = next(e for e in events if e.kind == "tool_result")
    assert "xyz" in tool_res.content  # the real tool actually ran
    final = next(e for e in events if e.kind == "final")
    assert final.result.status == "ok"
    assert final.result.text == "done"

    # Determinism: exactly ONE model call (the continuation) — the model was
    # never asked to reproduce the tool call. Its window already contained the
    # assistant tool_call and the tool result.
    assert len(p.calls) == 1
    window = p.calls[0]
    assert any(m["role"] == "assistant" and m.get("tool_calls") for m in window)
    assert any(m["role"] == "tool" for m in window)


def test_resume_denied_records_denial_and_continues():
    p = FakeProvider(context_window=4000)
    p.queue(content="ok, skipping that")
    h = _harness(p)

    approved = {"name": "Bash", "args": {"command": "echo nope"}, "call_id": "c1", "denied": True}
    events = list(
        h.resume_stateless_stream(_suspended_seed("echo nope"), approved, external_id="chat")
    )

    tool_res = next(e for e in events if e.kind == "tool_result")
    assert "denied by the user" in tool_res.content
    assert "nope" not in tool_res.content  # the command never executed
    final = next(e for e in events if e.kind == "final")
    assert final.result.text == "ok, skipping that"


def test_resume_can_suspend_again_on_a_later_gated_tool():
    # Continuation emits a SECOND tool call; in manual mode the asker fires and,
    # if it raises, the exception propagates so the backend can re-suspend.
    p = FakeProvider(context_window=4000)
    p.queue(
        tool_calls=[
            {"id": "c2", "function": {"name": "Bash", "arguments": '{"command": "echo again"}'}}
        ]
    )
    h = _harness(p, permissions=PermissionConfig(mode="manual"))

    class _Suspend(Exception):
        pass

    def _asker(name, args):
        raise _Suspend()

    h.permissions.asker = _asker
    approved = {"name": "Bash", "args": {"command": "echo xyz"}, "call_id": "c1"}

    with pytest.raises(_Suspend):
        list(h.resume_stateless_stream(_suspended_seed(), approved, external_id="chat"))


def test_resume_requires_in_memory_repo(monkeypatch):
    p = FakeProvider(context_window=4000)
    h = _harness(p)
    # Simulate a non-ephemeral repo to hit the guard.
    monkeypatch.setattr(h, "repo", object())
    with pytest.raises(RuntimeError, match="InMemoryRepository"):
        list(h.resume_stateless_stream(_suspended_seed(), {"name": "Bash", "call_id": "c1"}))
