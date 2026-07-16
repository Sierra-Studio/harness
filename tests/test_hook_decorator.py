"""@before_turn/@after_turn/@before_tool/@after_tool decorators."""

from __future__ import annotations

import pytest
from helpers import make_harness

from harness import Hook, after_tool, after_turn, before_tool, before_turn
from harness.core.hooks import FunctionHook
from harness.llm.provider import FakeProvider
from harness.tools.builtin import default_tools


def test_decorated_functions_are_hooks_and_stay_callable():
    @before_tool
    def audit(session, name, args):
        return None

    assert isinstance(audit, Hook)
    assert isinstance(audit, FunctionHook)
    assert audit.point == "before_tool"
    assert audit("s", "Bash", {}) is None  # still a plain call
    # only the decorated point is filled; the others stay class no-ops
    assert "after_tool" not in vars(audit)


def test_wrong_signature_fails_at_decoration():
    with pytest.raises(TypeError, match="session, name, args"):

        @before_tool
        def bad(name):
            return None

    with pytest.raises(TypeError, match="session, result"):

        @after_turn
        def also_bad(session, result, extra_required):
            return None


def test_hooks_fire_and_transform_through_the_loop():
    events = []

    @before_turn
    def on_start(session, message):
        events.append(("before_turn", message))

    @after_turn
    def on_end(session, result):
        events.append(("after_turn", result.status))

    @before_tool
    def rewrite(session, name, args):
        events.append(("before_tool", name))
        return {"command": "echo rewritten"}

    @after_tool
    def annotate(session, name, result):
        return result + "\n[audited]"

    p = FakeProvider(context_window=4000)
    p.queue(
        tool_calls=[
            {"id": "c1", "function": {"name": "Bash", "arguments": '{"command": "echo orig"}'}}
        ]
    )
    p.queue(content="done")
    h = make_harness(p, tools=default_tools(), hooks=[on_start, on_end, rewrite, annotate])
    s = h.start_session("u1")
    r = h.run_turn(s, "go")

    assert ("before_turn", "go") in events
    assert ("after_turn", "ok") in events
    assert ("before_tool", "Bash") in events
    tool_turns = [t for t in h.repo.turns if t.role == "tool"]
    assert "rewritten" in tool_turns[0].content["content"]
    assert "orig" not in tool_turns[0].content["content"]
    assert "[audited]" in tool_turns[0].content["content"]
    assert r.status == "ok"
    h.close()
